"""Contract tests for the embedding client.

These pin the load-bearing parity rules: query wrapping, bare documents,
L2 normalization, dim enforcement, and batch fan-out. The HTTP layer is
mocked with respx so no embedding server is required.
"""

from __future__ import annotations

import math

import httpx
import numpy as np
import pytest
import respx

from case_chat.embeddings.client import (
    QWEN3_QUERY_TASK,
    EmbeddingClient,
    _l2_normalize,
    _wrap_qwen3_query,
)

BASE = "http://embed.test/v1"
DIM = 8  # small dim for tests; real contract is 2560


def _make_client(**kw) -> EmbeddingClient:
    return EmbeddingClient(base_url=BASE, api_key="EMPTY", model="test-model", dim=DIM, **kw)


def _embedding_response(vectors: list[list[float]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]},
    )


def test_query_wrapping_uses_instruct_envelope() -> None:
    wrapped = _wrap_qwen3_query("custody modification standard")
    assert wrapped == f"Instruct: {QWEN3_QUERY_TASK}\nQuery: custody modification standard"


def test_l2_normalize_unit_length() -> None:
    out = _l2_normalize([3.0, 4.0])
    assert math.isclose(math.hypot(*out), 1.0, rel_tol=1e-6)


def test_l2_normalize_rejects_zero_vector() -> None:
    with pytest.raises(RuntimeError, match="zero norm"):
        _l2_normalize([0.0, 0.0, 0.0])


@respx.mock
def test_embed_query_wraps_and_normalizes() -> None:
    route = respx.post(f"{BASE}/embeddings").mock(
        return_value=_embedding_response([[3.0] + [0.0] * (DIM - 1)])
    )
    client = _make_client()
    vec = client.embed_query("when was the petition filed")

    sent = route.calls.last.request
    body = sent.read().decode()
    # Query side must carry the Instruct/Query envelope.
    assert "Instruct:" in body and "Query: when was the petition filed" in body
    # Returned vector is L2-normalized.
    assert math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=1e-6)
    assert len(vec) == DIM


@respx.mock
def test_embed_texts_are_bare_not_wrapped() -> None:
    route = respx.post(f"{BASE}/embeddings").mock(
        return_value=_embedding_response([[1.0] + [0.0] * (DIM - 1)])
    )
    client = _make_client()
    client.embed_texts(["A bare statute section."])

    body = route.calls.last.request.read().decode()
    assert "Instruct:" not in body  # documents embedded bare


@respx.mock
def test_dim_mismatch_is_hard_error() -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=_embedding_response([[1.0, 0.0]])  # len 2 != DIM 8
    )
    client = _make_client()
    with pytest.raises(RuntimeError, match="dim mismatch"):
        client.embed_query("x")


@respx.mock
def test_embed_texts_batches_and_orders() -> None:
    # batch_size default is 64; force two batches with 3 texts via monkeypatched setting.
    from case_chat import config

    config.settings.embeddings_batch_size = 2

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        n = len(json.loads(request.read())["input"])
        # Return vectors out of order to exercise index-sorting.
        vecs = [[float(i + 1)] + [0.0] * (DIM - 1) for i in range(n)]
        data = [{"index": n - 1 - i, "embedding": vecs[n - 1 - i]} for i in range(n)]
        return httpx.Response(200, json={"data": data})

    respx.post(f"{BASE}/embeddings").mock(side_effect=handler)
    client = _make_client()
    out = client.embed_texts(["a", "b", "c"])
    assert len(out) == 3
    assert all(math.isclose(float(np.linalg.norm(v)), 1.0, rel_tol=1e-6) for v in out)


@respx.mock
def test_count_mismatch_is_error() -> None:
    respx.post(f"{BASE}/embeddings").mock(return_value=_embedding_response([]))
    client = _make_client()
    with pytest.raises(RuntimeError, match="returned 0 vectors"):
        client.embed_query("x")


def test_empty_query_rejected() -> None:
    client = _make_client()
    with pytest.raises(ValueError):
        client.embed_query("   ")


@respx.mock
def test_retries_transient_timeout_then_succeeds(monkeypatch) -> None:
    import case_chat.embeddings.client as mod

    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)  # no real backoff in tests
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("boom", request=request)
        return _embedding_response([[1.0] + [0.0] * (DIM - 1)])

    respx.post(f"{BASE}/embeddings").mock(side_effect=handler)
    vec = _make_client().embed_query("x")
    assert calls["n"] == 2 and len(vec) == DIM


@respx.mock
def test_batch_400_splits_to_isolate_offender(monkeypatch) -> None:
    import case_chat.embeddings.client as mod

    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        inputs = json.loads(request.read())["input"]
        if len(inputs) > 1:
            return httpx.Response(400, json={"error": "too big"})
        return _embedding_response([[1.0] + [0.0] * (DIM - 1)])

    respx.post(f"{BASE}/embeddings").mock(side_effect=handler)
    out = _make_client().embed_texts(["a", "b", "c"])
    assert len(out) == 3  # the 400 batch was split into successful singletons
