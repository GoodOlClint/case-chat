"""Query- and document-side embedding for case-chat retrieval.

This is the build/query parity seam, ported from case-project's
``legal_corpus/embeddings.py`` with the Athena dependency removed. The
*contract* is unchanged and load-bearing:

    Qwen/Qwen3-Embedding-4B   (2560-dim, cosine, L2-normalized)

ASYMMETRIC RETRIEVAL — queries are wrapped with the Qwen3 instruction
prefix (:data:`QWEN3_QUERY_TASK`); documents are embedded bare. This must
match the side that builds the index. Both the re-embedded domain corpus
and the synthetic index in this repo embed documents bare through
:func:`embed_documents`, so parity is true by construction here.

Embeddings are fetched from an OpenAI-compatible ``/embeddings`` endpoint
(TEI serving Qwen3-Embedding-4B on the 5090, or a local fallback server on
the Mac). The payload shape is OpenAI-standard, so swapping servers is a
config change. We L2-normalize client-side regardless of what the server
returns, so the cosine contract holds even if a backend forgets to
normalize — cheap insurance against silent rank corruption.
"""

from __future__ import annotations

import logging
import time

import httpx
import numpy as np

from case_chat.config import settings

logger = logging.getLogger(__name__)


# Qwen3 retrieval-task description, wrapped around each QUERY before
# embedding (documents are embedded bare). Pinned here so the contract lives
# in code. Kept identical to case-project / domain-knowledge so that vectors
# built there and queried here align. Changing the wording shifts the query
# vector measurably — keep it in lockstep with the build side.
QWEN3_QUERY_TASK = (
    "Given a legal-research query, retrieve relevant statutes, "
    "case-law passages, and behavioral-pattern card text that answer it"
)


def _wrap_qwen3_query(text: str) -> str:
    """Apply the Qwen3 asymmetric-retrieval instruction prefix."""
    return f"Instruct: {QWEN3_QUERY_TASK}\nQuery: {text}"


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector; defensive against an un-normalizing backend."""
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise RuntimeError("embedding has zero norm — backend returned a null vector")
    return (arr / norm).tolist()


class EmbeddingClient:
    """Thin OpenAI-/embeddings client honoring the Qwen3-4B/2560 contract.

    Stateless apart from an ``httpx.Client``; safe to construct once and
    reuse. ``embed_query`` wraps with the instruction prefix; ``embed_texts``
    embeds bare (document side). Every returned vector is L2-normalized and
    length-checked against :data:`settings.embeddings_dim`.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        timeout: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = (base_url or settings.embeddings_base_url).rstrip("/")
        self._api_key = api_key if api_key is not None else settings.embeddings_api_key
        self._model = model or settings.embeddings_model
        self._dim = dim or settings.embeddings_dim
        self._timeout = timeout or settings.embeddings_timeout_secs
        self._max_chars = settings.embeddings_max_chars
        self._max_retries = settings.embeddings_max_retries
        self._client = client or httpx.Client(timeout=self._timeout)

    @property
    def dim(self) -> int:
        return self._dim

    def _headers(self) -> dict[str, str]:
        if self._api_key and self._api_key != "EMPTY":
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    def _post(self, inputs: list[str]) -> list[list[float]]:
        # Defensive truncation: a handful of anomalous corpus chunks exceed the
        # embedder's context and would 400 the whole batch.
        inputs = [t[: self._max_chars] for t in inputs]
        try:
            data = self._post_with_retry(inputs)
        except httpx.HTTPStatusError as exc:
            # A multi-item batch rejected with 400: isolate the offender so one
            # bad input can't sink its siblings. A lone item that still 400s is a
            # real error and propagates.
            if exc.response.status_code == 400 and len(inputs) > 1:
                out: list[list[float]] = []
                for item in inputs:
                    out.extend(self._post([item]))
                return out
            raise
        items = data.get("data") or []
        if len(items) != len(inputs):
            raise RuntimeError(
                f"/embeddings returned {len(items)} vectors for {len(inputs)} inputs "
                f"(model={self._model})"
            )
        # The OpenAI contract does not promise input order is preserved; sort
        # by the returned index to be safe.
        items = sorted(items, key=lambda it: it.get("index", 0))
        out: list[list[float]] = []
        for it in items:
            emb = it.get("embedding")
            if not emb:
                raise RuntimeError(f"/embeddings returned an empty vector (model={self._model})")
            if len(emb) != self._dim:
                raise RuntimeError(
                    f"embedding dim mismatch: got {len(emb)}, expected {self._dim}. "
                    "Wrong embedding model loaded — this silently corrupts ranks."
                )
            out.append(_l2_normalize(emb))
        return out

    def _post_with_retry(self, inputs: list[str]) -> dict:
        """POST with retry on transient failures (timeouts, transport, 5xx).

        Long background builds outlast brief embedder hiccups (model reload,
        momentary overload). 4xx other than via the caller's 400-split are
        non-retryable and propagate immediately.
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.post(
                    f"{self._base_url}/embeddings",
                    json={"input": inputs, "model": self._model},
                    headers=self._headers(),
                    timeout=self._timeout,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise  # client error (incl. 400) — let the caller decide
                last_exc = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
            if attempt < self._max_retries - 1:
                time.sleep(2 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string (instruction-wrapped, L2-normalized)."""
        if not text or not text.strip():
            raise ValueError("embed_query requires non-empty text")
        return self._post([_wrap_qwen3_query(text)])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed documents BARE in batches (build side); L2-normalized."""
        if not texts:
            return []
        if any(not (t and t.strip()) for t in texts):
            raise ValueError("embed_texts requires every text to be non-empty")
        out: list[list[float]] = []
        batch = settings.embeddings_batch_size
        for i in range(0, len(texts), batch):
            out.extend(self._post(texts[i : i + batch]))
        return out

    def close(self) -> None:
        self._client.close()


# Module-level convenience wrappers over a lazily-built default client.
_default_client: EmbeddingClient | None = None


def _get_default_client() -> EmbeddingClient:
    global _default_client
    if _default_client is None:
        _default_client = EmbeddingClient()
    return _default_client


def embed_query(text: str) -> list[float]:
    """Embed a single query for retrieval (wrapped + L2-normalized)."""
    return _get_default_client().embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed documents bare for indexing (L2-normalized)."""
    return _get_default_client().embed_texts(texts)
