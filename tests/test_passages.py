"""Tests for the domain-passage assembler (full card/statute from chunks)."""

from __future__ import annotations

from case_chat.web.passages import _assemble


def test_assemble_orders_and_joins_chunks() -> None:
    rows = [
        {"chunk_index": 2, "text": "third", "card_name": "Coercive Control", "framework": "Stark"},
        {"chunk_index": 0, "text": "first", "card_name": "Coercive Control", "framework": "Stark"},
        {"chunk_index": 1, "text": "second", "card_name": "Coercive Control", "framework": "Stark"},
    ]
    out = _assemble(rows, "card_name", ("framework",), "coercive-control")
    assert out["title"] == "Coercive Control"
    assert out["text"] == "first\n\nsecond\n\nthird"  # ordered by chunk_index
    assert out["meta"] == {"framework": "Stark"}
    assert out["chunk_count"] == 3


def test_assemble_empty_is_none() -> None:
    assert _assemble([], "card_name", (), "x") is None


def test_assemble_skips_blank_text() -> None:
    rows = [{"chunk_index": 0, "text": "body"}, {"chunk_index": 1, "text": ""}]
    out = _assemble(rows, "title", (), "k")
    assert out["text"] == "body"
    assert out["chunk_count"] == 2  # counts all chunks, joins only non-empty
