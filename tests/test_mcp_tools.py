"""Pure unit tests for MCP tool helpers (no live services)."""

from __future__ import annotations

import pytest

from case_chat.mcp_server.tools import parse_reference


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("James 2:3", {"book": "James", "chapter": 2, "verse_start": 3, "verse_end": 3}),
        ("James 2:1-4", {"book": "James", "chapter": 2, "verse_start": 1, "verse_end": 4}),
        ("James 2", {"book": "James", "chapter": 2, "verse_start": None, "verse_end": None}),
        ("1 Corinthians 13:4", {"book": "1 Corinthians", "chapter": 13, "verse_start": 4, "verse_end": 4}),
        ("1 John 4", {"book": "1 John", "chapter": 4, "verse_start": None, "verse_end": None}),
        ("Song of Solomon 2:1", {"book": "Song of Solomon", "chapter": 2, "verse_start": 1, "verse_end": 1}),
    ],
)
def test_parse_reference_ok(ref, expected) -> None:
    assert parse_reference(ref) == expected


@pytest.mark.parametrize("ref", ["", "garbage", "John", "just words no numbers"])
def test_parse_reference_unparseable(ref) -> None:
    assert parse_reference(ref) is None
