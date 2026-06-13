"""Assemble the FULL text of a domain-knowledge source for the citation viewer.

A retrieved hit is a single chunk — a behavioral-pattern card is split across
~14 chunks, a statute/opinion across several. When the user clicks a domain
citation we re-assemble the whole source by its grouping key (``card_id`` for
patterns, ``citation`` for law) so they see the complete material, not one line.

This only surfaces domain *text* already in the index (no file access); it stays
within the "domain knowledge isn't a browsable file" boundary.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from case_chat.config import (
    COLLECTION_BEHAVIORAL_PATTERNS,
    COLLECTION_LAW,
    settings,
)

# kind -> (collection, grouping field, title field, extra metadata fields)
_PASSAGE_SPECS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "pattern": (COLLECTION_BEHAVIORAL_PATTERNS, "card_id", "card_name", ("framework", "wing")),
    "law": (COLLECTION_LAW, "citation", "title", ("jurisdiction", "doc_type", "court", "date_decided")),
}


def _assemble(rows: list[dict[str, Any]], title_field: str, meta_fields: tuple[str, ...],
              key: str) -> dict[str, Any] | None:
    """Order chunks by chunk_index and join their text into one passage."""
    if not rows:
        return None
    ordered = sorted(rows, key=lambda r: r.get("chunk_index") or 0)
    text = "\n\n".join(r.get("text", "") for r in ordered if r.get("text"))
    head = ordered[0]
    return {
        "key": key,
        "title": head.get(title_field) or key,
        "text": text,
        "meta": {m: head.get(m) for m in meta_fields if head.get(m) is not None},
        "chunk_count": len(ordered),
    }


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def get_passage(kind: str, key: str) -> dict[str, Any] | None:
    spec = _PASSAGE_SPECS.get(kind)
    if not spec or not key:
        return None
    collection, field, title_field, meta_fields = spec
    points, _ = _client().scroll(
        collection,
        scroll_filter=qm.Filter(must=[qm.FieldCondition(key=field, match=qm.MatchValue(value=key))]),
        limit=300,
        with_payload=True,
    )
    result = _assemble([dict(p.payload or {}) for p in points], title_field, meta_fields, key)
    if result is not None:
        result["kind"] = kind
    return result
