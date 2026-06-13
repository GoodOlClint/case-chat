"""Retrieval tool logic behind the MCP surface.

Pure-ish functions (shared lazy singletons for Qdrant + the SQLite dataset) so
they're unit-testable without the MCP runtime. ``server.py`` wraps these as
MCP tools. Every tool returns plain JSON-able dicts/lists.

Two access modes:
- semantic ``*.search`` — cosine kNN over a Qdrant collection;
- exact ``*.lookup`` — payload scroll (no embedding) for a named reference /
  citation (e.g. "James 2:3", "Ark. Code Ann. § 9-13-101").
"""

from __future__ import annotations

import re
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from case_chat.casedata.queries import CaseDataset
from case_chat.config import (
    COLLECTION_BEHAVIORAL_PATTERNS,
    COLLECTION_BEHAVIORAL_SOURCES,
    COLLECTION_LAW,
    COLLECTION_PROFESSIONAL_STANDARDS,
    COLLECTION_SCRIPTURE,
    COLLECTION_SYNTHETIC,
    settings,
)
from case_chat.retrieval.corpus import QdrantRetriever, partition_by_binding

# -- shared lazy singletons -------------------------------------------------
_retriever: QdrantRetriever | None = None
_client: QdrantClient | None = None
_dataset: CaseDataset | None = None


def retriever() -> QdrantRetriever:
    global _retriever
    if _retriever is None:
        _retriever = QdrantRetriever()
    return _retriever


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return _client


def dataset() -> CaseDataset:
    global _dataset
    if _dataset is None:
        _dataset = CaseDataset()
    return _dataset


def _hit_view(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {f: payload.get(f) for f in fields if payload.get(f) is not None}


# -- law --------------------------------------------------------------------
_LAW_FIELDS = ("citation", "title", "case_name", "court", "date_decided", "doc_type",
               "jurisdiction", "chunk_type", "section_id", "source_url", "path")


def law_search(
    query: str, *, jurisdiction: str | None = None, doc_type: str | None = None, limit: int = 8
) -> dict[str, Any]:
    """Semantic search over statutes + caselaw + constitutions (all jurisdictions).

    jurisdiction ∈ {ar, federal, ...} or None for all; doc_type ∈
    {statute, opinion, constitution, ...} or None. Hits are labeled
    binding/persuasive/non-authority for the active jurisdiction.
    """
    where: dict[str, Any] = {}
    if jurisdiction and jurisdiction.lower() != "both":
        where["jurisdiction"] = jurisdiction.lower()
    if doc_type:
        where["doc_type"] = doc_type.lower()
    hits = retriever().search(COLLECTION_LAW, query, limit=limit, where=where or None)
    part = partition_by_binding(hits, settings.active_jurisdiction)
    klass = {id(h): "binding" for h in part.binding}
    klass.update({id(h): "persuasive" for h in part.persuasive})
    klass.update({id(h): "non_authority" for h in part.non_authority})
    return {
        "active_jurisdiction": part.active_jurisdiction,
        "binding_count": len(part.binding),
        "persuasive_count": len(part.persuasive),
        "non_authority_count": len(part.non_authority),
        "hits": [
            {
                "binding_class": klass[id(h)],
                "score": round(h.score, 4),
                "text": h.payload.get("text"),
                **_hit_view(h.payload, _LAW_FIELDS),
            }
            for h in hits
        ],
    }


def law_lookup(citation: str, *, jurisdiction: str | None = None, limit: int = 10) -> dict[str, Any]:
    """Exact statute/section/case lookup by citation (payload scroll, no kNN)."""
    must = [qm.FieldCondition(key="citation", match=qm.MatchValue(value=citation))]
    if jurisdiction:
        must.append(qm.FieldCondition(key="jurisdiction", match=qm.MatchValue(value=jurisdiction.lower())))
    points, _ = client().scroll(
        COLLECTION_LAW, scroll_filter=qm.Filter(must=must), limit=limit, with_payload=True
    )
    rows = sorted((p.payload or {} for p in points), key=lambda p: p.get("chunk_index") or 0)
    return {
        "citation": citation,
        "match_count": len(rows),
        "passages": [{"text": r.get("text"), **_hit_view(r, _LAW_FIELDS)} for r in rows],
    }


# -- scripture --------------------------------------------------------------
_REF = re.compile(r"^\s*(.*?)\s+(\d+)(?::(\d+)(?:\s*-\s*(\d+))?)?\s*$")
_SCRIPTURE_FIELDS = ("translation", "book", "chapter", "verse", "reference")


def parse_reference(reference: str) -> dict[str, Any] | None:
    """Parse 'James 2:3' / 'James 2:1-4' / 'James 2' / '1 Corinthians 13'."""
    m = _REF.match(reference)
    if not m:
        return None
    book, chapter, v1, v2 = m.groups()
    return {
        "book": book.strip(),
        "chapter": int(chapter),
        "verse_start": int(v1) if v1 else None,
        "verse_end": int(v2) if v2 else (int(v1) if v1 else None),
    }


def scripture_search(
    query: str, *, translation: str | None = None, book: str | None = None, limit: int = 8
) -> dict[str, Any]:
    """Semantic search over scripture (KJV + WEB)."""
    where: dict[str, Any] = {}
    if translation:
        where["translation"] = translation.lower()
    if book:
        where["book"] = book
    hits = retriever().search(COLLECTION_SCRIPTURE, query, limit=limit, where=where or None)
    return {
        "hits": [
            {"score": round(h.score, 4), "text": h.payload.get("text"),
             **_hit_view(h.payload, _SCRIPTURE_FIELDS)}
            for h in hits
        ]
    }


def scripture_lookup(reference: str, *, translation: str | None = None) -> dict[str, Any]:
    """Exact verse lookup by reference (payload scroll). translation None → all."""
    ref = parse_reference(reference)
    if not ref:
        return {"reference": reference, "error": "could not parse reference", "verses": []}
    must = [
        qm.FieldCondition(key="book", match=qm.MatchValue(value=ref["book"])),
        qm.FieldCondition(key="chapter", match=qm.MatchValue(value=ref["chapter"])),
    ]
    if translation:
        must.append(qm.FieldCondition(key="translation", match=qm.MatchValue(value=translation.lower())))
    if ref["verse_start"] is not None:
        must.append(qm.FieldCondition(
            key="verse", range=qm.Range(gte=ref["verse_start"], lte=ref["verse_end"])
        ))
    points, _ = client().scroll(
        COLLECTION_SCRIPTURE, scroll_filter=qm.Filter(must=must), limit=200, with_payload=True
    )
    rows = sorted(
        (p.payload or {} for p in points),
        key=lambda p: (p.get("translation") or "", p.get("verse") or 0),
    )
    return {
        "reference": reference,
        "match_count": len(rows),
        "verses": [{"text": r.get("text"), **_hit_view(r, _SCRIPTURE_FIELDS)} for r in rows],
    }


# -- behavioral / standards / synthetic ------------------------------------
def pattern_search(query: str, *, wing: str | None = None, limit: int = 8) -> dict[str, Any]:
    """Semantic search over behavioral-pattern cards (coercive control, etc.)."""
    where = {"wing": wing} if wing else None
    hits = retriever().search(COLLECTION_BEHAVIORAL_PATTERNS, query, limit=limit, where=where)
    fields = ("card_id", "card_name", "framework", "category", "wing", "source_url")
    return {"hits": [{"score": round(h.score, 4), "text": h.payload.get("text"),
                      **_hit_view(h.payload, fields)} for h in hits]}


def behavioral_source_search(query: str, *, limit: int = 8) -> dict[str, Any]:
    """Semantic search over behavioral framework source notes / papers."""
    hits = retriever().search(COLLECTION_BEHAVIORAL_SOURCES, query, limit=limit)
    fields = ("title", "source_type", "citation", "doi", "note_id", "source_url")
    return {"hits": [{"score": round(h.score, 4), "text": h.payload.get("text"),
                      **_hit_view(h.payload, fields)} for h in hits]}


def standards_search(query: str, *, limit: int = 8) -> dict[str, Any]:
    """Semantic search over professional ethics / practice standards."""
    hits = retriever().search(COLLECTION_PROFESSIONAL_STANDARDS, query, limit=limit)
    fields = ("standard_id", "name", "citation", "issuing_authority", "authority_type",
              "profession", "source_url")
    return {"hits": [{"score": round(h.score, 4), "text": h.payload.get("text"),
                      **_hit_view(h.payload, fields)} for h in hits]}


def corpus_search(query: str, *, source_type: str | None = None, limit: int = 8) -> dict[str, Any]:
    """Semantic search over the raw case documents (notes, emails, messages, etc.)."""
    where = {"source_type": source_type} if source_type else None
    hits = retriever().search(COLLECTION_SYNTHETIC, query, limit=limit, where=where)
    fields = ("source_type", "source_path", "title")
    return {"hits": [{"score": round(h.score, 4), "text": h.payload.get("text"),
                      **_hit_view(h.payload, fields)} for h in hits]}


# -- structured fake-case ---------------------------------------------------
def case_timeline(**kw: Any) -> dict[str, Any]:
    return {"events": dataset().timeline_query(**kw)}


def case_entity(name: str, *, limit: int = 5) -> dict[str, Any]:
    return {"entities": dataset().entity_lookup(name, limit=limit)}


def case_overview() -> dict[str, Any]:
    return dataset().case_overview()


def case_participants(*, role: str | None = None, entity_type: str | None = None) -> dict[str, Any]:
    return {"participants": dataset().list_participants(role=role, entity_type=entity_type)}


def case_facts(**kw: Any) -> dict[str, Any]:
    return {"facts": dataset().facts_query(**kw)}


def case_flags(**kw: Any) -> dict[str, Any]:
    return {"flags": dataset().flags_query(**kw)}


def case_observations(**kw: Any) -> dict[str, Any]:
    return {"observations": dataset().observations_query(**kw)}
