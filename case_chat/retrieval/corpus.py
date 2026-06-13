"""Qdrant-backed retrieval + binding/persuasive authority partitioning.

Replaces case-project's pgvector ``LegalCorpusRetriever``. Cosine kNN over
Qdrant collections; payload filters stand in for the old typed metadata
columns. ``partition_by_binding`` is pure app-side logic driven by each hit's
``binding_jurisdictions`` payload and the active jurisdiction (default ``ar``,
a different value for multi-state later — no re-embed needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from case_chat.config import settings
from case_chat.embeddings.client import EmbeddingClient


@dataclass(frozen=True)
class Hit:
    score: float
    payload: dict[str, Any]
    id: str | int | None = None


@dataclass(frozen=True)
class PartitionedHits:
    """Hits split by authority for the active jurisdiction."""

    active_jurisdiction: str
    binding: list[Hit] = field(default_factory=list)
    persuasive: list[Hit] = field(default_factory=list)
    non_authority: list[Hit] = field(default_factory=list)


def build_filter(where: dict[str, Any] | None) -> qm.Filter | None:
    """Build a Qdrant ``must``-filter from a flat field→value dict.

    Scalar value → exact match. List/tuple/set value → match-any. ``None``
    values are skipped (lets callers pass optional filters unconditionally).
    Matching a scalar against an array payload field (e.g. ``binding_jurisdictions``)
    tests membership, per Qdrant semantics.
    """
    if not where:
        return None
    conditions: list[qm.Condition] = []
    for key, value in where.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            conditions.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=list(value))))
        else:
            conditions.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
    return qm.Filter(must=conditions) if conditions else None


def partition_by_binding(
    hits: list[Hit],
    active_jurisdiction: str | None = None,
) -> PartitionedHits:
    """Split hits into binding / persuasive / non-authority.

    A hit is **binding** when the active jurisdiction is in its
    ``binding_jurisdictions`` payload, or that list contains ``federal``
    (federal authority binds state courts). It is **persuasive** when the list
    is non-empty but doesn't satisfy binding, and **non-authority** when the
    list is empty/missing (e.g. behavioral-pattern cards, scripture).
    """
    active = (active_jurisdiction or settings.active_jurisdiction or "").lower()
    binding: list[Hit] = []
    persuasive: list[Hit] = []
    non_authority: list[Hit] = []
    for hit in hits:
        raw = hit.payload.get("binding_jurisdictions")
        if not isinstance(raw, (list, tuple)) or not raw:
            non_authority.append(hit)
            continue
        bj = {str(j).lower() for j in raw}
        if active in bj or "federal" in bj:
            binding.append(hit)
        else:
            persuasive.append(hit)
    return PartitionedHits(active, binding, persuasive, non_authority)


class QdrantRetriever:
    """Cosine-kNN retrieval over Qdrant collections."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        self._client = client or QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
        self._embedder = embedder or EmbeddingClient()

    def search(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Embed ``query`` and return the top ``limit`` hits from ``collection``."""
        vector = self._embedder.embed_query(query)
        response = self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            query_filter=build_filter(where),
            search_params=qm.SearchParams(hnsw_ef=settings.qdrant_search_hnsw_ef),
            with_payload=True,
        )
        return [
            Hit(score=float(p.score), payload=dict(p.payload or {}), id=p.id)
            for p in response.points
        ]
