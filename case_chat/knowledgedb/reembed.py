"""Re-embed the domain-knowledge corpus into Qdrant.

The knowledgedb pg_dump's chunk *text + metadata* are good; its *vectors* are
invalid (Athena embedding bug) and are never read. This pipeline streams rows
from the staged Postgres, re-embeds ``full_text`` **bare** (document side of the
Qwen3 asymmetric contract — see ``case_chat.embeddings``), and upserts into
Qdrant collections organized by content domain ([ADR 0002]).

Source tables fold into collections:

    family_law, constitutional        -> law           (doc_type from chunk_type)
    behavioral_patterns               -> behavioral_patterns
    behavioral_sources                -> behavioral_sources
    professional_standards            -> professional_standards
    bible_kjv, bible_web              -> scripture      (translation in payload)

Identifier fields needed by the exact-lookup tools are carried into the payload:
``citation`` (law), ``book``/``chapter``/``verse``/``reference`` (scripture).

Run:  uv run python -m case_chat.knowledgedb.reembed [--recreate] [--skip-bibles]
      [--limit N] [--tables family_law,...]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from case_chat.config import (
    COLLECTION_BEHAVIORAL_PATTERNS,
    COLLECTION_BEHAVIORAL_SOURCES,
    COLLECTION_LAW,
    COLLECTION_PROFESSIONAL_STANDARDS,
    COLLECTION_SCRIPTURE,
    settings,
)
from case_chat.embeddings.client import EmbeddingClient

logger = logging.getLogger(__name__)

# Stable namespace so chunk_id -> point-id is deterministic across re-runs
# (Qdrant point ids must be uint64 or UUID; chunk_id is a string).
_POINT_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "case-chat.knowledgedb")

READ_BATCH = 256  # rows fetched + embedded + upserted per cycle


def _point_id(table: str, chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NS, f"{table}:{chunk_id}"))


def _jsonable(value: Any) -> Any:
    """Coerce DB values into Qdrant/JSON-safe payload values."""
    if value is None:
        return None
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _doc_type_from_chunk_type(chunk_type: str | None) -> str:
    """Derive the law `doc_type` filter value from the source `chunk_type`."""
    ct = (chunk_type or "").lower()
    if "statute" in ct:
        return "statute"
    if "constitution" in ct:
        return "constitution"
    if "opinion" in ct or "caselaw" in ct or "case" in ct:
        return "opinion"
    if "note" in ct:
        return "note"
    return ct or "unknown"


def _base_payload(row: dict[str, Any], table: str, *, keep: list[str]) -> dict[str, Any]:
    """Common payload: chunk_id, source table, full text, and selected fields."""
    payload: dict[str, Any] = {
        "chunk_id": row["chunk_id"],
        "source_table": table,
        "chunk_index": row.get("chunk_index"),
        "text": row.get("full_text"),
        "text_preview": row.get("text_preview"),
        "path": row.get("path"),
        "source_url": row.get("source_url"),
    }
    for k in keep:
        payload[k] = _jsonable(row.get(k))
    return {k: v for k, v in payload.items() if v is not None}


def _law_payload(row: dict[str, Any], table: str) -> dict[str, Any]:
    payload = _base_payload(
        row,
        table,
        keep=[
            "chunk_type", "citation", "title", "case_name", "court", "date_decided",
            "jurisdiction", "binding_jurisdictions", "is_precedential", "topics",
            "section_id", "article",
        ],
    )
    payload["doc_type"] = _doc_type_from_chunk_type(row.get("chunk_type"))
    payload.setdefault("binding_jurisdictions", [])
    return payload


def _scripture_payload(row: dict[str, Any], table: str) -> dict[str, Any]:
    return _base_payload(
        row,
        table,
        keep=[
            "translation", "book", "book_order", "testament", "chapter",
            "verse", "reference",
        ],
    )


def _patterns_payload(row: dict[str, Any], table: str) -> dict[str, Any]:
    return _base_payload(
        row, table,
        keep=["card_id", "card_name", "framework", "category", "wing", "binding_jurisdictions"],
    )


def _sources_payload(row: dict[str, Any], table: str) -> dict[str, Any]:
    return _base_payload(
        row, table,
        keep=["source_type", "title", "note_id", "citation", "doi", "scope"],
    )


def _standards_payload(row: dict[str, Any], table: str) -> dict[str, Any]:
    return _base_payload(
        row, table,
        keep=[
            "standard_id", "name", "citation", "issuing_authority", "authority_type",
            "profession", "scope", "binding_in_ar", "binding_jurisdictions", "topics",
        ],
    )


@dataclass(frozen=True)
class SourceSpec:
    table: str
    collection: str
    payload_builder: Callable[[dict[str, Any], str], dict[str, Any]]


SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("family_law", COLLECTION_LAW, _law_payload),
    SourceSpec("constitutional", COLLECTION_LAW, _law_payload),
    SourceSpec("behavioral_patterns", COLLECTION_BEHAVIORAL_PATTERNS, _patterns_payload),
    SourceSpec("behavioral_sources", COLLECTION_BEHAVIORAL_SOURCES, _sources_payload),
    SourceSpec("professional_standards", COLLECTION_PROFESSIONAL_STANDARDS, _standards_payload),
    SourceSpec("bible_kjv", COLLECTION_SCRIPTURE, _scripture_payload),
    SourceSpec("bible_web", COLLECTION_SCRIPTURE, _scripture_payload),
)

BIBLE_TABLES = {"bible_kjv", "bible_web"}

# Payload fields indexed per collection for fast filtering / exact-lookup scroll.
PAYLOAD_INDEXES: dict[str, dict[str, qm.PayloadSchemaType]] = {
    COLLECTION_LAW: {
        "jurisdiction": qm.PayloadSchemaType.KEYWORD,
        "doc_type": qm.PayloadSchemaType.KEYWORD,
        "binding_jurisdictions": qm.PayloadSchemaType.KEYWORD,
        "citation": qm.PayloadSchemaType.KEYWORD,
        "chunk_type": qm.PayloadSchemaType.KEYWORD,
    },
    COLLECTION_SCRIPTURE: {
        "translation": qm.PayloadSchemaType.KEYWORD,
        "book": qm.PayloadSchemaType.KEYWORD,
        "chapter": qm.PayloadSchemaType.INTEGER,
        "verse": qm.PayloadSchemaType.INTEGER,
        "reference": qm.PayloadSchemaType.KEYWORD,
    },
    COLLECTION_BEHAVIORAL_PATTERNS: {
        "wing": qm.PayloadSchemaType.KEYWORD,
        "card_id": qm.PayloadSchemaType.KEYWORD,
        "framework": qm.PayloadSchemaType.KEYWORD,
    },
    COLLECTION_BEHAVIORAL_SOURCES: {
        "source_type": qm.PayloadSchemaType.KEYWORD,
    },
    COLLECTION_PROFESSIONAL_STANDARDS: {
        "profession": qm.PayloadSchemaType.KEYWORD,
        "authority_type": qm.PayloadSchemaType.KEYWORD,
        "citation": qm.PayloadSchemaType.KEYWORD,
    },
}


def ensure_collection(client: QdrantClient, name: str, dim: int, *, recreate: bool) -> None:
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        for field_name, schema in PAYLOAD_INDEXES.get(name, {}).items():
            client.create_payload_index(name, field_name=field_name, field_schema=schema)
        logger.info("created collection %s (dim=%d)", name, dim)


def _columns_for(conn: psycopg.Connection, table: str) -> list[str]:
    """All column names except the invalid `embedding` and the bulky `metadata`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s AND column_name NOT IN ('embedding', 'metadata') "
            "ORDER BY ordinal_position",
            (table,),
        )
        return [r[0] for r in cur.fetchall()]


def reembed_table(
    conn: psycopg.Connection,
    client: QdrantClient,
    embedder: EmbeddingClient,
    spec: SourceSpec,
    *,
    limit: int | None = None,
) -> int:
    cols = _columns_for(conn, spec.table)
    if "full_text" not in cols:
        raise RuntimeError(f"{spec.table} has no full_text column")
    select_cols = ", ".join(f'"{c}"' for c in cols)
    sql = f"SELECT {select_cols} FROM {spec.table}"
    if limit:
        sql += f" LIMIT {int(limit)}"

    total = 0
    skipped = 0
    # Server-side cursor streams rows so we never hold a whole table in memory.
    with conn.cursor(name=f"reembed_{spec.table}", row_factory=dict_row) as cur:
        cur.itersize = READ_BATCH
        cur.execute(sql)
        while True:
            rows = cur.fetchmany(READ_BATCH)
            if not rows:
                break
            # Drop rows with empty full_text (e.g. WEB's omitted textual-variant
            # verses) individually, so one empty row can't sink its whole batch.
            rows = [r for r in rows if (r.get("full_text") or "").strip()]
            if not rows:
                continue
            try:
                texts = [r["full_text"] for r in rows]
                vectors = embedder.embed_texts(texts)
                points = [
                    qm.PointStruct(
                        id=_point_id(spec.table, r["chunk_id"]),
                        vector=vec,
                        payload=spec.payload_builder(r, spec.table),
                    )
                    for r, vec in zip(rows, vectors)
                ]
                client.upsert(collection_name=spec.collection, points=points, wait=True)
                total += len(points)
                logger.info("%s -> %s: %d", spec.table, spec.collection, total)
            except Exception as exc:  # keep the long build alive across transient errors
                skipped += len(rows)
                logger.warning(
                    "%s: skipped a batch of %d after error (%s); continuing",
                    spec.table, len(rows), exc,
                )
    if skipped:
        logger.warning("%s: %d rows skipped total", spec.table, skipped)
    return total


def reembed(
    *,
    recreate: bool = False,
    skip_bibles: bool = False,
    tables: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    specs = [s for s in SPECS if not (skip_bibles and s.table in BIBLE_TABLES)]
    if tables:
        specs = [s for s in specs if s.table in tables]

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = EmbeddingClient()

    # Create each target collection once (recreate clears it for a clean build).
    for coll in {s.collection for s in specs}:
        ensure_collection(client, coll, settings.embeddings_dim, recreate=recreate)

    counts: dict[str, int] = {}
    with psycopg.connect(settings.knowledgedb_staging_dsn) as conn:
        for spec in specs:
            n = reembed_table(conn, client, embedder, spec, limit=limit)
            counts[spec.table] = n
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Re-embed knowledgedb into Qdrant")
    ap.add_argument("--recreate", action="store_true", help="drop+recreate target collections")
    ap.add_argument("--skip-bibles", action="store_true", help="skip bible_kjv/bible_web")
    ap.add_argument("--tables", help="comma-separated subset of source tables")
    ap.add_argument("--limit", type=int, help="rows per table (smoke testing)")
    args = ap.parse_args()

    counts = reembed(
        recreate=args.recreate,
        skip_bibles=args.skip_bibles,
        tables=set(args.tables.split(",")) if args.tables else None,
        limit=args.limit,
    )
    total = sum(counts.values())
    for table, n in counts.items():
        logger.info("DONE %s: %d", table, n)
    logger.info("TOTAL upserted: %d", total)


if __name__ == "__main__":
    main()
