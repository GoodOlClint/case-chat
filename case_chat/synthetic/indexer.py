"""Build the synthetic raw-document index in Qdrant.

Collects chunks from every per-type loader (an explicit allowlist of source
globs — ground-truth JSON and generator/meta files are unreachable), embeds
each chunk's text **bare**, and upserts into the ``synthetic_corpus`` collection
with a payload carrying ``source_type``/``source_path``/``title`` for citation
and filtering.

Run:  uv run python -m case_chat.synthetic.indexer [--recreate] [--limit N]
"""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from case_chat.config import COLLECTION_SYNTHETIC, settings
from case_chat.embeddings.client import EmbeddingClient
from case_chat.synthetic import loaders
from case_chat.synthetic.loaders import Chunk

logger = logging.getLogger(__name__)

_POINT_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "case-chat.synthetic")
UPSERT_BATCH = 128


def collect_chunks(root: Path) -> list[Chunk]:
    """Load every allowed source file into chunks. Ground-truth never touched."""
    chunks: list[Chunk] = []

    def rel(p: Path) -> str:
        return str(p.relative_to(root))

    for source_type, glob in loaders.SOURCE_GLOBS.items():
        for path in sorted(root.glob(glob)):
            r = rel(path)
            if source_type == "apple_note":
                chunks += loaders.load_apple_note(path, r)
            elif source_type == "attachment_ocr":
                chunks += loaders.load_attachment_ocr(path, r)
            elif source_type == "email":
                chunks += loaders.load_email(path, r)
            elif source_type == "court_document":
                chunks += loaders.load_court_document(path, r)
            elif source_type == "messages":
                chunks += loaders.load_messages_rsmf(path, r)
            elif source_type in ("transcript", "witness_statement"):
                chunks += loaders.load_text_doc(path, r, source_type)

    # Visitation log: CSV + schema sidecar (handled specially).
    csv_path = root / "structured-data" / "supervised-visitation-log.csv"
    schema_path = root / "structured-data" / "supervised-visitation-log.schema.yaml"
    if csv_path.exists() and schema_path.exists():
        chunks += loaders.load_visitation_log(csv_path, schema_path, rel(csv_path))

    return chunks


def ensure_collection(client: QdrantClient, *, recreate: bool) -> None:
    name = COLLECTION_SYNTHETIC
    if client.collection_exists(name) and recreate:
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=settings.embeddings_dim, distance=qm.Distance.COSINE),
        )
        client.create_payload_index(name, field_name="source_type", field_schema=qm.PayloadSchemaType.KEYWORD)
        logger.info("created collection %s", name)


def _payload(chunk: Chunk) -> dict:
    payload = {
        "chunk_id": chunk.chunk_id,
        "source_type": chunk.source_type,
        "source_path": chunk.source_path,
        "title": chunk.title,
        "text": chunk.text,
        "text_preview": chunk.text[:300],
    }
    payload.update({k: v for k, v in chunk.metadata.items() if v is not None})
    return {k: v for k, v in payload.items() if v is not None}


def index(*, recreate: bool = False, limit: int | None = None) -> dict[str, int]:
    root = Path(settings.synthetic_corpus_path)
    if not root.exists():
        raise FileNotFoundError(f"synthetic corpus not found: {root}")

    chunks = collect_chunks(root)
    if limit:
        chunks = chunks[:limit]

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.source_type] = by_type.get(c.source_type, 0) + 1
    logger.info("collected %d chunks: %s", len(chunks), by_type)

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = EmbeddingClient()
    ensure_collection(client, recreate=recreate)

    for i in range(0, len(chunks), UPSERT_BATCH):
        batch = chunks[i : i + UPSERT_BATCH]
        vectors = embedder.embed_texts([c.text for c in batch])
        points = [
            qm.PointStruct(
                id=str(uuid.uuid5(_POINT_NS, c.chunk_id)),
                vector=vec,
                payload=_payload(c),
            )
            for c, vec in zip(batch, vectors)
        ]
        client.upsert(collection_name=COLLECTION_SYNTHETIC, points=points, wait=True)
        logger.info("upserted %d/%d", min(i + UPSERT_BATCH, len(chunks)), len(chunks))

    return by_type


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Index the synthetic raw-doc corpus into Qdrant")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    by_type = index(recreate=args.recreate, limit=args.limit)
    logger.info("DONE: %s", by_type)


if __name__ == "__main__":
    main()
