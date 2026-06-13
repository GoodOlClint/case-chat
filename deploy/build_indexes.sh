#!/usr/bin/env bash
# One-time index build for the cloud stack (run via the `build` profile):
#   docker compose --profile build run --rm builder
#
# Restores the knowledgedb dump into the staging Postgres, re-embeds the whole
# domain corpus into Qdrant (via TEI), indexes the synthetic raw docs, and
# builds the fake-case SQLite. Idempotent (each step --recreate / rebuilds).
set -euo pipefail

PGHOST="${PGHOST:-pg-staging}"
DUMP="${DUMP_PATH_IN:-/dump/knowledgedb.dump}"

echo "==> waiting for pg-staging…"
until PGPASSWORD=postgres pg_isready -h "$PGHOST" -U postgres >/dev/null 2>&1; do sleep 2; done

echo "==> restoring knowledgedb dump (text + metadata; vectors regenerated)…"
PGPASSWORD=postgres pg_restore --no-owner --disable-triggers -j 4 \
  -h "$PGHOST" -U postgres -d knowledgedb "$DUMP" || true

echo "==> re-embedding domain corpus into Qdrant (full, incl. scripture)…"
python -m case_chat.knowledgedb.reembed --recreate

echo "==> indexing synthetic raw-doc corpus…"
python -m case_chat.synthetic.indexer --recreate

echo "==> building fake-case SQLite dataset…"
python -m case_chat.casedata.dataset

echo "==> index build complete."
