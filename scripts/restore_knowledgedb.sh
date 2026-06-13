#!/usr/bin/env bash
# Stage the knowledgedb pg_dump into the throwaway Docker Postgres so the
# re-embed pipeline can read its GOOD chunk text + metadata. The dump's vectors
# are invalid (Athena bug) and are never read — we drop the (pointless) HNSW
# indexes to speed the data load, since the re-embed regenerates all vectors.
#
# Usage: scripts/restore_knowledgedb.sh [/path/to/knowledgedb.dump]
set -euo pipefail

DUMP="${1:-$HOME/Source/domain-knowledge/dist/knowledgedb-71123c1dirty_2026-05-27T181856Z.dump}"
CONTAINER="casechat-pg-staging"

[ -f "$DUMP" ] || { echo "dump not found: $DUMP" >&2; exit 1; }

echo "==> copying dump into $CONTAINER"
docker cp "$DUMP" "$CONTAINER:/tmp/knowledgedb.dump"

echo "==> restoring schema"
docker exec "$CONTAINER" pg_restore --schema-only --no-owner -U postgres -d knowledgedb \
  /tmp/knowledgedb.dump || true

echo "==> dropping non-PK indexes (HNSW on invalid vectors) to speed COPY"
docker exec "$CONTAINER" psql -U postgres -d knowledgedb -t -c \
  "SELECT 'DROP INDEX IF EXISTS '||indexname||';' FROM pg_indexes \
   WHERE schemaname='public' AND indexname NOT LIKE '%pkey%';" \
  | docker exec -i "$CONTAINER" psql -U postgres -d knowledgedb

echo "==> restoring data (parallel, triggers disabled)"
docker exec "$CONTAINER" pg_restore --data-only --no-owner --disable-triggers \
  -U postgres -d knowledgedb -j 4 /tmp/knowledgedb.dump

echo "==> row counts"
for t in family_law constitutional bible_kjv bible_web behavioral_patterns behavioral_sources professional_standards; do
  printf "  %-24s " "$t"
  docker exec "$CONTAINER" psql -U postgres -d knowledgedb -t -c "SELECT count(*) FROM $t;" | tr -d ' \n'
  echo
done
echo "==> done. Next: make reembed"
