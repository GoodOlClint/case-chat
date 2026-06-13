#!/usr/bin/env bash
# Web container entrypoint. Builds the (cheap) fake-case SQLite from the mounted
# corpus if missing, then serves. The heavy index build (Qdrant) is a one-time
# job run via the `build` profile — see deploy/build_indexes.sh.
set -e

if [ ! -f "${CASECHAT_CASEDATA_SQLITE_PATH:-/data/fake_case.sqlite3}" ]; then
  echo "[entrypoint] building fake-case dataset…"
  python -m case_chat.casedata.dataset || echo "[entrypoint] casedata build skipped (corpus not mounted?)"
fi

exec uvicorn case_chat.web.app:app --host 0.0.0.0 --port "${CASECHAT_WEB_PORT:-8080}"
