"""Structured fake-case dataset (SQLite) built from synthetic ground-truth.

A faithful, fictional stand-in for what case-project's extraction pipeline will
eventually emit (timeline events, entities, facts, flags, observations). Built
from the synthetic corpus's ground-truth JSON — NEVER indexed for RAG ([ADR
0003]); it backs exact-lookup ``case.*`` tools so questions like "when was the
guardianship petition filed?" resolve against structured data.

Build:  uv run python -m case_chat.casedata.dataset [--rebuild]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from case_chat.config import settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE entities (
  id TEXT PRIMARY KEY, canonical_name TEXT, type TEXT, dob TEXT, dod TEXT,
  role TEXT, description TEXT, notes TEXT, relationships TEXT, contact_info TEXT
);
CREATE TABLE entity_aliases (entity_id TEXT, alias TEXT);
CREATE TABLE timeline (
  id TEXT PRIMARY KEY, date TEXT, date_precision TEXT, event TEXT,
  category TEXT, notes TEXT, source_documents TEXT
);
CREATE TABLE timeline_entities (timeline_id TEXT, entity_id TEXT);
CREATE TABLE facts (
  id TEXT PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT,
  valid_from TEXT, valid_to TEXT, category TEXT, source_documents TEXT
);
CREATE TABLE flags (
  id TEXT PRIMARY KEY, type TEXT, description TEXT, expected_system_behavior TEXT,
  ground_truth_resolution TEXT, severity TEXT, documents TEXT
);
CREATE TABLE observations (
  id TEXT PRIMARY KEY, observer TEXT, claim TEXT, observed_date TEXT,
  referent_subject TEXT, referent_window_start TEXT, referent_window_end TEXT,
  claim_basis TEXT, confidence REAL, documents TEXT
);
CREATE TABLE case_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX idx_timeline_date ON timeline(date);
CREATE INDEX idx_timeline_category ON timeline(category);
CREATE INDEX idx_te_entity ON timeline_entities(entity_id);
CREATE INDEX idx_alias ON entity_aliases(alias);
CREATE INDEX idx_facts_subject ON facts(subject);
"""

GROUND_TRUTH_FILES = (
    "entities.json", "timeline.json", "master_facts.json",
    "expected_flags.json", "expected_observations.json", "manifest.json",
)


def _normalize_docs(paths: Any) -> Any:
    """Remap ground-truth message references to the canonical RSMF format.

    The corpus's ground-truth points at ``messages/<name>.xml``, but RSMF is now
    the canonical (and viewable) message format, so links resolve to the .rsmf
    sibling. Non-message paths pass through unchanged.
    """
    if not isinstance(paths, list):
        return paths
    out = []
    for p in paths:
        if isinstance(p, str) and p.startswith("messages/") and p.endswith(".xml"):
            out.append(p[:-4] + ".rsmf")
        else:
            out.append(p)
    return out


def _j(value: Any) -> str | None:
    return json.dumps(value) if value is not None else None


def _jdocs(value: Any) -> str | None:
    """JSON-encode a source-document list after normalizing message paths."""
    return _j(_normalize_docs(value))


def build_dataset(*, corpus_root: Path | None = None, sqlite_path: Path | None = None) -> Path:
    root = corpus_root or Path(settings.synthetic_corpus_path)
    out = sqlite_path or Path(settings.casedata_sqlite_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def load(name: str) -> Any:
        return json.loads((root / name).read_text(encoding="utf-8"))

    if out.exists():
        out.unlink()
    conn = sqlite3.connect(out)
    try:
        conn.executescript(SCHEMA)

        for e in load("entities.json"):
            conn.execute(
                "INSERT INTO entities VALUES (?,?,?,?,?,?,?,?,?,?)",
                (e["id"], e.get("canonical_name"), e.get("type"), e.get("dob"),
                 e.get("dod"), e.get("role"), e.get("description"), e.get("notes"),
                 _j(e.get("relationships")), _j(e.get("contact_info"))),
            )
            for a in e.get("aliases") or []:
                alias = a.get("alias") if isinstance(a, dict) else a
                if alias:
                    conn.execute("INSERT INTO entity_aliases VALUES (?,?)", (e["id"], alias))
            # The canonical name is itself a lookup key.
            conn.execute("INSERT INTO entity_aliases VALUES (?,?)", (e["id"], e.get("canonical_name")))

        for t in load("timeline.json"):
            conn.execute(
                "INSERT INTO timeline VALUES (?,?,?,?,?,?,?)",
                (t["id"], t.get("date"), t.get("date_precision"), t.get("event"),
                 t.get("category"), t.get("notes"), _jdocs(t.get("source_documents"))),
            )
            for eid in t.get("entities_involved") or []:
                conn.execute("INSERT INTO timeline_entities VALUES (?,?)", (t["id"], eid))

        for f in load("master_facts.json"):
            conn.execute(
                "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?)",
                (f["id"], f.get("subject"), f.get("predicate"), f.get("object"),
                 f.get("valid_from"), f.get("valid_to"), f.get("category"),
                 _jdocs(f.get("source_documents"))),
            )

        for fl in load("expected_flags.json"):
            conn.execute(
                "INSERT INTO flags VALUES (?,?,?,?,?,?,?)",
                (fl["id"], fl.get("type"), fl.get("description"),
                 fl.get("expected_system_behavior"), fl.get("ground_truth_resolution"),
                 fl.get("severity"), _jdocs(fl.get("documents"))),
            )

        for o in load("expected_observations.json"):
            conn.execute(
                "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)",
                (o["id"], o.get("observer"), o.get("claim"), o.get("observed_date"),
                 o.get("referent_subject"), o.get("referent_window_start"),
                 o.get("referent_window_end"), o.get("claim_basis"),
                 o.get("confidence"), _jdocs(o.get("documents"))),
            )

        manifest = load("manifest.json")
        for key in ("corpus_name", "description", "jurisdiction", "case_number", "generated"):
            if manifest.get(key) is not None:
                conn.execute("INSERT INTO case_meta VALUES (?,?)", (key, str(manifest[key])))

        conn.commit()
    finally:
        conn.close()
    logger.info("built fake-case dataset at %s", out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build the SQLite fake-case dataset")
    ap.add_argument("--rebuild", action="store_true", help="(default) rebuild from ground-truth")
    ap.parse_args()
    path = build_dataset()
    conn = sqlite3.connect(path)
    counts = {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in ("entities", "timeline", "facts", "flags", "observations")
    }
    conn.close()
    logger.info("row counts: %s", counts)


if __name__ == "__main__":
    main()
