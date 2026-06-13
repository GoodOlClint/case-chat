"""Query layer over the SQLite fake-case dataset.

These methods back the ``case.*`` MCP tools: exact/filtered lookups against the
structured fictional case (timeline, entities, facts, flags, observations).
Read-only; JSON columns are parsed back into Python on the way out.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from case_chat.config import settings

_JSON_FIELDS = {"source_documents", "documents", "relationships", "contact_info"}
_AND = " AND "


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in row.keys():
        val = row[key]
        if key in _JSON_FIELDS and isinstance(val, str):
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                pass
        if val is not None:
            out[key] = val
    return out


class CaseDataset:
    """Read-only accessor for the fake-case SQLite dataset."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or settings.casedata_sqlite_path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"fake-case dataset not built: {self._path} "
                "(run: python -m case_chat.casedata.dataset)"
            )
        # check_same_thread=False is safe here: read-only connection, and the
        # web server may touch it from different worker threads.
        self._conn = sqlite3.connect(
            f"file:{self._path}?mode=ro", uri=True, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

    def _select(
        self,
        table: str,
        where: list[str],
        params: list[Any],
        *,
        order: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Build and run a filtered SELECT; returns rows as dicts."""
        # `table` is always a fixed literal from our own call sites, never user input.
        sql = f"SELECT * FROM {table}"
        if where:
            sql += " WHERE " + _AND.join(where)
        if order:
            sql += f" ORDER BY {order}"
        sql += " LIMIT ?"
        return [_row_to_dict(r) for r in self._conn.execute(sql, [*params, limit]).fetchall()]

    # -- entities ----------------------------------------------------------
    def resolve_entity_ids(self, name: str) -> list[str]:
        """Map a name / alias / id to entity ids.

        Three escalating strategies: exact alias/id match; whole-string substring
        on aliases; then token-AND on the canonical name (so 'Kaylee Holcomb'
        resolves to 'Kaylee Ann Holcomb' even though no alias matches exactly).
        """
        cur = self._conn.execute(
            "SELECT DISTINCT entity_id FROM entity_aliases WHERE lower(alias) = lower(?) "
            "UNION SELECT id FROM entities WHERE lower(id) = lower(?)",
            (name, name),
        )
        ids = [r[0] for r in cur.fetchall()]
        if ids:
            return ids

        cur = self._conn.execute(
            "SELECT DISTINCT entity_id FROM entity_aliases WHERE lower(alias) LIKE lower(?)",
            (f"%{name}%",),
        )
        ids = [r[0] for r in cur.fetchall()]
        if ids:
            return ids

        tokens = [t for t in re.split(r"\s+", name.strip()) if t]
        if len(tokens) >= 2:
            clause = _AND.join(["lower(canonical_name) LIKE lower(?)"] * len(tokens))
            cur = self._conn.execute(
                f"SELECT id FROM entities WHERE {clause}", tuple(f"%{t}%" for t in tokens)
            )
            ids = [r[0] for r in cur.fetchall()]
        return ids

    def _name_map(self) -> dict[str, dict[str, Any]]:
        """id -> {name, role}, cached on the instance (entities are static)."""
        if getattr(self, "_namemap", None) is None:
            rows = self._conn.execute("SELECT id, canonical_name, role FROM entities").fetchall()
            self._namemap = {r["id"]: {"name": r["canonical_name"], "role": r["role"]} for r in rows}
        return self._namemap

    def _resolve_rels(self, raw: Any) -> list[dict[str, Any]]:
        """Resolve an entity's outgoing relationships (related_entity ids → names)."""
        nm = self._name_map()
        out = []
        for rel in raw or []:
            tid = rel.get("related_entity")
            info = nm.get(tid, {})
            out.append({"related_entity": tid, "name": info.get("name"),
                        "role": info.get("role"), "relationship": rel.get("relationship")})
        return out

    def _referenced_by(self, entity_id: str) -> list[dict[str, Any]]:
        """Reverse relationships: other entities whose relationships point here."""
        nm = self._name_map()
        out = []
        rows = self._conn.execute(
            "SELECT id, relationships FROM entities WHERE relationships LIKE ?",
            (f"%{entity_id}%",),
        ).fetchall()
        for r in rows:
            if r["id"] == entity_id:
                continue
            try:
                rels = json.loads(r["relationships"] or "[]")
            except json.JSONDecodeError:
                rels = []
            for rel in rels:
                if rel.get("related_entity") == entity_id:
                    info = nm.get(r["id"], {})
                    out.append({"entity": r["id"], "name": info.get("name"),
                                "role": info.get("role"), "relationship": rel.get("relationship")})
        return out

    def entity_lookup(self, name: str, *, limit: int = 5) -> list[dict[str, Any]]:
        ids = self.resolve_entity_ids(name)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT * FROM entities WHERE id IN ({placeholders}) LIMIT ?",
            (*ids, limit),
        ).fetchall()
        out = []
        for r in rows:
            d = _row_to_dict(r)
            aliases = self._conn.execute(
                "SELECT DISTINCT alias FROM entity_aliases WHERE entity_id = ?", (d["id"],)
            ).fetchall()
            d["aliases"] = [a[0] for a in aliases]
            d["relationships"] = self._resolve_rels(d.get("relationships"))
            d["referenced_by"] = self._referenced_by(d["id"])
            out.append(d)
        return out

    def list_participants(
        self, *, role: str | None = None, entity_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List case participants, optionally filtered by role (substring) or type."""
        where: list[str] = []
        params: list[Any] = []
        if role:
            where.append("lower(role) LIKE lower(?)")
            params.append(f"%{role}%")
        if entity_type:
            where.append("lower(type) = lower(?)")
            params.append(entity_type)
        sql = "SELECT id, canonical_name, type, role, dob, description FROM entities"
        if where:
            sql += " WHERE " + _AND.join(where)
        sql += " ORDER BY id LIMIT ?"
        params.append(limit)
        return [_row_to_dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def case_overview(self) -> dict[str, Any]:
        """Case metadata + full participant roster + dataset counts."""
        counts = {
            t: self._conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("entities", "timeline", "facts", "flags", "observations")
        }
        return {
            "case": self.case_meta(),
            "counts": counts,
            "participants": self.list_participants(limit=200),
        }

    def case_facts_view(self) -> dict[str, Any]:
        """The full structured case record for the 'Case facts' browse view."""
        return {
            "case": self.case_meta(),
            "participants": self.list_participants(limit=500),
            "timeline": self.timeline_query(limit=500),
            "facts": self.facts_query(limit=500),
            "flags": self.flags_query(limit=500),
            "observations": self.observations_query(limit=500),
        }

    # -- timeline ----------------------------------------------------------
    def timeline_query(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        entity: str | None = None,
        category: str | None = None,
        text: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if date_from:
            where.append("date >= ?")
            params.append(date_from)
        if date_to:
            where.append("date <= ?")
            params.append(date_to)
        if category:
            where.append("lower(category) = lower(?)")
            params.append(category)
        if text:
            where.append("(event LIKE ? OR notes LIKE ?)")
            params += [f"%{text}%", f"%{text}%"]
        if entity:
            ids = self.resolve_entity_ids(entity)
            if not ids:
                return []
            ph = ",".join("?" * len(ids))
            where.append(f"id IN (SELECT timeline_id FROM timeline_entities WHERE entity_id IN ({ph}))")
            params += ids
        return self._select("timeline", where, params, order="date", limit=limit)

    # -- facts -------------------------------------------------------------
    def facts_query(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        category: str | None = None,
        text: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        for col, val in (("subject", subject), ("predicate", predicate), ("object", obj),
                         ("category", category)):
            if val:
                where.append(f"{col} LIKE ?")
                params.append(f"%{val}%")
        if text:
            where.append("(subject LIKE ? OR predicate LIKE ? OR object LIKE ?)")
            params += [f"%{text}%"] * 3
        return self._select("facts", where, params, limit=limit)

    # -- flags -------------------------------------------------------------
    def flags_query(
        self,
        *,
        flag_type: str | None = None,
        severity: str | None = None,
        text: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if flag_type:
            where.append("type LIKE ?")
            params.append(f"%{flag_type}%")
        if severity:
            where.append("lower(severity) = lower(?)")
            params.append(severity)
        if text:
            where.append("(description LIKE ? OR ground_truth_resolution LIKE ?)")
            params += [f"%{text}%", f"%{text}%"]
        return self._select("flags", where, params, limit=limit)

    # -- observations ------------------------------------------------------
    def observations_query(
        self,
        *,
        observer: str | None = None,
        subject: str | None = None,
        text: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if observer:
            where.append("observer LIKE ?")
            params.append(f"%{observer}%")
        if subject:
            where.append("referent_subject LIKE ?")
            params.append(f"%{subject}%")
        if text:
            where.append("claim LIKE ?")
            params.append(f"%{text}%")
        return self._select("observations", where, params, limit=limit)

    def case_meta(self) -> dict[str, str]:
        return dict(self._conn.execute("SELECT key, value FROM case_meta").fetchall())

    def close(self) -> None:
        self._conn.close()
