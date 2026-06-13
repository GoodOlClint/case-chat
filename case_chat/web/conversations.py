"""Persistent chat-history store (SQLite), per authenticated subject.

Each conversation has its own running context (sessions are keyed by conversation
id, not subject), so a new chat is genuinely isolated and a reload can restore
exactly what was asked. Stores the full display transcript — user/assistant text,
the answer's citations, and the model's thinking — so the UI can re-render a
saved conversation faithfully.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from case_chat.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, subject TEXT NOT NULL, title TEXT NOT NULL DEFAULT '',
  created REAL NOT NULL, updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  conv_id TEXT NOT NULL, idx INTEGER NOT NULL, role TEXT NOT NULL,
  content TEXT, citations TEXT, thinking TEXT
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conv_id);
CREATE INDEX IF NOT EXISTS idx_conv_subject ON conversations(subject);
"""


class ConversationStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or settings.conversations_sqlite_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def _now(self) -> float:
        return time.time()

    def create(self, conv_id: str, subject: str) -> None:
        now = self._now()
        self._conn.execute(
            "INSERT OR IGNORE INTO conversations(id, subject, title, created, updated) "
            "VALUES (?,?,?,?,?)",
            (conv_id, subject, "", now, now),
        )
        self._conn.commit()

    def owns(self, conv_id: str, subject: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM conversations WHERE id=? AND subject=?", (conv_id, subject)
        ).fetchone()
        return row is not None

    def add_turn(
        self, conv_id: str, subject: str, role: str, content: str,
        *, citations: list[dict[str, Any]] | None = None, thinking: str | None = None,
    ) -> None:
        self.create(conv_id, subject)
        idx = self._conn.execute(
            "SELECT COALESCE(MAX(idx), -1) + 1 FROM messages WHERE conv_id=?", (conv_id,)
        ).fetchone()[0]
        self._conn.execute(
            "INSERT INTO messages(conv_id, idx, role, content, citations, thinking) "
            "VALUES (?,?,?,?,?,?)",
            (conv_id, idx, role, content,
             json.dumps(citations) if citations else None, thinking),
        )
        if role == "user":
            cur = self._conn.execute("SELECT title FROM conversations WHERE id=?", (conv_id,))
            row = cur.fetchone()
            if row is not None and not row["title"]:
                title = (content or "").strip().replace("\n", " ")[:60] or "Untitled"
                self._conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, conv_id))
        self._conn.execute(
            "UPDATE conversations SET updated=? WHERE id=?", (self._now(), conv_id)
        )
        self._conn.commit()

    def list(self, subject: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, title, updated FROM conversations "
            "WHERE subject=? AND title != '' ORDER BY updated DESC",
            (subject,),
        ).fetchall()
        return [{"id": r["id"], "title": r["title"], "updated": r["updated"]} for r in rows]

    def get(self, conv_id: str, subject: str) -> dict[str, Any] | None:
        conv = self._conn.execute(
            "SELECT id, title FROM conversations WHERE id=? AND subject=?", (conv_id, subject)
        ).fetchone()
        if conv is None:
            return None
        msgs = self._conn.execute(
            "SELECT role, content, citations, thinking FROM messages "
            "WHERE conv_id=? ORDER BY idx",
            (conv_id,),
        ).fetchall()
        return {
            "id": conv["id"],
            "title": conv["title"],
            "messages": [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "citations": json.loads(m["citations"]) if m["citations"] else [],
                    "thinking": m["thinking"],
                }
                for m in msgs
            ],
        }

    def delete(self, conv_id: str, subject: str) -> bool:
        if not self.owns(conv_id, subject):
            return False
        self._conn.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
        self._conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        self._conn.commit()
        return True
