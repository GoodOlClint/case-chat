"""Per-source-type loaders + chunkers for the synthetic raw-document corpus.

Each loader turns one raw file into a list of :class:`Chunk`. Chunking is
tuned per format (notes by size, emails per message, message threads by
turn-window, court docs by section/size, the visitation CSV per visit row).

ONLY raw case documents are loaded — an explicit allowlist of globs
(:data:`SOURCE_GLOBS`) means the evaluation ground-truth JSON
(entities/timeline/master_facts/expected_*) and generator/meta files are never
reachable. RSMF duplicates of the XML threads, the combined messages CSV, PDFs
and the .m4a are intentionally excluded (see DESIGN §7).
"""

from __future__ import annotations

import csv
import email
import email.policy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Source types → glob (relative to the synthetic corpus root).
SOURCE_GLOBS: dict[str, str] = {
    "apple_note": "apple-notes/notes/*.md",
    "attachment_ocr": "apple-notes/attachments/*.jpeg.txt",
    "email": "emails/*.eml",
    "court_document": "court-documents/*.txt",
    # RSMF is the canonical message format (matches case-project ingestion); the
    # .xml siblings are ignored. Parsed via the vendored rsmf_viewer.
    "messages": "messages/*.rsmf",
    "transcript": "transcripts/*.txt",
    "witness_statement": "witness-statements/*.txt",
    # visitation_log is handled specially (CSV + schema sidecar).
}


@dataclass
class Chunk:
    source_type: str
    source_path: str  # relative to corpus root
    text: str
    seq: int = 0
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.source_path}::{self.seq}"


# --------------------------------------------------------------------------
# Chunking helpers
# --------------------------------------------------------------------------
def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of blank lines (court docs are double-spaced).
    return re.sub(r"\n[ \t]*\n([ \t]*\n)+", "\n\n", text).strip()


def window_text(text: str, *, target: int = 1200, hard_max: int = 1800) -> list[str]:
    """Group paragraphs into ~target-char windows, never exceeding hard_max."""
    text = _normalize(text)
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    windows: list[str] = []
    buf = ""
    for para in paras:
        if len(para) > hard_max:
            if buf:
                windows.append(buf)
                buf = ""
            windows.extend(_split_long(para, hard_max))
            continue
        if buf and len(buf) + len(para) + 2 > target:
            windows.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        windows.append(buf)
    return windows


def _split_long(para: str, hard_max: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", para)
    out: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > hard_max:
            out.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}" if buf else s
    if buf:
        out.append(buf)
    return out


def _frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter (--- ... ---) from a markdown body."""
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), m.group(2)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_apple_note(path: Path, rel: str) -> list[Chunk]:
    meta, body = _frontmatter(path.read_text(encoding="utf-8"))
    title = str(meta.get("title") or path.stem).strip().strip("“”\"'")
    md = {
        "title": title,
        "created": str(meta.get("created")) if meta.get("created") else None,
        "folder": meta.get("folder"),
        "tags": meta.get("tags") or [],
    }
    return [
        Chunk("apple_note", rel, w, seq=i, title=title, metadata=md)
        for i, w in enumerate(window_text(body))
    ]


def load_email(path: Path, rel: str) -> list[Chunk]:
    msg = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
    body_part = msg.get_body(preferencelist=("plain",))
    body = body_part.get_content() if body_part else (msg.get_content() or "")
    subject = msg.get("subject", path.stem)
    md = {
        "title": subject,
        "from": msg.get("from"),
        "to": msg.get("to"),
        "cc": msg.get("cc"),
        "date": msg.get("date"),
        "subject": subject,
        "message_id": msg.get("message-id"),
    }
    header_line = f"From: {md['from']}\nTo: {md['to']}\nDate: {md['date']}\nSubject: {subject}"
    windows = window_text(body, target=1500, hard_max=2000)
    # Prepend the header context to the first chunk so retrieval has provenance.
    if windows:
        windows[0] = f"{header_line}\n\n{windows[0]}"
    return [
        Chunk("email", rel, w, seq=i, title=subject, metadata={k: v for k, v in md.items() if v})
        for i, w in enumerate(windows)
    ]


def _window_turns(lines: list[str], rel: str, *, thread: str, meta: dict[str, Any]) -> list[Chunk]:
    """Window message lines into ~15-msg / ~1200-char chunks, preserving order."""
    chunks: list[Chunk] = []
    buf: list[str] = []
    cur_len = 0
    title = f"Text thread: {thread}"
    for ln in lines:
        if buf and (len(buf) >= 15 or cur_len + len(ln) > 1200):
            chunks.append(Chunk("messages", rel, "\n".join(buf), seq=len(chunks),
                                title=title, metadata={**meta, "title": title}))
            buf, cur_len = [], 0
        buf.append(ln)
        cur_len += len(ln)
    if buf:
        chunks.append(Chunk("messages", rel, "\n".join(buf), seq=len(chunks),
                            title=title, metadata={**meta, "title": title}))
    return chunks


def load_messages_rsmf(path: Path, rel: str) -> list[Chunk]:
    """Parse an RSMF thread (the canonical message format) into windowed chunks.

    Uses the vendored rsmf_viewer to resolve participants, conversation title,
    and message order, then renders each message as ``[ts] Sender: body``.
    """
    from case_chat.vendor import rsmf_viewer as rv

    manifest, zf = rv.parse_rsmf(path)
    try:
        # embed_assets=True keeps attachments in memory (we discard them here);
        # it avoids the on-disk asset_dir path entirely.
        convs, _ = rv.build_conversations(manifest, zf, embed_assets=True, asset_dir=None)
    finally:
        zf.close()

    chunks: list[Chunk] = []
    for conv in convs:
        thread = conv.display or path.stem.replace("-thread", "")
        participants = [p.display for p in conv.participants]
        lines: list[str] = []
        for ev in conv.events:
            if ev.etype != "message":
                continue
            body = (ev.body or "").strip()
            if not body:
                continue
            sender = ev.participant.display if ev.participant else "Unknown"
            when = ev.timestamp_raw or ""
            lines.append(f"[{when}] {sender}: {body}")
        chunks += _window_turns(lines, rel, thread=thread,
                                meta={"thread": thread, "participants": participants})
    return chunks


_CASE_NO = re.compile(r"Case\s+No\.?\s*([0-9A-Z-]+)", re.IGNORECASE)


def load_court_document(path: Path, rel: str) -> list[Chunk]:
    text = _normalize(path.read_text(encoding="utf-8"))
    case_no = None
    m = _CASE_NO.search(text)
    if m:
        case_no = m.group(1)
    title = path.stem.replace("-", " ").title()
    md = {"title": title, "case_no": case_no, "document": path.name}
    return [
        Chunk("court_document", rel, w, seq=i, title=title,
              metadata={k: v for k, v in md.items() if v})
        for i, w in enumerate(window_text(text, target=1800, hard_max=2400))
    ]


def load_text_doc(path: Path, rel: str, source_type: str) -> list[Chunk]:
    title = path.stem.replace("-", " ").title()
    return [
        Chunk(source_type, rel, w, seq=i, title=title, metadata={"title": title})
        for i, w in enumerate(window_text(path.read_text(encoding="utf-8"), target=1500, hard_max=2000))
    ]


def load_attachment_ocr(path: Path, rel: str) -> list[Chunk]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    image = path.name[:-4] if path.name.endswith(".txt") else path.name
    return [Chunk("attachment_ocr", rel, text, title=image, metadata={"image": image, "title": image})]


def load_visitation_log(csv_path: Path, schema_path: Path, rel: str) -> list[Chunk]:
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
    fields = {f["name"]: f for f in schema.get("fields", [])}
    chunks: list[Chunk] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            lines: list[str] = []
            for name, val in row.items():
                if val is None or not str(val).strip():
                    continue
                label = fields.get(name, {}).get("label", name)
                lines.append(f"{label}: {val}")
            if not lines:
                continue
            vdate = row.get("visit_date", "")
            vtype = row.get("visit_type", "")
            parent = row.get("parent", "")
            title = f"Supervised visit {vdate} — {vtype} ({parent})".strip()
            md = {
                "title": title,
                "dataset": schema.get("dataset_name"),
                "visit_date": vdate or None,
                "parent": parent or None,
                "children_present": row.get("children_present") or None,
                "incident_reported": row.get("incident_reported") or None,
            }
            chunks.append(
                Chunk("visitation_log", rel, "\n".join(lines), seq=i, title=title,
                      metadata={k: v for k, v in md.items() if v})
            )
    return chunks
