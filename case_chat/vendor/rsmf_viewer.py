#!/usr/bin/env python3
"""rsmf_viewer — convert RSMF 2.0.0 message exports into pretty HTML or Markdown.

Reads .rsmf (RFC 5322 MIME envelope wrapping a base64 ZIP) or raw .zip
files containing an RSMF manifest JSON plus attachment files, and emits either
a single self-contained HTML page or a Markdown document that renders each
conversation as a chat transcript with reactions, edits, attachments, and
join/leave/disclaimer system events. Read receipts are parsed but not rendered.

The Markdown export carries conversation metadata in YAML frontmatter, retains
per-message timestamps under date dividers, and attributes every inline
attachment to its sender.

Usage:
    rsmf_viewer.py INPUT.rsmf [-o OUTPUT.html]
    rsmf_viewer.py INPUT.rsmf -o OUTPUT.md              # Markdown (by extension)
    rsmf_viewer.py INPUT.rsmf --format markdown --no-embed   # assets on disk
    rsmf_viewer.py INPUT.rsmf --format both            # HTML + Markdown
    rsmf_viewer.py *.rsmf --out-dir ./html             # batch mode

Schema reference:
    https://github.com/relativitydev/rsmf-validator-samples
"""

from __future__ import annotations

import argparse
import base64
import bisect
import email.parser
import hashlib
import html
import io
import json
import logging
import mimetypes
import re
import sys
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("rsmf_viewer")

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class RsmfParseError(Exception):
    """Raised when an .rsmf input cannot be decoded."""


def _decode_chunked_base64(payload: str) -> bytes:
    """Decode base64 that may have per-line padding (iMazing-style chunking).

    Python's stdlib base64 stops at the first padding sequence, so chunked
    streams must be decoded line-by-line and concatenated.
    """
    chunks: list[bytes] = []
    for line in payload.split("\n"):
        line = line.strip()
        if line:
            chunks.append(base64.b64decode(line))
    return b"".join(chunks)


def _try_open_zip(blob: bytes) -> zipfile.ZipFile | None:
    """Open a ZIP and verify every entry's CRC.

    `ZipFile()` only validates the central directory at EOF, so a partially
    corrupt decode (e.g., from chunked-base64 stripping) can construct cleanly
    and only blow up later on `.read()`. `testzip()` reads every local header
    and CRC; if it returns None, the archive is genuinely intact.
    """
    if not blob or blob[:4] != b"PK\x03\x04":
        return None
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob), "r")
        bad = zf.testzip()
        if bad is None:
            return zf
        logger.debug("ZIP entry %r failed CRC — trying next decoder", bad)
        zf.close()
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError) as e:
        logger.debug("ZIP open failed: %s", e)
    return None


def _extract_zip_from_mime(raw: bytes, source: Path) -> zipfile.ZipFile:
    msg = email.parser.BytesParser().parsebytes(raw)
    seen: list[str] = []
    for part in msg.walk():
        ct = part.get_content_type()
        seen.append(ct)
        if ct != "application/zip":
            continue

        # Try every decoder; pick the first whose payload reads cleanly.
        # Order: stdlib (fast, correct for normal MIME), then chunked
        # (handles iMazing's per-line padding), then chunked from the
        # transport-decoded payload (handles odd quoted-printable wrappers).
        candidates: list[bytes] = []

        stdlib = part.get_payload(decode=True)
        if stdlib:
            candidates.append(stdlib)

        raw_payload = part.get_payload(decode=False)
        if isinstance(raw_payload, str):
            try:
                candidates.append(_decode_chunked_base64(raw_payload))
            except Exception as e:  # noqa: BLE001 — try the next strategy
                logger.debug("chunked base64 decode failed: %s", e)

        if isinstance(stdlib, bytes):
            try:
                candidates.append(_decode_chunked_base64(stdlib.decode("ascii", "ignore")))
            except Exception as e:  # noqa: BLE001
                logger.debug("re-decode of stdlib payload failed: %s", e)

        for blob in candidates:
            zf = _try_open_zip(blob)
            if zf is not None:
                return zf

        raise RsmfParseError(
            f"application/zip part in {source.name} could not be decoded "
            f"into a valid ZIP (tried {len(candidates)} decoder(s))"
        )
    raise RsmfParseError(
        f"No application/zip part found in {source.name}; saw {seen}"
    )


def _find_manifest(zf: zipfile.ZipFile) -> str:
    candidates = sorted(
        n for n in zf.namelist()
        if "/" not in n and n.lower().endswith(".json")
    )
    if not candidates:
        raise RsmfParseError(
            f"No root-level JSON manifest in ZIP; files: {zf.namelist()[:20]}"
        )
    if len(candidates) > 1:
        logger.warning("Multiple root JSON files; using %s", candidates[0])
    return candidates[0]


def parse_rsmf(path: Path) -> tuple[dict, zipfile.ZipFile]:
    """Open an .rsmf or raw .zip and return (manifest_dict, open_zip)."""
    raw = path.read_bytes()
    head = raw[:20]
    if path.suffix.lower() == ".rsmf" or head.startswith((b"MIME-", b"From ", b"Content-")):
        zf = _extract_zip_from_mime(raw, path)
    elif raw[:4] == b"PK\x03\x04":
        zf = zipfile.ZipFile(io.BytesIO(raw), "r")
    else:
        zf = _extract_zip_from_mime(raw, path)

    manifest_name = _find_manifest(zf)
    try:
        manifest_bytes = zf.read(manifest_name)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError) as e:
        zf.close()
        raise RsmfParseError(
            f"{path.name}: ZIP central directory was valid but manifest "
            f"{manifest_name!r} could not be read ({e}). "
            f"The base64 payload is likely truncated or chunked oddly."
        ) from e
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as e:
        zf.close()
        raise RsmfParseError(
            f"{path.name}: manifest {manifest_name!r} is not valid JSON: {e}"
        ) from e
    return manifest, zf


# ---------------------------------------------------------------------------
# Model normalization
# ---------------------------------------------------------------------------


@dataclass
class Asset:
    """A resolved attachment ready for embedding/linking in HTML."""
    attachment_id: str
    filename: str
    mime_type: str | None
    size: int
    data: bytes | None  # None when --no-embed and we wrote to disk
    href: str           # data: URL or relative path


@dataclass
class Participant:
    pid: str
    display: str
    email: str | None = None
    account_id: str | None = None
    color_idx: int = 0  # for stable per-participant bubble color


@dataclass
class RenderEvent:
    raw: dict
    etype: str
    eid: str | None
    timestamp: datetime | None
    timestamp_raw: str | None
    participant: Participant | None
    body: str
    deleted: bool
    direction: str | None
    importance: str | None
    parent: str | None
    reactions: list[dict] = field(default_factory=list)
    attachments: list[Asset] = field(default_factory=list)
    edits: list[dict] = field(default_factory=list)
    read_receipts: list[dict] = field(default_factory=list)


@dataclass
class Conversation:
    cid: str
    display: str
    platform: str
    ctype: str | None
    participants: list[Participant]
    custodian: Participant | None
    events: list[RenderEvent]


# ---------------------------------------------------------------------------
# Asset extraction
# ---------------------------------------------------------------------------


def _asset_candidates(att_id: str, display: str | None) -> list[str]:
    out = [att_id, f"attachments/{att_id}", f"Attachments/{att_id}"]
    if display:
        out += [display, f"attachments/{display}", f"Attachments/{display}"]
    return out


def _resolve_asset(
    zf: zipfile.ZipFile,
    att: dict,
    *,
    embed: bool,
    asset_dir: Path | None,
    used_names: set[str],
) -> Asset | None:
    att_id = att.get("id") or ""
    display = att.get("display")
    declared_size = int(att.get("size") or 0)

    names = set(zf.namelist())
    src = next((c for c in _asset_candidates(att_id, display) if c in names), None)
    if src is None:
        logger.warning("Attachment %s not in ZIP", att_id)
        return None

    info = zf.getinfo(src)
    raw = zf.read(src)
    filename = display or att_id
    mime, _ = mimetypes.guess_type(filename)

    if embed:
        b64 = base64.b64encode(raw).decode("ascii")
        href = f"data:{mime or 'application/octet-stream'};base64,{b64}"
        return Asset(att_id, filename, mime, info.file_size or declared_size, raw, href)

    assert asset_dir is not None
    out_name = filename
    # Avoid collisions across different attachments with the same display name.
    if out_name in used_names:
        sha = hashlib.sha1(raw).hexdigest()[:8]
        stem = Path(out_name).stem
        suffix = Path(out_name).suffix
        out_name = f"{stem}_{sha}{suffix}"
    used_names.add(out_name)

    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / out_name).write_bytes(raw)
    href = f"{asset_dir.name}/{out_name}"
    return Asset(att_id, filename, mime, info.file_size or declared_size, None, href)


# ---------------------------------------------------------------------------
# Build conversations from manifest
# ---------------------------------------------------------------------------


_PALETTE_LEN = 8  # see CSS .bub.p0 ... .p7

# Identifier-shape heuristics, used when picking the canonical display name
# for a merged group of participant records and when auto-detecting the
# iMazing "device owner" record.
_PHONE_RE = re.compile(r"^[+\d][\d\s().\-]{4,}$")
_IMAZING_DEVICE_RE = re.compile(r"^(iPhone|iPad|iPod|Mac)(\s*\(\d+\))?\s*$", re.IGNORECASE)


def _is_phoneish(s: str) -> bool:
    return bool(s and _PHONE_RE.match(s.strip()))


def _is_imazing_device(s: str) -> bool:
    return bool(s and _IMAZING_DEVICE_RE.match(s.strip()))


def _participant_keys(p: dict) -> set[str]:
    """All identifier strings that should resolve to this participant.

    Includes raw field values and, for iMazing custodian records, each
    comma-separated token from account_id (which packs every device identifier
    into a single field).
    """
    keys: set[str] = set()
    for field_name in ("id", "display", "email", "account_id"):
        v = (p.get(field_name) or "").strip()
        if not v:
            continue
        keys.add(v)
        if "," in v:
            for part in v.split(","):
                part = part.strip()
                if part:
                    keys.add(part)
    return keys


def _pick_canonical_display(group: list[dict]) -> str:
    """From merged participant records, pick the most human-readable display."""
    # Pass 1: any display field that isn't a phone number or device label
    for p in group:
        d = (p.get("display") or "").strip()
        if d and not _is_phoneish(d) and not _is_imazing_device(d) and "," not in d:
            return d
    # Pass 2: id field that isn't a phone or device label
    for p in group:
        i = (p.get("id") or "").strip()
        if i and not _is_phoneish(i) and not _is_imazing_device(i) and "," not in i:
            return i
    # Pass 3: anything non-empty
    for p in group:
        for v in ((p.get("display") or "").strip(), (p.get("id") or "").strip()):
            if v:
                return v
    return "Unknown"


def _merge_participants(
    raw: list[dict],
) -> tuple[list[Participant], dict[str, Participant]]:
    """Union-find merge of participant records that share any identifier.

    Returns (canonical_participants, alias_lookup) where alias_lookup maps
    every original identifier (id, display, email, account_id, comma-split
    tokens, plus the canonical pid) to the canonical Participant instance.
    """
    n = len(raw)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Index every key to the participant indices that mention it; merge them.
    key_to_indices: dict[str, list[int]] = {}
    for idx, p in enumerate(raw):
        for k in _participant_keys(p):
            key_to_indices.setdefault(k, []).append(idx)
    for indices in key_to_indices.values():
        for i in indices[1:]:
            union(indices[0], i)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    canonical: list[Participant] = []
    alias_lookup: dict[str, Participant] = {}
    color_idx = 0
    # Preserve manifest order via the smallest index in each group
    for root in sorted(groups.keys(), key=lambda r: min(groups[r])):
        indices = groups[root]
        group = [raw[i] for i in indices]
        display = _pick_canonical_display(group)
        # Stable canonical pid: first record's id, falls back to display
        pid = (group[0].get("id") or "").strip() or display
        email = next((p.get("email") for p in group if p.get("email")), None)
        account_id = next((p.get("account_id") for p in group if p.get("account_id")), None)

        participant = Participant(
            pid=pid,
            display=display,
            email=email,
            account_id=account_id,
            color_idx=color_idx % _PALETTE_LEN,
        )
        canonical.append(participant)
        for p in group:
            for k in _participant_keys(p):
                alias_lookup[k] = participant
        alias_lookup[pid] = participant
        color_idx += 1

    return canonical, alias_lookup


def _resolve_me(
    me: str,
    alias_lookup: dict[str, Participant],
    canonical: list[Participant],
) -> Participant | None:
    """Resolve the --me string to a canonical participant.

    Tries: exact alias match, then case-insensitive substring against display,
    email, account_id, and any alias key.
    """
    if not me:
        return None
    if me in alias_lookup:
        return alias_lookup[me]
    tokens = [t for t in me.casefold().split() if t]
    if not tokens:
        return None

    def hay_for(p: Participant) -> str:
        # All identifier strings concatenated, lowered. Includes alias keys
        # so "Sam Rivera" matches samrivera@example.com (each token
        # appears in the email, regardless of spacing).
        keys = [k for k, v in alias_lookup.items() if v is p]
        return " ".join([p.display or "", p.email or "", p.account_id or "", *keys]).casefold()

    matches = [p for p in canonical if all(t in hay_for(p) for t in tokens)]
    if not matches:
        logger.warning("--me %r did not match any participant", me)
        return None
    if len(matches) > 1:
        logger.warning(
            "--me %r matched %d participants (%s); using the first",
            me, len(matches), ", ".join(p.display for p in matches),
        )
    return matches[0]


def _auto_detect_custodian(canonical: list[Participant]) -> Participant | None:
    """iMazing convention: the custodian's display is 'iPhone' / 'iPhone (10)'."""
    for p in canonical:
        if _is_imazing_device(p.display):
            return p
    return None


def _resolve_tz(spec: str | None):
    """Parse a --tz value into a tzinfo. Accepts an IANA name (America/Chicago)
    or a fixed offset (-05:00, +0530, -5). Returns None for empty input."""
    if not spec:
        return None
    spec = spec.strip()
    m = re.fullmatch(r'([+-])(\d{1,2})(?::?(\d{2}))?', spec)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0))
        return timezone(sign * delta)
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(spec)
    except Exception as ex:  # noqa: BLE001 - normalize to a clear caller error
        raise ValueError(f"unknown timezone {spec!r}: {ex}") from ex


def _parse_ts(ts: str | None, assume_tz=None) -> datetime | None:
    if not ts:
        return None
    try:
        # fromisoformat in 3.11+ accepts trailing 'Z'; older versions don't.
        s = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    if assume_tz is not None:
        # The source records wall-clock time in assume_tz (some exporters, e.g.
        # iMazing, stamp local time with a bogus 'Z'). Relabel the wall-clock to
        # assume_tz without shifting the displayed digits.
        return dt.replace(tzinfo=assume_tz)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def build_conversations(
    manifest: dict,
    zf: zipfile.ZipFile,
    *,
    embed_assets: bool,
    asset_dir: Path | None,
    me: str | None = None,
    me_name: str | None = None,
    assume_tz=None,
) -> tuple[list[Conversation], dict]:
    """Normalize the raw manifest into Conversation/RenderEvent objects.

    `me` overrides custodian detection (and renames the matched participant).
    `me_name` renames the resolved device owner / custodian — most useful for
    giving the iMazing "iPhone (N)" device participant a real person's name
    without having to match it by an identifier it doesn't carry.
    `assume_tz` relabels every timestamp's wall-clock to that timezone (for
    exporters that record local time with a bogus 'Z').
    """
    raw_participants = manifest.get("participants", [])
    canonical_participants, alias_lookup = _merge_participants(raw_participants)

    me_participant = _resolve_me(me, alias_lookup, canonical_participants) if me else None
    auto_custodian = _auto_detect_custodian(canonical_participants)

    # Display-name override for the device owner. `--me-name` takes precedence
    # and can target the auto-detected device participant even with no `--me`;
    # otherwise `--me NAME` renames the participant it matched (unless `me` was
    # a phone/email that matched an already-named participant).
    rename_target = me_participant or auto_custodian
    if me_name and rename_target is not None:
        rename_target.display = me_name
    elif me_name:
        logger.warning(
            "--me-name %r: no device-owner/custodian participant was found to "
            "rename (no --me match and no iMazing device participant)", me_name,
        )
    elif me_participant is not None and me and not _is_phoneish(me) and "@" not in me:
        me_participant.display = me

    convs_by_id: dict[str, dict] = {c["id"]: c for c in manifest.get("conversations", [])}

    grouped: dict[str, list[dict]] = {}
    for ev in manifest.get("events", []):
        cid = ev.get("conversation") or "__no_conversation__"
        grouped.setdefault(cid, []).append(ev)

    used_asset_names: set[str] = set()
    conversations: list[Conversation] = []

    # Stable order: follow manifest declaration order, then any stragglers.
    order = list(convs_by_id.keys()) + [
        cid for cid in grouped if cid not in convs_by_id
    ]
    seen: set[str] = set()
    for cid in order:
        if cid in seen:
            continue
        seen.add(cid)
        events_raw = grouped.get(cid, [])
        if not events_raw and cid not in convs_by_id:
            continue

        meta = convs_by_id.get(cid, {})
        # De-dup the conversation participant list through the alias map so
        # iMazing's name-as-id + phone-as-id record pairs collapse to one entry.
        seen_participants: set[str] = set()
        participant_objs: list[Participant] = []
        for pid in meta.get("participants", []):
            p = alias_lookup.get(pid)
            if p and p.pid not in seen_participants:
                seen_participants.add(p.pid)
                participant_objs.append(p)

        # Custodian resolution priority:
        #   1. --me override (applies to every conversation in the export)
        #   2. The conversation's declared custodian, if it resolves
        #   3. iMazing auto-detect: a participant whose display is "iPhone (N)"
        custodian = (
            me_participant
            or alias_lookup.get(meta.get("custodian") or "")
            or auto_custodian
        )

        rendered: list[RenderEvent] = []
        for ev in events_raw:
            assets: list[Asset] = []
            for att in ev.get("attachments") or []:
                a = _resolve_asset(
                    zf, att,
                    embed=embed_assets,
                    asset_dir=asset_dir,
                    used_names=used_asset_names,
                )
                if a:
                    assets.append(a)

            rendered.append(RenderEvent(
                raw=ev,
                etype=ev.get("type") or "unknown",
                eid=ev.get("id"),
                timestamp=_parse_ts(ev.get("timestamp"), assume_tz),
                timestamp_raw=ev.get("timestamp"),
                participant=alias_lookup.get(ev.get("participant") or ""),
                body=ev.get("body") or "",
                deleted=bool(ev.get("deleted")),
                direction=ev.get("direction"),
                importance=ev.get("importance"),
                parent=ev.get("parent"),
                reactions=ev.get("reactions") or [],
                attachments=assets,
                edits=ev.get("edits") or [],
                read_receipts=ev.get("read_receipts") or [],
            ))

        # Stable sort by timestamp; events without one keep insertion order.
        rendered.sort(key=lambda e: (e.timestamp is None, e.timestamp or datetime.min.replace(tzinfo=timezone.utc)))

        conversations.append(Conversation(
            cid=cid,
            display=meta.get("display") or cid,
            platform=meta.get("platform") or "unknown",
            ctype=meta.get("type"),
            participants=participant_objs or list({
                e.participant.pid: e.participant
                for e in rendered if e.participant
            }.values()),
            custodian=custodian,
            events=rendered,
        ))

    summary = {
        "version": manifest.get("version", "?"),
        "participant_count": len(canonical_participants),
        "conversation_count": len(conversations),
        "event_count": sum(len(c.events) for c in conversations),
        "message_count": sum(
            sum(1 for e in c.events if e.etype == "message")
            for c in conversations
        ),
    }
    return conversations, summary


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


_URL_RE = re.compile(r"(https?://[^\s<>\"']+)")


def _linkify(text: str) -> str:
    """Escape, then auto-link bare URLs."""
    escaped = html.escape(text)
    return _URL_RE.sub(
        lambda m: f'<a href="{m.group(1)}" target="_blank" rel="noopener">{m.group(1)}</a>',
        escaped,
    )


def _fmt_ts(dt: datetime | None, raw: str | None) -> str:
    if dt is None:
        return html.escape(raw or "")
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def _fmt_date(dt: datetime | None) -> str:
    if dt is None:
        return "Undated"
    return dt.strftime("%A, %B %-d, %Y")


def _month_key(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m") if dt else "undated"


def _fmt_month(dt: datetime | None) -> str:
    return dt.strftime("%B %Y") if dt else "Undated"


def _ordered_months(events: list[RenderEvent]) -> list[tuple[str, str]]:
    """Distinct (month_key, "Month YYYY") pairs in first-seen order. Used to
    decide whether a conversation spans multiple months and to build the
    'jump to month' navigation."""
    seen_keys: set[str] = set()
    months: list[tuple[str, str]] = []
    for ev in events:
        k = _month_key(ev.timestamp)
        if k not in seen_keys:
            seen_keys.add(k)
            months.append((k, _fmt_month(ev.timestamp)))
    return months


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def _is_image(mime: str | None) -> bool:
    return bool(mime and mime.startswith("image/"))


def _is_video(mime: str | None) -> bool:
    return bool(mime and mime.startswith("video/"))


def _is_audio(mime: str | None) -> bool:
    return bool(mime and mime.startswith("audio/"))


def _render_attachment(a: Asset) -> str:
    label = html.escape(a.filename)
    href = a.href  # already a data: URI or safe relative path
    size_label = _fmt_size(a.size) if a.size else ""
    mime_label = html.escape(a.mime_type or "application/octet-stream")
    if _is_image(a.mime_type):
        return (
            f'<figure class="att att-img">'
            f'<a href="{href}" target="_blank"><img src="{href}" alt="{label}" loading="lazy"></a>'
            f'<figcaption>{label} <span class="muted">({mime_label}, {size_label})</span></figcaption>'
            f'</figure>'
        )
    if _is_video(a.mime_type):
        return (
            f'<figure class="att att-video">'
            f'<video controls preload="metadata" src="{href}"></video>'
            f'<figcaption>{label} <span class="muted">({mime_label}, {size_label})</span></figcaption>'
            f'</figure>'
        )
    if _is_audio(a.mime_type):
        return (
            f'<figure class="att att-audio">'
            f'<audio controls preload="metadata" src="{href}"></audio>'
            f'<figcaption>{label} <span class="muted">({mime_label}, {size_label})</span></figcaption>'
            f'</figure>'
        )
    return (
        f'<a class="att att-file" href="{href}" download="{label}">'
        f'<span class="att-icon">📎</span>'
        f'<span class="att-meta"><strong>{label}</strong>'
        f'<span class="muted">{mime_label}{(", " + size_label) if size_label else ""}</span></span>'
        f'</a>'
    )


def _render_reactions(reactions: list[dict], pmap: dict[str, Participant]) -> str:
    if not reactions:
        return ""
    chips = []
    for r in reactions:
        value = html.escape(r.get("value") or "?")
        count = r.get("count") or len(r.get("participants") or [])
        names = ", ".join(
            (pmap[p].display if p in pmap else p) for p in (r.get("participants") or [])
        )
        title = html.escape(names) if names else ""
        chips.append(
            f'<span class="reaction" title="{title}">{value}'
            f'{f"<span class=count>{count}</span>" if count else ""}</span>'
        )
    return f'<div class="reactions">{"".join(chips)}</div>'


# Field-name variants for edit history. RSMF 2.0.0 spec uses "previous" / "new",
# but real-world exporters sometimes ship "old"/"after"/"body"/etc. Try them all.
_EDIT_PREV_KEYS = ("previous", "old", "before", "prior", "original_body", "body_before")
_EDIT_NEW_KEYS = ("new", "current", "after", "new_body", "body_after", "body")


def _pick_first(d: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _render_edit_history(edits: list[dict], pmap: dict[str, Participant]) -> str:
    """Render the edit history block.

    RSMF 2.0.0 only requires `participant` on each edit; `previous` and `new`
    are optional, and many exporters omit them (recording only that an edit
    happened, not the body deltas). The bubble already shows the final body,
    so when no body data is available we render a compact inline tag.
    """
    if not edits:
        return ""

    rows = []
    has_any_body = False
    for e in edits:
        prev = _pick_first(e, _EDIT_PREV_KEYS)
        new = _pick_first(e, _EDIT_NEW_KEYS)
        if prev or new:
            has_any_body = True
        rows.append((e, prev, new))

    if not has_any_body:
        # Compact form: "edited 10:24 · 09:06" — final body is in the bubble.
        stamps = []
        for e, _p, _n in rows:
            ts = e.get("timestamp") or ""
            short = ts.split("T")[1][:5] if "T" in ts else ts
            stamps.append(html.escape(short or "?"))
        n = len(stamps)
        label = "edited" if n == 1 else f"edited {n}×"
        return f'<div class="edits-tag" title="{html.escape("; ".join(e.get("timestamp","") for e,_,_ in rows))}">{label} · {", ".join(stamps)}</div>'

    # Full form: collapsible was/now history
    items = []
    for e, prev, new in rows:
        pid = e.get("participant") or ""
        name = pmap[pid].display if pid in pmap else pid
        ts = html.escape(e.get("timestamp") or "")
        items.append(
            f"<li><span class=muted>{ts} — {html.escape(name)} edited:</span>"
            f"{f'<div class=edit-prev>was: {html.escape(prev)}</div>' if prev else ''}"
            f"{f'<div class=edit-new>now: {html.escape(new)}</div>' if new else ''}"
            f"</li>"
        )
    return (
        '<details class="edits"><summary>edited (history)</summary>'
        f'<ol>{"".join(items)}</ol></details>'
    )


def _render_event(
    ev: RenderEvent,
    pmap: dict[str, Participant],
    custodian: Participant | None,
    parents_by_id: dict[str, RenderEvent],
    show_grouping: bool,
) -> str:
    if ev.etype != "message":
        return _render_system_event(ev)

    sender = ev.participant
    sender_name = sender.display if sender else "Unknown"
    color_idx = sender.color_idx if sender else 0
    side = "right" if (custodian and sender and sender.pid == custodian.pid) else "left"
    if ev.direction == "outgoing":
        side = "right"
    elif ev.direction == "incoming":
        side = "left"

    classes = ["msg", f"msg-{side}", f"p{color_idx}"]
    if ev.importance == "high":
        classes.append("important")
    if ev.deleted:
        classes.append("deleted")

    body_html: str
    if ev.deleted:
        body_html = '<em class="deleted-marker">[deleted]</em>'
    elif ev.body:
        body_html = _linkify(ev.body)
    else:
        body_html = ""

    # Reply quote
    reply_html = ""
    if ev.parent and ev.parent in parents_by_id:
        parent = parents_by_id[ev.parent]
        pname = parent.participant.display if parent.participant else "Unknown"
        snippet = (parent.body or ("[deleted]" if parent.deleted else ""))[:140]
        reply_html = (
            f'<a class="reply" href="#ev-{html.escape(parent.eid or "")}">'
            f'<span class="reply-name">↳ {html.escape(pname)}</span>'
            f'<span class="reply-snippet">{html.escape(snippet)}</span>'
            f'</a>'
        )
    elif ev.parent:
        reply_html = (
            f'<div class="reply reply-orphan">↳ replying to '
            f'<code>{html.escape(ev.parent)}</code></div>'
        )

    attachments_html = "".join(_render_attachment(a) for a in ev.attachments)

    header_html = (
        f'<div class="msg-head"><span class="sender">{html.escape(sender_name)}</span>'
        f'<span class="ts" title="{html.escape(ev.timestamp_raw or "")}">{_fmt_ts(ev.timestamp, ev.timestamp_raw)}</span></div>'
    ) if show_grouping else ""

    eid_attr = f'id="ev-{html.escape(ev.eid)}"' if ev.eid else ""

    return (
        f'<div {eid_attr} class="{ " ".join(classes) }">'
        f'  <div class="bub">'
        f'    {header_html}'
        f'    {reply_html}'
        f'    {f"<div class=body>{body_html}</div>" if body_html else ""}'
        f'    {f"<div class=attachments>{attachments_html}</div>" if attachments_html else ""}'
        f'    {_render_reactions(ev.reactions, pmap)}'
        f'    {_render_edit_history(ev.edits, pmap)}'
        f'  </div>'
        f'</div>'
    )


def _render_system_event(ev: RenderEvent) -> str:
    icons = {
        "join": "→",
        "leave": "←",
        "history": "ℹ︎",
        "disclaimer": "⚖︎",
        "unknown": "•",
    }
    icon = icons.get(ev.etype, "•")
    sender = ev.participant.display if ev.participant else None
    label = ev.etype.upper()
    body = html.escape(ev.body or "")
    sender_part = f"<strong>{html.escape(sender)}</strong> · " if sender else ""
    ts = f'<span class="ts">{_fmt_ts(ev.timestamp, ev.timestamp_raw)}</span>' if ev.timestamp_raw else ""
    return (
        f'<div class="sysrow"><span class="sys-pill">{icon} {label}</span>'
        f'<span class="sys-text">{sender_part}{body}</span>{ts}</div>'
    )


def _render_conversation(c: Conversation) -> str:
    pmap = {p.pid: p for p in c.participants}
    parents_by_id = {e.eid: e for e in c.events if e.eid}

    # For long, multi-month threads, mark month boundaries and offer a jump
    # nav in the header. Short threads keep the flat day dividers only.
    months = _ordered_months(c.events)
    group_by_month = len(months) > 1

    def _month_anchor(key: str) -> str:
        return f"month-{c.cid}-{key}"

    rows: list[str] = []
    last_month_key: str | None = None
    last_date_key: str | None = None
    last_sender_id: str | None = None
    last_ts: datetime | None = None

    for ev in c.events:
        # Month divider (anchor target for the jump nav)
        if group_by_month:
            month_key = _month_key(ev.timestamp)
            if month_key != last_month_key:
                anchor = html.escape(_month_anchor(month_key))
                rows.append(
                    f'<h3 class="month-divider" id="{anchor}">'
                    f'{html.escape(_fmt_month(ev.timestamp))}</h3>'
                )
                last_month_key = month_key

        # Date divider
        date_key = ev.timestamp.strftime("%Y-%m-%d") if ev.timestamp else "undated"
        if date_key != last_date_key:
            rows.append(f'<div class="date-divider"><span>{html.escape(_fmt_date(ev.timestamp))}</span></div>')
            last_date_key = date_key
            last_sender_id = None  # always show first message of a day's header

        # Group consecutive messages from same sender within 5 minutes
        cur_sender_id = ev.participant.pid if ev.participant else None
        time_gap_big = (
            last_ts is None or ev.timestamp is None or
            (ev.timestamp - last_ts).total_seconds() > 300
        )
        show_header = (
            ev.etype != "message"
            or cur_sender_id != last_sender_id
            or time_gap_big
        )

        rows.append(_render_event(ev, pmap, c.custodian, parents_by_id, show_grouping=show_header))

        if ev.etype == "message":
            last_sender_id = cur_sender_id
            last_ts = ev.timestamp

    msg_count = sum(1 for e in c.events if e.etype == "message")
    sys_count = len(c.events) - msg_count

    participants_chip = ", ".join(html.escape(p.display) for p in c.participants) or "—"
    custodian_chip = (
        f'<span class="muted">(custodian: {html.escape(c.custodian.display)})</span>'
        if c.custodian else ""
    )

    month_nav = (
        '<nav class="month-nav">' + "".join(
            f'<a href="#{html.escape(_month_anchor(key))}">{html.escape(lbl)}</a>'
            for key, lbl in months
        ) + '</nav>'
    ) if group_by_month else ""

    return (
        f'<section class="conv" id="conv-{html.escape(c.cid)}">'
        f'  <header class="conv-head">'
        f'    <h2>{html.escape(c.display)}</h2>'
        f'    <div class="conv-meta">'
        f'      <span class="pill">{html.escape(c.platform)}</span>'
        f'      {f"<span class=pill>{html.escape(c.ctype)}</span>" if c.ctype else ""}'
        f'      <span class="muted">{msg_count} messages{f" · {sys_count} system events" if sys_count else ""}</span>'
        f'    </div>'
        f'    <div class="conv-participants">'
        f'      <strong>Participants:</strong> {participants_chip} {custodian_chip}'
        f'    </div>'
        f'    {month_nav}'
        f'  </header>'
        f'  <div class="thread">{"".join(rows)}</div>'
        f'</section>'
    )


_CSS = """
:root {
  --bg: #f4f5f8;
  --panel: #ffffff;
  --text: #1d1d1f;
  --muted: #6b7280;
  --line: #e5e7eb;
  --accent: #007aff;
  --bub-out: #007aff;
  --bub-out-text: #fff;
  --bub-in: #e9e9eb;
  --bub-in-text: #1d1d1f;
  --shadow: 0 1px 2px rgba(0,0,0,.05);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e0f12;
    --panel: #17181d;
    --text: #ececec;
    --muted: #9aa0a6;
    --line: #2a2c33;
    --bub-in: #2a2c33;
    --bub-in-text: #ececec;
  }
}
* { box-sizing: border-box; }
html, body { margin:0; padding:0; background: var(--bg); color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
a { color: var(--accent); }
.muted { color: var(--muted); font-weight: 400; }
.page { max-width: 920px; margin: 0 auto; padding: 24px 16px 80px; }
.page-head { padding: 18px 20px; background: var(--panel); border: 1px solid var(--line);
  border-radius: 12px; box-shadow: var(--shadow); margin-bottom: 18px; }
.page-head h1 { margin: 0 0 6px; font-size: 20px; }
.page-head .sub { color: var(--muted); font-size: 13px; }
.toc { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.toc a { display: inline-block; padding: 4px 10px; border-radius: 999px;
  background: var(--bg); border: 1px solid var(--line); color: var(--text);
  text-decoration: none; font-size: 12px; }
.toc a:hover { border-color: var(--accent); color: var(--accent); }

section.conv { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  box-shadow: var(--shadow); margin-bottom: 18px; overflow: hidden; }
.conv-head { padding: 14px 18px 10px; border-bottom: 1px solid var(--line); position: sticky;
  top: 0; background: var(--panel); z-index: 1; }
.conv-head h2 { margin: 0; font-size: 16px; }
.conv-meta { margin-top: 4px; display:flex; gap: 6px; align-items: center; flex-wrap: wrap; font-size: 12px; }
.conv-participants { margin-top: 4px; font-size: 12px; color: var(--muted); }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px;
  background: var(--bg); border: 1px solid var(--line); font-size: 11px; }

.thread { padding: 12px 16px; }

.date-divider { text-align: center; margin: 16px 0 10px; position: relative; }
.date-divider::before { content: ""; position: absolute; left: 0; right: 0; top: 50%;
  height: 1px; background: var(--line); }
.date-divider span { position: relative; background: var(--panel); padding: 0 10px;
  color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }

.month-divider { margin: 26px 0 6px; padding: 6px 12px; font-size: 13px; font-weight: 700;
  letter-spacing: .03em; background: var(--accent); color: #fff; border-radius: 8px;
  position: sticky; top: 64px; z-index: 1; scroll-margin-top: 70px; }
.month-nav { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }
.month-nav a { font-size: 11px; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--line); color: var(--muted); text-decoration: none; }
.month-nav a:hover { border-color: var(--accent); color: var(--accent); }

.split-nav { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; font-size: 12px; }
.split-nav a { padding: 4px 10px; border-radius: 999px; border: 1px solid var(--line);
  color: var(--muted); text-decoration: none; }
.split-nav a:hover { border-color: var(--accent); color: var(--accent); }
.index-list { list-style: none; padding: 0; margin: 0; }
.index-list li { padding: 8px 12px; border-bottom: 1px solid var(--line); }
.index-list a { font-weight: 600; }

.msg { display: flex; margin: 2px 0; }
.msg-left { justify-content: flex-start; }
.msg-right { justify-content: flex-end; }
.bub { max-width: 70%; padding: 8px 12px; border-radius: 18px; word-wrap: break-word;
  position: relative; }
.msg-left  .bub { background: var(--bub-in); color: var(--bub-in-text);
  border-bottom-left-radius: 4px; }
.msg-right .bub { background: var(--bub-out); color: var(--bub-out-text);
  border-bottom-right-radius: 4px; }
.msg-right .bub a { color: #fff; text-decoration: underline; }
.msg-right .bub .muted { color: rgba(255,255,255,.75); }

/* Per-participant accent for left bubbles (group chats) */
.msg-left.p0 .bub { background: #e9e9eb; }
.msg-left.p1 .bub { background: #e0f2ff; }
.msg-left.p2 .bub { background: #ffe9e9; }
.msg-left.p3 .bub { background: #e9ffe9; }
.msg-left.p4 .bub { background: #fff5d6; }
.msg-left.p5 .bub { background: #f0e0ff; }
.msg-left.p6 .bub { background: #d0f0f0; }
.msg-left.p7 .bub { background: #ffe0f0; }
@media (prefers-color-scheme: dark) {
  .msg-left.p0 .bub { background: #2a2c33; }
  .msg-left.p1 .bub { background: #1c3445; }
  .msg-left.p2 .bub { background: #46232a; }
  .msg-left.p3 .bub { background: #1d3a25; }
  .msg-left.p4 .bub { background: #4a3d12; }
  .msg-left.p5 .bub { background: #361c4a; }
  .msg-left.p6 .bub { background: #143838; }
  .msg-left.p7 .bub { background: #491c33; }
  .msg-left .bub { color: var(--text); }
  .msg-left .bub a { color: #6cb6ff; }
}

.msg-head { display: flex; gap: 10px; align-items: baseline; margin-bottom: 2px; font-size: 11px; }
.sender { font-weight: 600; }
.ts { color: var(--muted); font-size: 11px; }
.msg-right .ts { color: rgba(255,255,255,.75); }

.body { white-space: pre-wrap; }
.deleted .body, .deleted-marker { color: var(--muted); font-style: italic; }

.attachments { margin-top: 6px; display: flex; flex-direction: column; gap: 6px; }
.att { display: block; }
.att-img img, .att-video video { max-width: 100%; max-height: 360px; border-radius: 10px;
  background: #000; }
.att audio { width: 100%; }
.att figcaption { font-size: 11px; color: var(--muted); margin-top: 2px; }
.msg-right .att figcaption { color: rgba(255,255,255,.8); }
.att-file { display: flex; gap: 10px; align-items: center; padding: 8px 10px;
  background: rgba(0,0,0,.04); border-radius: 10px; text-decoration: none; color: inherit; }
.msg-right .att-file { background: rgba(255,255,255,.15); }
.att-file .att-icon { font-size: 18px; }
.att-meta { display: flex; flex-direction: column; line-height: 1.2; }

.reactions { margin-top: 4px; display: flex; gap: 4px; flex-wrap: wrap; }
.reaction { background: rgba(0,0,0,.06); padding: 1px 8px; border-radius: 999px;
  font-size: 12px; }
.msg-right .reaction { background: rgba(255,255,255,.18); }
.reaction .count { margin-left: 4px; opacity: .7; font-size: 10px; }

.edits { margin-top: 6px; font-size: 11px; }
.edits summary { cursor: pointer; color: var(--muted); }
.edits-tag { margin-top: 4px; font-size: 10px; color: var(--muted); font-style: italic; }
.msg-right .edits-tag { color: rgba(255,255,255,.75); }
.edit-prev, .edit-new { white-space: pre-wrap; padding-left: 8px; border-left: 2px solid var(--line); }
.edit-prev { color: var(--muted); }
.msg-right .edits summary, .msg-right .edit-prev { color: rgba(255,255,255,.75); }

.reply { display: block; padding: 4px 8px; margin-bottom: 4px; border-left: 3px solid var(--line);
  background: rgba(0,0,0,.04); border-radius: 6px; font-size: 12px; text-decoration: none;
  color: inherit; }
.msg-right .reply { background: rgba(255,255,255,.18); border-left-color: rgba(255,255,255,.5); }
.reply-name { font-weight: 600; display: block; }
.reply-snippet { color: var(--muted); }
.msg-right .reply-snippet { color: rgba(255,255,255,.8); }
.reply-orphan { font-style: italic; }

.important .bub { box-shadow: 0 0 0 2px #ff9500 inset; }

.sysrow { display: flex; align-items: center; gap: 8px; justify-content: center;
  margin: 10px 0; padding: 6px 10px; font-size: 11px; color: var(--muted);
  background: rgba(0,0,0,.03); border-radius: 999px; max-width: max-content;
  margin-left: auto; margin-right: auto; }
@media (prefers-color-scheme: dark) {
  .sysrow { background: rgba(255,255,255,.05); }
  .att-file { background: rgba(255,255,255,.06); }
  .reaction { background: rgba(255,255,255,.08); }
  .reply { background: rgba(255,255,255,.05); }
}
.sys-pill { font-weight: 600; letter-spacing: .04em; }
.sys-text { color: var(--text); }
.sysrow .ts { margin-left: auto; }

footer.foot { text-align: center; color: var(--muted); font-size: 11px; margin-top: 24px; }
"""


def render_html(
    conversations: list[Conversation],
    summary: dict,
    *,
    source_name: str,
    title: str | None = None,
    nav_html: str | None = None,
) -> str:
    title_text = title or f"RSMF — {source_name}"
    toc = "".join(
        f'<a href="#conv-{html.escape(c.cid)}">{html.escape(c.display)}</a>'
        for c in conversations
    )
    body_sections = "".join(_render_conversation(c) for c in conversations)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title_text)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page">
  <div class="page-head">
    <h1>{html.escape(title_text)}</h1>
    <div class="sub">
      RSMF v{html.escape(summary['version'])} ·
      {summary['conversation_count']} conversation(s) ·
      {summary['message_count']} messages ·
      {summary['participant_count']} participants ·
      source: <code>{html.escape(source_name)}</code>
    </div>
    {nav_html or ''}
    {f'<div class="toc">{toc}</div>' if len(conversations) > 1 else ''}
  </div>
  {body_sections}
  <footer class="foot">Generated by rsmf_viewer · {generated}</footer>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
#
# A forensic-friendly counterpart to render_html(). Conversation metadata goes
# into YAML frontmatter; every message is attributed to its sender and stamped
# with its time; attachments render inline (images) or as links, each credited
# to the sender. Reuses the resolved Asset.href (relative path under --no-embed,
# data: URI under embed) so the export pipeline is shared with HTML.


# Inline chars that would otherwise be interpreted as Markdown emphasis/code/
# links, or as raw HTML. Escaped everywhere inside body/label text.
_MD_INLINE_RE = re.compile(r"([\\`*_\[\]<>|])")

# Unusual vertical separators that editors flag in the output file: vertical tab,
# form feed, NEL, LINE SEPARATOR (U+2028, iMessage's soft line break), and
# PARAGRAPH SEPARATOR (U+2029). Normalized to ordinary newlines/spaces on output.
_VWS_RE = re.compile("[\x0b\x0c\x85\u2028\u2029]")


def _to_newlines(text: str) -> str:
    """Map CR/CRLF and unusual vertical separators to plain '\\n' (multi-line)."""
    return _VWS_RE.sub("\n", text.replace("\r\n", "\n").replace("\r", "\n"))


def _to_space(text: str) -> str:
    """Collapse CR and unusual vertical separators to spaces (single-line)."""
    return _VWS_RE.sub(" ", text.replace("\r\n", " ").replace("\r", " "))


def _md_inline(text: str) -> str:
    """Backslash-escape inline Markdown/HTML control characters in a single-line
    string, flattening any stray vertical separators to spaces first."""
    return _MD_INLINE_RE.sub(r"\\\1", _to_space(text))


def _md_escape(text: str) -> str:
    """Escape body text: inline chars everywhere, plus block markers (#, +, -,
    ordered-list 'N.') at the start of each line. URLs are left intact so
    renderers auto-link them. Unusual line terminators are first normalized to
    real newlines so they don't leak into the output file."""
    lines = []
    for line in _to_newlines(text).split("\n"):
        line = _md_inline(line)  # also neutralises a leading '>' (blockquote)
        line = re.sub(r"^(\s*)([#+])", r"\1\\\2", line)
        line = re.sub(r"^(\s*)(-)(\s)", r"\1\\\2\3", line)
        line = re.sub(r"^(\s*)(\d+)([.)])(\s)", r"\1\2\\\3\4", line)
        lines.append(line)
    return "\n".join(lines)


def _md_body(text: str) -> str:
    """Escape a message body and preserve its internal line breaks as Markdown
    hard breaks (two trailing spaces)."""
    return _md_escape(text).replace("\n", "  \n")


def _md_url(href: str) -> str:
    """Wrap a link target in angle brackets when it contains characters that
    would break inline-link parsing (spaces, parentheses)."""
    return f"<{href}>" if re.search(r"[\s()]", href) else href


def _md_anchor(text: str) -> str:
    """GitHub-style heading slug for the table of contents (best-effort; most
    Markdown renderers use this same algorithm)."""
    s = re.sub(r"[^\w\s-]", "", text.strip().lower())
    return re.sub(r"\s+", "-", s)


def _yaml_str(value: object) -> str:
    """Render a scalar as a safe single-line YAML string."""
    s = _to_space(str(value))
    if s == "":
        return '""'
    if s != s.strip() or s[0] in "?-:#&*!|>%@`\"'[]{}," or re.search(r'[:#\n"]', s):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def _fmt_time(dt: datetime | None, raw: str | None) -> str:
    if dt is None:
        return raw or ""
    return dt.strftime("%H:%M:%S %Z").strip()


def _conv_date_range(c: Conversation) -> tuple[datetime | None, datetime | None]:
    stamps = [e.timestamp for e in c.events if e.timestamp]
    if not stamps:
        return None, None
    return min(stamps), max(stamps)


def _first_date_iso(conversations: list[Conversation]) -> str | None:
    """Earliest event date (YYYY-MM-DD) across the given conversations, used as
    the Obsidian `date:` field so a thread note sorts on the vault timeline."""
    stamps = [e.timestamp for c in conversations for e in c.events if e.timestamp]
    return min(stamps).date().isoformat() if stamps else None


# Attachment -> Obsidian tag, mirroring the obsidian-vault build's has/* scheme.
_TAG_SCREENSHOTS = "has/screenshots"
_TAG_AUDIO = "has/audio"
_TAG_VIDEO = "has/video"
_TAG_DOCUMENTS = "has/documents"
# Emitted in this fixed order so notes diff cleanly.
_HAS_TAG_ORDER = (_TAG_SCREENSHOTS, _TAG_AUDIO, _TAG_VIDEO, _TAG_DOCUMENTS)
_HAS_EXT = {
    **dict.fromkeys(("png", "jpg", "jpeg", "heic", "gif"), _TAG_SCREENSHOTS),
    **dict.fromkeys(("m4a", "mp3", "wav", "aac"), _TAG_AUDIO),
    **dict.fromkeys(("mov", "mp4"), _TAG_VIDEO),
    **dict.fromkeys(("pdf", "docx"), _TAG_DOCUMENTS),
}
_DOC_MIMES = (
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
)


def _has_tag(a: Asset) -> str | None:
    if _is_image(a.mime_type):
        return _TAG_SCREENSHOTS
    if _is_audio(a.mime_type):
        return _TAG_AUDIO
    if _is_video(a.mime_type):
        return _TAG_VIDEO
    ext = Path(a.filename).suffix.lower().lstrip(".")
    if ext in _HAS_EXT:
        return _HAS_EXT[ext]
    if a.mime_type in _DOC_MIMES:
        return _TAG_DOCUMENTS
    return None


def _vault_tags(conversations: list[Conversation]) -> list[str]:
    """Distinct has/* attachment tags present in these conversations, in the
    vault's canonical order."""
    found = {
        tag for c in conversations for e in c.events for a in e.attachments
        if (tag := _has_tag(a))
    }
    return [t for t in _HAS_TAG_ORDER if t in found]


def _md_attachment(a: Asset, sender_name: str) -> str:
    label = a.filename
    size_label = _fmt_size(a.size) if a.size else ""
    mime_label = a.mime_type or "application/octet-stream"
    meta = mime_label + (f", {size_label}" if size_label else "")
    attrib = f" — sent by {sender_name}" if sender_name else ""
    url = _md_url(a.href)
    if _is_image(a.mime_type):
        alt = label.replace("[", "").replace("]", "")
        # <sub> caption is HTML, so escape its text rather than Markdown-escape.
        return (
            f"![{alt}]({url})\n"
            f"<sub>📎 {html.escape(label)} · {html.escape(meta)}{html.escape(attrib)}</sub>"
        )
    link_text = label.replace("[", "").replace("]", "")
    return f"[📎 {link_text}]({url}) ({meta}){attrib}"


def _md_reactions(reactions: list[dict], pmap: dict[str, Participant]) -> str:
    if not reactions:
        return ""
    chips = []
    for r in reactions:
        value = r.get("value") or "?"
        count = r.get("count") or len(r.get("participants") or [])
        names = ", ".join(
            (pmap[p].display if p in pmap else p) for p in (r.get("participants") or [])
        )
        chip = value
        if count:
            chip += f" ×{count}"
        if names:
            chip += f" ({names})"
        chips.append(chip)
    return "> " + " · ".join(chips)


def _md_edits(edits: list[dict], pmap: dict[str, Participant]) -> str:
    """Markdown counterpart to _render_edit_history: a compact inline tag when
    only timestamps are recorded, a collapsible <details> block when body
    deltas are present."""
    if not edits:
        return ""
    rows = []
    has_any_body = False
    for e in edits:
        prev = _pick_first(e, _EDIT_PREV_KEYS)
        new = _pick_first(e, _EDIT_NEW_KEYS)
        if prev or new:
            has_any_body = True
        rows.append((e, prev, new))

    if not has_any_body:
        stamps = []
        for e, _p, _n in rows:
            ts = e.get("timestamp") or ""
            short = ts.split("T")[1][:5] if "T" in ts else ts
            stamps.append(short or "?")
        n = len(stamps)
        label = "edited" if n == 1 else f"edited {n}×"
        return f"_({label} · {', '.join(stamps)})_"

    items = []
    for e, prev, new in rows:
        pid = e.get("participant") or ""
        name = pmap[pid].display if pid in pmap else pid
        ts = e.get("timestamp") or ""
        line = f"<li>{html.escape(ts)} — {html.escape(name)} edited:"
        if prev:
            line += f"<br>was: {html.escape(prev)}"
        if new:
            line += f"<br>now: {html.escape(new)}"
        items.append(line + "</li>")
    return (
        "<details><summary>edited (history)</summary>"
        f"<ol>{''.join(items)}</ol></details>"
    )


def _md_system_event(ev: RenderEvent) -> str:
    sender = ev.participant.display if ev.participant else None
    bits = [ev.etype.upper()]
    if sender:
        bits.append(sender)
    if ev.body:
        bits.append(ev.body)
    ts = _fmt_time(ev.timestamp, ev.timestamp_raw)
    if ts:
        bits.append(ts)
    return f"_— {_md_inline(' · '.join(bits))}_"


def _md_event(
    ev: RenderEvent,
    pmap: dict[str, Participant],
    parents_by_id: dict[str, RenderEvent],
) -> str:
    if ev.etype != "message":
        return _md_system_event(ev)

    sender = ev.participant
    sender_name = sender.display if sender else "Unknown"

    head = f"**{_md_inline(sender_name)}**"
    time_str = _fmt_time(ev.timestamp, ev.timestamp_raw)
    if time_str:
        head += f" · {time_str}"
    if ev.importance == "high":
        head += " · ❗"
    if ev.parent and ev.parent in parents_by_id:
        parent = parents_by_id[ev.parent]
        pname = parent.participant.display if parent.participant else "Unknown"
        head += f" · ↩ replying to {_md_inline(pname)}"
    elif ev.parent:
        head += f" · ↩ replying to `{_md_inline(ev.parent)}`"

    parts = [head]
    if ev.deleted:
        parts.append("_[message deleted]_")
    elif ev.body:
        parts.append(_md_body(ev.body))

    parts.extend(_md_attachment(a, sender_name) for a in ev.attachments)

    react = _md_reactions(ev.reactions, pmap)
    if react:
        parts.append(react)
    edits = _md_edits(ev.edits, pmap)
    if edits:
        parts.append(edits)

    return "\n\n".join(parts)


def _md_conversation(c: Conversation, *, in_frontmatter: bool) -> str:
    pmap = {p.pid: p for p in c.participants}
    parents_by_id = {e.eid: e for e in c.events if e.eid}

    out = [f"## {_md_inline(c.display)}"]

    # For single-conversation exports the metadata is carried in the document
    # frontmatter, so we skip the inline bullet block to avoid duplication.
    if not in_frontmatter:
        start, end = _conv_date_range(c)
        msg_count = sum(1 for e in c.events if e.etype == "message")
        platform_line = f"- **Platform:** {_md_inline(c.platform)}"
        if c.ctype:
            platform_line += f" · **Type:** {_md_inline(c.ctype)}"
        meta = [platform_line]
        if c.custodian:
            meta.append(f"- **Custodian:** {_md_inline(c.custodian.display)}")
        if start and end:
            meta.append(
                f"- **Dates:** {start.date()} → {end.date()} · "
                f"**Messages:** {msg_count}"
            )
        else:
            meta.append(f"- **Messages:** {msg_count}")
        plist = ", ".join(
            (f"{p.display} <{p.email}>" if p.email else p.display)
            for p in c.participants
        ) or "—"
        meta.append(f"- **Participants:** {_md_inline(plist)}")
        out.append("\n".join(meta))

    # For long, multi-month threads, group the timeline under "Month YYYY"
    # headers and offer a jump index. Short threads keep the flat day dividers.
    months = _ordered_months(c.events)
    group_by_month = len(months) > 1
    if group_by_month:
        jump = " · ".join(f"[{lbl}](#{_md_anchor(lbl)})" for _k, lbl in months)
        out.append(f"**Jump to month:** {jump}")
    day_heading = "####" if group_by_month else "###"

    last_month_key: str | None = None
    last_date_key: str | None = None
    for ev in c.events:
        if group_by_month:
            month_key = _month_key(ev.timestamp)
            if month_key != last_month_key:
                out.append(f"### {_fmt_month(ev.timestamp)}")
                last_month_key = month_key
                last_date_key = None  # re-emit the day header under each month
        date_key = ev.timestamp.strftime("%Y-%m-%d") if ev.timestamp else "undated"
        if date_key != last_date_key:
            out.append(f"{day_heading} {_fmt_date(ev.timestamp)}")
            last_date_key = date_key
        out.append(_md_event(ev, pmap, parents_by_id))

    return "\n\n".join(out)


def _md_frontmatter(
    conversations: list[Conversation],
    summary: dict,
    *,
    source_name: str,
    generated: str,
    period: str | None = None,
) -> str:
    # Lead with the Obsidian-vault schema fields (date / type / summary / tags)
    # so these notes drop into the vault and its summarize pass cleanly; the
    # rsmf-specific provenance follows. `summary` is an empty placeholder filled
    # later by the summarize.py LLM pass.
    lines = ["---"]
    date_iso = _first_date_iso(conversations)
    if date_iso:
        lines.append(f"date: {date_iso}")
    lines.append("type: message-thread")
    lines.append('summary: ""')
    tags = _vault_tags(conversations)
    if tags:
        lines.append("tags:")
        lines += [f"  - {t}" for t in tags]
    lines += [
        "generator: rsmf_viewer",
        f"source: {_yaml_str(source_name)}",
        f"rsmf_version: {_yaml_str(summary['version'])}",
        f"generated: {generated}",
    ]
    if period:
        lines.append(f"period: {_yaml_str(period)}")
    lines += [
        f"conversations: {summary['conversation_count']}",
        f"messages: {summary['message_count']}",
        f"participant_count: {summary['participant_count']}",
    ]
    if len(conversations) == 1:
        c = conversations[0]
        start, end = _conv_date_range(c)
        lines.append("conversation:")
        lines.append(f"  id: {_yaml_str(c.cid)}")
        lines.append(f"  display: {_yaml_str(c.display)}")
        lines.append(f"  platform: {_yaml_str(c.platform)}")
        if c.ctype:
            lines.append(f"  type: {_yaml_str(c.ctype)}")
        if c.custodian:
            lines.append(f"  custodian: {_yaml_str(c.custodian.display)}")
        if start:
            lines.append(f"  date_start: {start.isoformat()}")
        if end:
            lines.append(f"  date_end: {end.isoformat()}")
        lines.append(
            f"  message_count: {sum(1 for e in c.events if e.etype == 'message')}"
        )
        lines.append("participants:")
        for p in c.participants:
            if p.email:
                lines.append(
                    f"  - {{ display: {_yaml_str(p.display)}, "
                    f"email: {_yaml_str(p.email)} }}"
                )
            else:
                lines.append(f"  - {{ display: {_yaml_str(p.display)} }}")
    lines.append("---")
    return "\n".join(lines)


def render_markdown(
    conversations: list[Conversation],
    summary: dict,
    *,
    source_name: str,
    title: str | None = None,
    period: str | None = None,
    nav: str | None = None,
) -> str:
    title_text = title or f"RSMF — {source_name}"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    single = len(conversations) == 1

    parts = [
        _md_frontmatter(
            conversations, summary, source_name=source_name,
            generated=generated, period=period,
        ),
        f"# {_md_inline(title_text)}",
        f"_RSMF v{summary['version']} · {summary['conversation_count']} "
        f"conversation(s) · {summary['message_count']} messages · "
        f"{summary['participant_count']} participants · source: `{source_name}`_",
    ]
    if nav:
        parts.append(nav)
    if not single:
        toc = "\n".join(
            f"- [{_md_inline(c.display)}](#{_md_anchor(c.display)})"
            for c in conversations
        )
        parts.append("## Contents\n\n" + toc)
    parts.extend(_md_conversation(c, in_frontmatter=single) for c in conversations)
    parts.append(f"_Generated by rsmf_viewer · {generated}_")
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Period splitting (one output file per month / year, plus an index)
# ---------------------------------------------------------------------------


def _period_key(dt: datetime | None, granularity: str) -> str:
    if dt is None:
        return "undated"
    return dt.strftime("%Y") if granularity == "year" else dt.strftime("%Y-%m")


def _period_label(key: str, granularity: str) -> str:
    if key == "undated":
        return "Undated"
    if granularity == "year":
        return key
    return datetime.strptime(key, "%Y-%m").strftime("%B %Y")


def _split_by_period(
    conversations: list[Conversation], granularity: str
) -> list[tuple[str, str, list[Conversation]]]:
    """Bucket events into ordered (key, label, [filtered Conversation]) periods.

    Each bucket holds shallow copies of the conversations that have events in
    that period, carrying only those events — so every period file stays a
    self-contained, attributable slice of the original conversation(s).
    """
    keys = sorted(
        {_period_key(e.timestamp, granularity) for c in conversations for e in c.events},
        key=lambda k: (k == "undated", k),  # chronological, undated bucket last
    )
    buckets: list[tuple[str, str, list[Conversation]]] = []
    for key in keys:
        sliced = [
            replace(c, events=evs)
            for c in conversations
            if (evs := [e for e in c.events
                        if _period_key(e.timestamp, granularity) == key])
        ]
        if sliced:
            buckets.append((key, _period_label(key, granularity), sliced))
    return buckets


def _parse_duration(spec: str) -> timedelta:
    """Parse a gap like '24h', '2d', '90m', '45s', or a bare number (hours)."""
    m = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*', spec.lower())
    if not m:
        raise ValueError(f"invalid duration {spec!r} (try 24h, 2d, 90m)")
    seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2) or "h"]
    return timedelta(seconds=float(m.group(1)) * seconds)


def _session_label(start: datetime, end: datetime) -> str:
    if start.date() == end.date():
        return start.strftime("%a, %b %-d, %Y")
    if start.year == end.year:
        return f"{start.strftime('%b %-d')} – {end.strftime('%b %-d, %Y')}"
    return f"{start.strftime('%b %-d, %Y')} – {end.strftime('%b %-d, %Y')}"


def _is_custodian_msg(e: RenderEvent, custodian: Participant | None) -> bool:
    """True if event `e` is a message sent by the custodian (device owner)."""
    if e.etype != "message":
        return False
    if e.direction == "outgoing":
        return True
    return bool(custodian and e.participant and e.participant.pid == custodian.pid)


def _session_starts(
    conversations: list[Conversation], gap: timedelta, require_reply: bool
) -> list[datetime]:
    """Compute the start timestamp of each session from the combined message
    timeline. A gap longer than `gap` begins a new session — but when
    `require_reply` is set, only once the current session already contains a
    custodian message, so a one-sided run of template/repeated messages stays a
    single conversation regardless of how long it spans."""
    stream = sorted(
        ((e.timestamp, _is_custodian_msg(e, c.custodian))
         for c in conversations for e in c.events
         if e.timestamp and e.etype == "message"),
        key=lambda t: t[0],
    )
    if not stream:
        return []
    starts = [stream[0][0]]
    has_reply = stream[0][1]
    prev = stream[0][0]
    for ts, is_cust in stream[1:]:
        if ts - prev > gap and (not require_reply or has_reply):
            starts.append(ts)
            has_reply = is_cust
        else:
            has_reply = has_reply or is_cust
        prev = ts
    return starts


def _split_by_session(
    conversations: list[Conversation], gap: timedelta, require_reply: bool = False
) -> list[tuple[str, str, list[Conversation]]]:
    """Bucket events into conversational sessions on the combined timeline. A new
    file begins after a silence longer than `gap`; with `require_reply`, a gap
    only splits once the custodian has replied within the current session. Every
    dated event is assigned to the last session that started at or before it; an
    'undated' bucket collects events without a timestamp."""
    starts = _session_starts(conversations, gap, require_reply)

    buckets: list[tuple[str, str, list[Conversation]]] = []
    if starts:
        pad = max(4, len(str(len(starts))))
        # Per session, the conversations sliced to the events that fall in it.
        sliced_by_session: list[list[Conversation]] = [[] for _ in starts]
        for c in conversations:
            per: list[list[RenderEvent]] = [[] for _ in starts]
            for e in c.events:
                if e.timestamp:
                    idx = max(0, bisect.bisect_right(starts, e.timestamp) - 1)
                    per[idx].append(e)
            for idx, evs in enumerate(per):
                if evs:
                    sliced_by_session[idx].append(replace(c, events=evs))
        for i, convs in enumerate(sliced_by_session, 1):
            stamps = [e.timestamp for cc in convs for e in cc.events if e.timestamp]
            start, end = min(stamps), max(stamps)
            key = f"{i:0{pad}d}_{start.date().isoformat()}"
            buckets.append((key, _session_label(start, end), convs))

    undated = [
        replace(c, events=evs)
        for c in conversations
        if (evs := [e for e in c.events if not e.timestamp])
    ]
    if undated:
        buckets.append(("zzzz_undated", "Undated", undated))
    return buckets


def _has_custodian(conversations: list[Conversation]) -> bool:
    """Whether a custodian is identifiable for reply-aware session splitting."""
    return any(c.custodian for c in conversations)


def _period_summary(conversations: list[Conversation], base_summary: dict) -> dict:
    """A summary dict scoped to a single period's sliced conversations."""
    pids = {p.pid for c in conversations for p in c.participants}
    return {
        **base_summary,
        "conversation_count": len(conversations),
        "message_count": sum(
            1 for c in conversations for e in c.events if e.etype == "message"
        ),
        "participant_count": len(pids),
    }


def _bucket_msg_count(conversations: list[Conversation]) -> int:
    return sum(1 for c in conversations for e in c.events if e.etype == "message")


def _split_nav_md(buckets: list, i: int, ext: str) -> str:
    links = [f"[↑ Index](index{ext})"]
    if i > 0:
        pk, pl, _ = buckets[i - 1]
        links.append(f"[← {pl}]({pk}{ext})")
    if i < len(buckets) - 1:
        nk, nl, _ = buckets[i + 1]
        links.append(f"[{nl} →]({nk}{ext})")
    return " · ".join(links)


def _split_nav_html(buckets: list, i: int, ext: str) -> str:
    links = [f'<a href="index{ext}">↑ Index</a>']
    if i > 0:
        pk, pl, _ = buckets[i - 1]
        links.append(f'<a href="{html.escape(pk + ext)}">← {html.escape(pl)}</a>')
    if i < len(buckets) - 1:
        nk, nl, _ = buckets[i + 1]
        links.append(f'<a href="{html.escape(nk + ext)}">{html.escape(nl)} →</a>')
    return f'<nav class="split-nav">{"".join(links)}</nav>'


def _md_index(
    conversations: list[Conversation],
    buckets: list,
    summary: dict,
    *,
    source_name: str,
    title: str | None,
    ext: str,
) -> str:
    title_text = (title or f"RSMF — {source_name}") + " — Index"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Primary index: a Dataview table that reads each period note's summary and
    # tags live from frontmatter (so it reflects summarize.py without rebuilding),
    # mirroring the obsidian-vault dashboard. Scoped to sibling period notes in
    # this folder so it works wherever the export is dropped in the vault.
    dataview = (
        "```dataview\n"
        "TABLE WITHOUT ID\n"
        "  file.link AS Period,\n"
        "  summary AS Summary,\n"
        "  messages AS Msgs,\n"
        "  tags AS Tags\n"
        'WHERE type = "message-thread" AND file.folder = this.file.folder '
        "AND file.name != this.file.name\n"
        "SORT file.name ASC\n"
        "```"
    )
    # Static fallback for viewers without Dataview (GitHub, plain Markdown).
    rows = "\n".join(
        f"| [{_md_inline(label)}]({key}{ext}) | {_bucket_msg_count(convs)} |"
        for key, label, convs in buckets
    )
    return "\n\n".join([
        _md_frontmatter(
            conversations, summary, source_name=source_name, generated=generated
        ),
        f"# {_md_inline(title_text)}",
        f"_RSMF v{summary['version']} · {summary['message_count']} messages across "
        f"{len(buckets)} period(s) · source: `{source_name}`_",
        dataview,
        "_Without the Dataview plugin, use the plain list below._",
        "| Period | Messages |\n| --- | ---: |\n" + rows,
        f"_Generated by rsmf_viewer · {generated}_",
    ]) + "\n"


def _html_index(
    buckets: list,
    summary: dict,
    *,
    source_name: str,
    title: str | None,
    ext: str,
) -> str:
    title_text = (title or f"RSMF — {source_name}") + " — Index"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    items = "".join(
        f'<li><a href="{html.escape(key + ext)}">{html.escape(label)}</a> '
        f'<span class="muted">— {_bucket_msg_count(convs)} messages</span></li>'
        for key, label, convs in buckets
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title_text)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page">
  <div class="page-head">
    <h1>{html.escape(title_text)}</h1>
    <div class="sub">
      RSMF v{html.escape(summary['version'])} ·
      {summary['message_count']} messages across {len(buckets)} period(s) ·
      source: <code>{html.escape(source_name)}</code>
    </div>
  </div>
  <ul class="index-list">{items}</ul>
  <footer class="foot">Generated by rsmf_viewer · {generated}</footer>
</div>
</body>
</html>
"""


def _write_split_format(
    fmt: str,
    out_dir: Path,
    conversations: list[Conversation],
    buckets: list,
    summary: dict,
    *,
    source_name: str,
    title: str | None,
) -> None:
    ext = ".md" if fmt == "markdown" else ".html"
    base_title = title or f"RSMF — {source_name}"
    for i, (key, label, convs) in enumerate(buckets):
        psum = _period_summary(convs, summary)
        ptitle = f"{base_title} — {label}"
        if fmt == "markdown":
            doc = render_markdown(
                convs, psum, source_name=source_name, title=ptitle,
                period=label, nav=_split_nav_md(buckets, i, ext),
            )
        else:
            doc = render_html(
                convs, psum, source_name=source_name, title=ptitle,
                nav_html=_split_nav_html(buckets, i, ext),
            )
        (out_dir / (key + ext)).write_text(doc, encoding="utf-8")

    if fmt == "markdown":
        index = _md_index(conversations, buckets, summary,
                          source_name=source_name, title=title, ext=ext)
    else:
        index = _html_index(buckets, summary,
                            source_name=source_name, title=title, ext=ext)
    (out_dir / f"index{ext}").write_text(index, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _convert_one(
    src: Path,
    out: Path,
    *,
    embed: bool,
    fmt: str = "auto",
    split: str = "none",
    title: str | None = None,
    me: str | None = None,
    me_name: str | None = None,
    assume_tz=None,
    session_gap: timedelta | None = None,
    require_reply: bool = False,
) -> dict:
    if fmt == "auto":
        fmt = "markdown" if out.suffix.lower() in (".md", ".markdown") else "html"
    formats = ["html", "markdown"] if fmt == "both" else [fmt]

    # base = output path without extension. For an un-split export it is the
    # file stem (both formats + assets share it). For a split export it becomes
    # a directory holding the per-period files, an index, and the assets dir.
    base = out.with_suffix("")
    out_dir = base if split != "none" else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    asset_parent = out_dir if out_dir is not None else base.parent
    asset_dir = None if embed else asset_parent / (base.name + "_assets")

    manifest, zf = parse_rsmf(src)
    try:
        conversations, summary = build_conversations(
            manifest, zf, embed_assets=embed, asset_dir=asset_dir, me=me,
            me_name=me_name, assume_tz=assume_tz,
        )
    finally:
        zf.close()

    if split != "none":
        if split == "session":
            reply_aware = require_reply
            if reply_aware and not _has_custodian(conversations):
                logger.warning("--require-reply ignored: no custodian identified "
                               "(use --me / --me-name)")
                reply_aware = False
            buckets = _split_by_session(
                conversations, session_gap or timedelta(hours=24), reply_aware,
            )
        else:
            buckets = _split_by_period(conversations, split)
        for f in formats:
            _write_split_format(f, out_dir, conversations, buckets, summary,
                                source_name=src.name, title=title)
        return summary

    if "html" in formats:
        doc = render_html(conversations, summary, source_name=src.name, title=title)
        base.with_suffix(".html").write_text(doc, encoding="utf-8")
    if "markdown" in formats:
        doc = render_markdown(conversations, summary, source_name=src.name, title=title)
        base.with_suffix(".md").write_text(doc, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rsmf_viewer",
        description="Convert .rsmf message exports to a pretty self-contained "
                    "HTML page or a Markdown document.",
    )
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="One or more .rsmf or .zip RSMF files.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (single input only). The format is "
                             "inferred from the extension (.md/.markdown -> "
                             "Markdown, else HTML) unless --format is given. "
                             "Default: <input>.html")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Write outputs into this directory (batch mode).")
    parser.add_argument("-f", "--format", choices=["html", "markdown", "md", "both"],
                        default=None,
                        help="Output format. Default: inferred from --output's "
                             "extension, otherwise html. 'both' writes <stem>.html "
                             "and <stem>.md sharing one assets directory.")
    parser.add_argument("--no-embed", action="store_true",
                        help="Write attachments to <output>_assets/ instead of "
                             "embedding them as data: URIs. Recommended for "
                             "Markdown (data: URIs don't render on GitHub and "
                             "some Markdown viewers).")
    parser.add_argument("--split", choices=["none", "month", "year", "session"],
                        default="none",
                        help="Split the output into one file per period instead "
                             "of a single document. Writes <output-stem>/ as a "
                             "directory of per-period files plus an index, sharing "
                             "one assets directory. 'month'/'year' use calendar "
                             "boundaries; 'session' breaks the thread at natural "
                             "gaps (see --session-gap). Useful when a single "
                             "export is too long to open or render.")
    parser.add_argument("--session-gap", type=str, default="24h",
                        help="With --split session, start a new file after a "
                             "silence longer than this (e.g. 24h, 2d, 90m; a bare "
                             "number is hours). Default: 24h.")
    parser.add_argument("--require-reply", action="store_true",
                        help="With --split session, a gap only starts a new file "
                             "once the custodian has replied within the current "
                             "session — so a one-sided run of template/repeated "
                             "messages stays one conversation regardless of gaps. "
                             "Needs a custodian (--me / --me-name).")
    parser.add_argument("--tz", type=str, default=None,
                        help="Timezone the message timestamps are recorded in, "
                             "as an IANA name (America/Chicago) or fixed offset "
                             "(-05:00). The wall-clock time is relabeled to this "
                             "zone without shifting — use it when an exporter "
                             "(e.g. iMazing) stamps local time with a bogus 'Z' "
                             "so messages wrongly read as UTC.")
    parser.add_argument("--title", type=str, default=None,
                        help="Override the document title / page heading.")
    parser.add_argument("--me", type=str, default=None,
                        help="Identify yourself (the device owner) by name, "
                             "phone, or email. Your messages render on the right "
                             "and your participant's display is overridden when "
                             "a name is given. Defaults: the conversation's "
                             "declared custodian, then iMazing auto-detect "
                             "(participant whose display is 'iPhone (N)').")
    parser.add_argument("--me-name", type=str, default=None,
                        help="Display name for the device owner / custodian. "
                             "Renames the resolved custodian — including the "
                             "iMazing 'iPhone (N)' device participant — so it "
                             "renders as a real person. Use this when --me "
                             "can't match the device by an identifier it lacks; "
                             "combine with --me to both pick and name a custodian.")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.output and (len(args.inputs) > 1 or args.out_dir):
        parser.error("--output cannot be combined with multiple inputs or --out-dir")

    try:
        assume_tz = _resolve_tz(args.tz)
    except ValueError as e:
        parser.error(str(e))

    try:
        session_gap = _parse_duration(args.session_gap)
    except ValueError as e:
        parser.error(str(e))

    fmt = "markdown" if args.format == "md" else args.format
    # Extension used when we name the output ourselves (i.e. not via --output,
    # whose extension is honoured as-is). 'both' strips the suffix anyway.
    default_ext = ".md" if fmt == "markdown" else ".html"

    embed = not args.no_embed
    rc = 0
    for src in args.inputs:
        if not src.exists():
            logger.error("Input not found: %s", src)
            rc = 1
            continue
        if args.output:
            out = args.output
        elif args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out = args.out_dir / (src.stem + default_ext)
        else:
            out = src.with_suffix(default_ext)

        try:
            summary = _convert_one(src, out, embed=embed, fmt=fmt or "auto",
                                   split=args.split, title=args.title,
                                   me=args.me, me_name=args.me_name,
                                   assume_tz=assume_tz, session_gap=session_gap,
                                   require_reply=args.require_reply)
        except RsmfParseError as e:
            logger.error("%s: %s", src.name, e)
            rc = 1
            continue
        except Exception:
            logger.exception("Failed to convert %s", src)
            rc = 1
            continue
        dest = out.with_suffix("") if args.split != "none" else out
        logger.info(
            "%s → %s%s (%d conversations, %d messages)",
            src.name, dest, "/" if args.split != "none" else "",
            summary["conversation_count"], summary["message_count"],
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
