"""Serve raw TEST-CORPUS source documents for the UI's "view source" panel.

Scope is deliberately narrow ([DESIGN]): only the synthetic raw documents that
the corpus indexer actually ingests are viewable. Domain-knowledge text and the
ground-truth JSON are NOT served here. Viewability is defined as membership in
the exact set of files matched by the indexer's source globs — which makes path
traversal impossible (a candidate must resolve to one of those known files) and
excludes ground-truth/meta files (they don't match any glob).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from case_chat.config import settings
from case_chat.synthetic.loaders import SOURCE_GLOBS

# The viewer allows exactly what the synthetic indexer ingests: the per-type
# globs plus the visitation CSV. (RSMF dupes, ground-truth JSON, PDFs, .m4a,
# and all domain-knowledge text are excluded by construction.)
ALLOWED_GLOBS: tuple[str, ...] = (*SOURCE_GLOBS.values(), "structured-data/*.csv")


@lru_cache(maxsize=1)
def _allowed_files(root_str: str) -> frozenset[Path]:
    root = Path(root_str)
    files: set[Path] = set()
    for glob in ALLOWED_GLOBS:
        files.update(p.resolve() for p in root.glob(glob))
    return frozenset(files)


def resolve_document(source_path: str) -> Path | None:
    """Resolve a source_path to a viewable file, or None if not permitted.

    A path is viewable only if it resolves to one of the exact files the indexer
    ingested — so `../`, absolute paths, ground-truth JSON, and domain knowledge
    all return None.
    """
    if not source_path or "\x00" in source_path:
        return None
    root = Path(settings.synthetic_corpus_path).resolve()
    candidate = (root / source_path).resolve()
    if candidate in _allowed_files(str(root)) and candidate.is_file():
        return candidate
    return None


def list_documents() -> list[dict[str, str]]:
    """List every viewable test-corpus document, grouped-friendly by source_type."""
    root = Path(settings.synthetic_corpus_path).resolve()
    type_globs = [*SOURCE_GLOBS.items(), ("visitation_log", "structured-data/*.csv")]
    out: list[dict[str, str]] = []
    for source_type, glob in type_globs:
        for p in sorted(root.glob(glob)):
            out.append({
                "source_type": source_type,
                "source_path": p.relative_to(root).as_posix(),
                "name": p.name,
            })
    return out


def read_document(source_path: str) -> dict[str, str] | None:
    """Return {source_path, name, text} (raw) for a viewable document, else None."""
    path = resolve_document(source_path)
    if path is None:
        return None
    return {
        "source_path": source_path,
        "name": path.name,
        "text": path.read_text(encoding="utf-8", errors="replace"),
    }


def render_document(source_path: str) -> dict[str, str] | None:
    """Return a viewable document tagged with a render `format` for the UI:

    - .rsmf  → {format: 'messages', html}  (rendered chat via vendored rsmf_viewer)
    - .md    → {format: 'markdown', text}  (UI renders markdown)
    - else   → {format: 'text', text}      (raw monospace)
    """
    path = resolve_document(source_path)
    if path is None:
        return None
    suffix = path.suffix.lower()
    base = {"source_path": source_path, "name": path.name}

    if suffix == ".rsmf":
        from case_chat.vendor import rsmf_viewer as rv

        manifest, zf = rv.parse_rsmf(path)
        try:
            convs, summary = rv.build_conversations(manifest, zf, embed_assets=True, asset_dir=None)
        finally:
            zf.close()
        html = rv.render_html(convs, summary, source_name=path.name)
        return {**base, "format": "messages", "html": html}

    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".md":
        # Strip YAML frontmatter so the note body renders cleanly as markdown.
        m = re.match(r"^---\n.*?\n---\n?(.*)$", text, re.DOTALL)
        return {**base, "format": "markdown", "text": (m.group(1) if m else text)}
    return {**base, "format": "text", "text": text}
