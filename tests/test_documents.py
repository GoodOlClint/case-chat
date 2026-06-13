"""Security + behavior tests for the test-corpus source-document viewer."""

from __future__ import annotations

from pathlib import Path

import pytest

from case_chat.config import settings
from case_chat.web import documents

CORPUS = Path(settings.synthetic_corpus_path)
pytestmark = pytest.mark.skipif(
    not (CORPUS / "emails").exists(), reason="synthetic corpus not available"
)


@pytest.fixture(autouse=True)
def _clear_cache():
    documents._allowed_files.cache_clear()
    yield


def test_valid_document_resolves_and_reads() -> None:
    doc = documents.read_document("emails/aldridge-hearing-notice.eml")
    assert doc is not None
    assert doc["name"] == "aldridge-hearing-notice.eml"
    assert "From:" in doc["text"]


def test_valid_apple_note() -> None:
    assert documents.resolve_document("apple-notes/notes/2025-10-16_school-nurse-report.md") is not None


def test_visitation_csv_allowed() -> None:
    assert documents.resolve_document("structured-data/supervised-visitation-log.csv") is not None


def test_ground_truth_json_rejected() -> None:
    # Evaluation artifacts must never be viewable.
    for gt in ("entities.json", "timeline.json", "manifest.json", "master_facts.json"):
        assert documents.resolve_document(gt) is None


def test_rsmf_is_canonical_message_format() -> None:
    # RSMF is the canonical message format (viewable); the .xml sibling is not.
    assert documents.resolve_document("messages/ryan-megan-thread.rsmf") is not None
    assert documents.resolve_document("messages/ryan-megan-thread.xml") is None


def test_path_traversal_rejected() -> None:
    for evil in ("../../etc/passwd", "emails/../../../../etc/passwd",
                 "/etc/passwd", "emails/../entities.json"):
        assert documents.resolve_document(evil) is None


def test_nonexistent_rejected() -> None:
    assert documents.resolve_document("emails/does-not-exist.eml") is None
    assert documents.resolve_document("") is None


def test_pdf_not_indexed_not_viewable() -> None:
    # PDFs are out of scope for the index → not viewable.
    assert documents.resolve_document("court-documents/petition-for-guardianship.pdf") is None


def test_list_documents_only_viewable_types() -> None:
    docs = documents.list_documents()
    assert docs, "expected some documents"
    types = {d["source_type"] for d in docs}
    assert {"email", "apple_note", "court_document", "messages"} <= types
    paths = {d["source_path"] for d in docs}
    # No ground-truth, no xml dupes, no PDFs in the browseable list.
    assert "entities.json" not in paths
    assert not any(p.endswith((".xml", ".pdf")) for p in paths)
    assert any(p.endswith(".rsmf") for p in paths)  # messages present, as RSMF
    # Every listed doc must actually resolve (consistency with the viewer).
    assert all(documents.resolve_document(d["source_path"]) is not None for d in docs)


def test_render_document_formats() -> None:
    msg = documents.render_document("messages/ryan-megan-thread.rsmf")
    assert msg["format"] == "messages" and "<html" in msg["html"].lower()

    note = documents.render_document("apple-notes/notes/2025-10-16_school-nurse-report.md")
    assert note["format"] == "markdown" and "text" in note

    eml = documents.render_document("emails/aldridge-hearing-notice.eml")
    assert eml["format"] == "text" and "From:" in eml["text"]
