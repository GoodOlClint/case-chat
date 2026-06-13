"""Tests for the fake-case dataset build + query layer.

Builds from the real synthetic ground-truth into a temp SQLite and asserts
known facts. If the corpus isn't present the tests skip (CI without corpus).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from case_chat.config import settings
from case_chat.casedata.dataset import build_dataset
from case_chat.casedata.queries import CaseDataset

CORPUS = Path(settings.synthetic_corpus_path)
pytestmark = pytest.mark.skipif(
    not (CORPUS / "timeline.json").exists(), reason="synthetic corpus not available"
)


@pytest.fixture(scope="module")
def ds(tmp_path_factory) -> CaseDataset:
    out = tmp_path_factory.mktemp("casedata") / "fake.sqlite3"
    build_dataset(corpus_root=CORPUS, sqlite_path=out)
    return CaseDataset(out)


def test_petition_filed_date(ds: CaseDataset) -> None:
    hits = ds.timeline_query(text="files guardianship petition", limit=5)
    assert any(h["date"] == "2025-09-07" for h in hits)


def test_entity_lookup_by_alias(ds: CaseDataset) -> None:
    # "Kay" is an alias of Kaylee Ann Holcomb.
    hits = ds.entity_lookup("Kay")
    assert hits and hits[0]["id"] == "E007"
    assert hits[0]["role"] == "minor_child"
    assert "Kaylee" in hits[0]["aliases"]


def test_entity_lookup_partial_name_token_match(ds: CaseDataset) -> None:
    # 'Kaylee Holcomb' must resolve to 'Kaylee Ann Holcomb' (token-AND fallback).
    hits = ds.entity_lookup("Kaylee Holcomb")
    assert hits and hits[0]["id"] == "E007"


def test_timeline_entity_and_date_filter(ds: CaseDataset) -> None:
    hits = ds.timeline_query(entity="Gerald", date_from="2025-01-01", date_to="2025-12-31", limit=50)
    assert hits
    assert all("2025-01-01" <= h["date"] <= "2025-12-31" for h in hits)


def test_case_meta_case_number(ds: CaseDataset) -> None:
    assert ds.case_meta().get("case_number") == "04DR-25-1847"


def test_unknown_entity_returns_empty(ds: CaseDataset) -> None:
    assert ds.entity_lookup("Nonexistent Person XYZ") == []


def test_normalize_docs_maps_messages_xml_to_rsmf() -> None:
    from case_chat.casedata.dataset import _normalize_docs

    assert _normalize_docs(["messages/ryan-megan-thread.xml", "emails/x.eml"]) == [
        "messages/ryan-megan-thread.rsmf", "emails/x.eml"
    ]
    assert _normalize_docs(None) is None
    assert _normalize_docs(["messages/combined.csv"]) == ["messages/combined.csv"]


def test_message_refs_normalized_in_dataset(ds: CaseDataset) -> None:
    bad, rsmf_seen = [], False
    rows = ds.timeline_query(limit=500) + ds.facts_query(limit=500)
    docs_rows = ds.flags_query(limit=500) + ds.observations_query(limit=500)
    for r in rows:
        for d in r.get("source_documents", []):
            if d.startswith("messages/") and d.endswith(".xml"):
                bad.append(d)
            rsmf_seen = rsmf_seen or d.endswith(".rsmf")
    for r in docs_rows:
        for d in r.get("documents", []):
            if d.startswith("messages/") and d.endswith(".xml"):
                bad.append(d)
            rsmf_seen = rsmf_seen or d.endswith(".rsmf")
    assert not bad, f"un-normalized .xml message refs: {bad[:5]}"
    assert rsmf_seen, "expected at least one messages/*.rsmf reference"


def test_case_facts_view_has_all_sections(ds: CaseDataset) -> None:
    v = ds.case_facts_view()
    assert v["case"]["case_number"] == "04DR-25-1847"
    assert len(v["participants"]) == 29
    assert len(v["timeline"]) == 105
    assert len(v["facts"]) == 207
    assert len(v["flags"]) == 120
    assert len(v["observations"]) == 36


def test_case_overview_lists_parties(ds: CaseDataset) -> None:
    ov = ds.case_overview()
    assert ov["case"]["case_number"] == "04DR-25-1847"
    ids = {p["id"] for p in ov["participants"]}
    assert {"E001", "E002"} <= ids  # Gerald + Ryan present
    assert ov["counts"]["entities"] == 29


def test_list_participants_by_role(ds: CaseDataset) -> None:
    respondents = ds.list_participants(role="respondent")
    assert any(p["id"] == "E001" for p in respondents)  # Gerald = respondent
    children = ds.list_participants(role="child")
    assert {p["id"] for p in children} >= {"E007"}  # minor_child


def test_relationships_resolved_to_names(ds: CaseDataset) -> None:
    ryan = ds.entity_lookup("Ryan Matthew Holcomb")[0]
    sons = [r for r in ryan["relationships"] if r["relationship"] == "son_of"]
    assert any(r["name"] == "Gerald Wayne Holcomb" for r in sons)


def test_reverse_relationships_connect_parties(ds: CaseDataset) -> None:
    # Gerald's own relationships may be empty, but Ryan references him as son_of —
    # referenced_by must surface that so 'Gerald's relation to Ryan' is answerable.
    gerald = ds.entity_lookup("Gerald Wayne Holcomb")[0]
    refs = gerald["referenced_by"]
    assert any(r["name"] == "Ryan Matthew Holcomb" and r["relationship"] == "son_of" for r in refs)
