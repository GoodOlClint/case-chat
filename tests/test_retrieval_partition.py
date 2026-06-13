"""Unit tests for authority partitioning and filter building (no Qdrant)."""

from __future__ import annotations

from qdrant_client.http import models as qm

from case_chat.retrieval.corpus import Hit, build_filter, partition_by_binding


def _hit(bj) -> Hit:
    payload = {} if bj is None else {"binding_jurisdictions": bj}
    return Hit(score=1.0, payload=payload)


def test_ar_in_binding_jurisdictions_is_binding() -> None:
    part = partition_by_binding([_hit(["ar"])], active_jurisdiction="ar")
    assert len(part.binding) == 1 and not part.persuasive and not part.non_authority


def test_federal_binds_state_courts() -> None:
    part = partition_by_binding([_hit(["federal"])], active_jurisdiction="ar")
    assert len(part.binding) == 1


def test_other_state_only_is_persuasive() -> None:
    part = partition_by_binding([_hit(["mo"])], active_jurisdiction="ar")
    assert len(part.persuasive) == 1 and not part.binding


def test_empty_or_missing_is_non_authority() -> None:
    part = partition_by_binding([_hit([]), _hit(None)], active_jurisdiction="ar")
    assert len(part.non_authority) == 2


def test_active_jurisdiction_switch_reclassifies() -> None:
    # Same hit is binding for MO, persuasive for AR.
    hits = [_hit(["mo"])]
    assert len(partition_by_binding(hits, "mo").binding) == 1
    assert len(partition_by_binding(hits, "ar").persuasive) == 1


def test_case_insensitive_jurisdiction_match() -> None:
    part = partition_by_binding([_hit(["AR"])], active_jurisdiction="ar")
    assert len(part.binding) == 1


def test_build_filter_none_for_empty() -> None:
    assert build_filter(None) is None
    assert build_filter({}) is None
    assert build_filter({"jurisdiction": None}) is None  # skipped → no conditions


def test_build_filter_scalar_and_list() -> None:
    f = build_filter({"jurisdiction": "federal", "doc_type": ["statute", "opinion"]})
    assert isinstance(f, qm.Filter)
    keys = {c.key for c in f.must}
    assert keys == {"jurisdiction", "doc_type"}
    by_key = {c.key: c for c in f.must}
    assert isinstance(by_key["jurisdiction"].match, qm.MatchValue)
    assert isinstance(by_key["doc_type"].match, qm.MatchAny)
