"""Tests for the persistent conversation store (saved chat history)."""

from __future__ import annotations

import pytest

from case_chat.web.conversations import ConversationStore


@pytest.fixture()
def store(tmp_path) -> ConversationStore:
    return ConversationStore(tmp_path / "conv.sqlite3")


def test_create_add_and_get_roundtrip(store: ConversationStore) -> None:
    cid = store.new_id()
    store.create(cid, "alice")
    store.add_turn(cid, "alice", "user", "Who is Kaylee?")
    store.add_turn(cid, "alice", "assistant", "Kaylee Ann Holcomb.",
                   citations=[{"kind": "document", "source_path": "x.md"}], thinking="hmm")
    conv = store.get(cid, "alice")
    assert conv["title"] == "Who is Kaylee?"  # title from first user turn
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["messages"][1]["citations"][0]["source_path"] == "x.md"
    assert conv["messages"][1]["thinking"] == "hmm"


def test_list_orders_newest_first_and_titled_only(store: ConversationStore) -> None:
    a, b, empty = store.new_id(), store.new_id(), store.new_id()
    store.create(empty, "alice")  # no turns → no title → excluded from list
    store.add_turn(a, "alice", "user", "first question")
    store.add_turn(b, "alice", "user", "second question")
    ids = [c["id"] for c in store.list("alice")]
    assert ids == [b, a]  # newest first
    assert empty not in ids


def test_isolation_between_subjects(store: ConversationStore) -> None:
    cid = store.new_id()
    store.add_turn(cid, "alice", "user", "secret")
    assert store.owns(cid, "alice") and not store.owns(cid, "bob")
    assert store.get(cid, "bob") is None
    assert store.list("bob") == []


def test_delete(store: ConversationStore) -> None:
    cid = store.new_id()
    store.add_turn(cid, "alice", "user", "q")
    assert store.delete(cid, "bob") is False  # not owner
    assert store.delete(cid, "alice") is True
    assert store.get(cid, "alice") is None
