"""Tests for the agentic chat loop using a scripted fake model + fake tools."""

from __future__ import annotations

import json
from typing import Any


from case_chat.chat.orchestrator import ChatSession


def test_citations_carry_passage_text_and_kinds() -> None:
    from case_chat.chat.orchestrator import _citations_from

    law = _citations_from("kb_law_search", {"hits": [{
        "binding_class": "binding", "citation": "A.C.A. 9-13-101",
        "doc_type": "statute", "jurisdiction": "ar", "text": "The statute text."}]})
    assert law[0]["kind"] == "law" and law[0]["text"] == "The statute text."
    assert law[0]["binding_class"] == "binding"

    pat = _citations_from("kb_pattern_search", {"hits": [{
        "card_name": "Coercive Control", "framework": "Stark", "text": "card body"}]})
    assert pat[0]["kind"] == "pattern" and pat[0]["card_name"] == "Coercive Control"
    assert pat[0]["text"] == "card body"

    doc = _citations_from("corpus_search", {"hits": [{
        "source_path": "emails/x.eml", "title": "X", "source_type": "email"}]})
    assert doc[0]["kind"] == "document" and doc[0]["source_path"] == "emails/x.eml"

    verses = _citations_from("kb_scripture_lookup", {"verses": [{
        "reference": "James 2:3", "translation": "web", "text": "verse text"}]})
    assert verses[0]["kind"] == "scripture" and verses[0]["text"] == "verse text"


def test_case_tools_emit_resolvable_citations() -> None:
    from case_chat.chat.orchestrator import _annotate_citations, _citations_from

    overview = _citations_from("case_overview", {
        "case": {"case_number": "04DR-25-1847", "jurisdiction": "Benton County, AR"},
        "counts": {"facts": 12},
        "participants": [{"id": "E001", "canonical_name": "Ryan Holcomb", "role": "petitioner"}],
    })
    # The overview record itself is a fallback-eligible Source; the roster rides
    # along so [E001]-style markers resolve, but tagged out of the fallback.
    assert overview[0]["kind"] == "case" and overview[0]["ref_id"] == "case_overview"
    assert "04DR-25-1847" in overview[0]["text"] and "Ryan Holcomb" in overview[0]["text"]
    assert overview[0].get("fallback", True) is True
    roster = [c for c in overview if c["ref_id"] == "E001"]
    assert roster and roster[0]["fallback"] is False

    facts = _citations_from("case_facts_query", {"facts": [
        {"id": "F011", "subject": "Kaylee", "predicate": "is daughter of", "object": "Gerald"},
    ]})
    assert facts[0]["kind"] == "case" and facts[0]["ref_id"] == "F011"
    assert facts[0]["text"].startswith("Fact: F011")

    # the bare markers the model writes must resolve against the pool — including
    # participant ids like [E001] read off the case_overview roster.
    pool = overview + facts
    ans = "Ryan [E001] petitioned [case_overview]. Kaylee is Gerald's daughter [F011]."
    out, cites = _annotate_citations(ans, pool)
    assert out.count("[[CITE:") == 3
    assert [c["ref_id"] for c in cites] == ["E001", "case_overview", "F011"]


def test_annotate_citations_robust_matching() -> None:
    from case_chat.chat.orchestrator import _annotate_citations

    pool = [
        {"kind": "document", "source_path": "emails/pastor-whitfield-to-ryan.eml",
         "title": "A Word from Your Pastor", "source_type": "email"},
        {"kind": "pattern", "card_id": "simon-guilt-tripping",
         "card_name": "Guilt-tripping and induced shame as control"},
        {"kind": "document", "source_path": "witness-statements/megan-trask-statement.txt",
         "title": "Megan Trask Statement"},
        {"kind": "document", "source_path": "court-documents/affidavit-peggy-prater.txt",
         "title": "Affidavit Peggy Prater"},
    ]
    ans = ("Pastor cited it [Email: Pastor Whitfield to Ryan]. This is [simon-guilt-tripping]. "
           "Both noted [Megan Trask Statement; Affidavit Peggy Prater]. At [2026-01-28T06:20:12-06:00].")
    out, cites = _annotate_citations(ans, pool)

    assert "[[CITE:0]]" in out                 # email by descriptive label → filename substring
    assert "[[CITE:1]]" in out                 # pattern by card_id
    assert "[[CITE:2]][[CITE:3]]" in out       # two sources in one bracket → two tokens
    assert "[2026-01-28T06:20:12-06:00]" in out  # timestamp left untouched
    assert len(cites) == 4
    assert cites[0]["source_path"].endswith("pastor-whitfield-to-ryan.eml")
    assert cites[1]["card_id"] == "simon-guilt-tripping"
    assert cites[2]["title"] == "Megan Trask Statement"
    assert cites[3]["title"] == "Affidavit Peggy Prater"


def test_annotate_no_pool_or_markers_is_noop() -> None:
    from case_chat.chat.orchestrator import _annotate_citations

    assert _annotate_citations("hello", []) == ("hello", [])
    out, cites = _annotate_citations("no markers here", [{"kind": "document", "source_path": "x.md"}])
    assert out == "no markers here" and cites == []


class FakeModel:
    """Returns queued assistant messages, one per chat() call."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, messages, *, tools=None, temperature=0.3, tool_choice="auto"):
        self.calls.append(messages)
        return self._responses.pop(0)


class FakeTools:
    def __init__(self, results: dict[str, str]) -> None:
        self._results = results
        self.invoked: list[tuple[str, dict]] = []

    def openai_tools(self):
        return [{"type": "function", "function": {"name": "corpus_search", "parameters": {}}}]

    async def call(self, name: str, arguments: dict) -> str:
        self.invoked.append((name, arguments))
        return self._results[name]


def _tool_call(name: str, args: dict) -> dict:
    return {"id": "tc1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


async def test_loop_executes_tool_then_answers() -> None:
    corpus_result = json.dumps({"hits": [
        {"source_type": "apple_note", "source_path": "apple-notes/notes/x.md", "title": "X"}
    ]})
    model = FakeModel([
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("corpus_search", {"query": "bruise"})]},
        {"role": "assistant", "content": "The school nurse documented a bruise."},
    ])
    tools = FakeTools({"corpus_search": corpus_result})
    session = ChatSession(model, tools)

    out = await session.ask("What did the school nurse find?")

    assert out["answer"] == "The school nurse documented a bruise."
    assert tools.invoked == [("corpus_search", {"query": "bruise"})]
    assert out["tool_calls"][0]["name"] == "corpus_search"
    assert out["citations"][0]["source_path"] == "apple-notes/notes/x.md"
    # History: user, assistant(tool_calls), tool, assistant(answer)
    roles = [m["role"] for m in session.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert session.history[2]["tool_call_id"] == "tc1"


async def test_citations_resolve_from_earlier_turn() -> None:
    # Turn 1 retrieves a source; turn 2 answers from context (no tool) but
    # references it inline → it must still resolve from the conversation pool.
    res = json.dumps({"hits": [{"source_type": "witness_statement",
        "source_path": "witness-statements/megan-trask-statement.txt", "title": "Megan Trask Statement"}]})
    model = FakeModel([
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("corpus_search", {"query": "megan"})]},
        {"role": "assistant", "content": "Megan described control [Megan Trask Statement]."},
        {"role": "assistant", "content": "As noted, it was systemic [Megan Trask Statement]."},
    ])
    session = ChatSession(model, FakeTools({"corpus_search": res}))
    out1 = await session.ask("turn 1")
    assert any("megan-trask-statement" in (c.get("source_path") or "") for c in out1["citations"])
    out2 = await session.ask("turn 2")
    assert out2["tool_calls"] == []  # no tool called this turn
    assert any("megan-trask-statement" in (c.get("source_path") or "") for c in out2["citations"]), \
        "prior-turn source must resolve from the conversation pool"


async def test_citations_deduped_across_chunks() -> None:
    dup = json.dumps({"hits": [
        {"source_type": "messages", "source_path": "messages/x.rsmf", "title": "T"},
        {"source_type": "messages", "source_path": "messages/x.rsmf", "title": "T"},
    ]})
    model = FakeModel([
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("corpus_search", {"query": "x"})]},
        {"role": "assistant", "content": "done"},
    ])
    out = await ChatSession(model, FakeTools({"corpus_search": dup})).ask("q")
    assert len(out["citations"]) == 1  # same source_path collapsed


async def test_case_overview_surfaces_source_without_inline_markers() -> None:
    # Real-world case: the model calls only case_overview and writes a prose
    # answer with NO bracketed markers. The fallback must still surface the
    # overview as a Source (a single clean record, not the whole roster).
    overview = json.dumps({
        "case": {"case_number": "04DR-25-1847", "corpus_name": "Holcomb Family Guardianship"},
        "counts": {"facts": 12},
        "participants": [
            {"id": "E001", "canonical_name": "Ryan Holcomb", "role": "petitioner"},
            {"id": "E002", "canonical_name": "Gerald Holcomb", "role": "respondent"},
        ],
    })
    model = FakeModel([
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("case_overview", {})]},
        {"role": "assistant", "content": "This is a guardianship matter with no brackets at all."},
    ])
    out = await ChatSession(model, FakeTools({"case_overview": overview})).ask("get me up to speed")
    assert len(out["citations"]) == 1
    assert out["citations"][0]["ref_id"] == "case_overview"


async def test_direct_answer_without_tools() -> None:
    model = FakeModel([{"role": "assistant", "content": "Hello, how can I help?"}])
    session = ChatSession(model, FakeTools({}))
    out = await session.ask("hi")
    assert out["answer"] == "Hello, how can I help?"
    assert out["tool_calls"] == []


async def test_max_iterations_guard() -> None:
    # Model always asks for a tool → loop must terminate at the limit.
    looping = [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("corpus_search", {})]}
        for _ in range(10)
    ]
    model = FakeModel(looping)
    tools = FakeTools({"corpus_search": json.dumps({"hits": []})})
    session = ChatSession(model, tools, max_iterations=3)
    out = await session.ask("loop")
    assert "tool-call limit" in out["answer"]
    assert len(tools.invoked) == 3


async def test_thinking_event_streamed_per_turn() -> None:
    model = FakeModel([
        {"role": "assistant", "content": "", "reasoning_content": "let me check the docs",
         "tool_calls": [_tool_call("corpus_search", {"query": "x"})]},
        {"role": "assistant", "content": "Done.", "reasoning_content": "now I can answer"},
    ])
    tools = FakeTools({"corpus_search": json.dumps({"hits": []})})
    events = [e async for e in ChatSession(model, tools).stream("q")]
    thinking = [e for e in events if e["type"] == "thinking"]
    assert [t["text"] for t in thinking] == ["let me check the docs", "now I can answer"]
    # thinking precedes the tool call it motivated
    assert events.index(thinking[0]) < next(i for i, e in enumerate(events) if e["type"] == "tool_call")


async def test_thinking_field_variants(monkeypatch) -> None:
    # Ollama uses `reasoning`; Ollama-native uses `thinking`; both must surface.
    for field in ("reasoning", "thinking"):
        model = FakeModel([{"role": "assistant", "content": "A.", field: "deliberating"}])
        events = [e async for e in ChatSession(model, FakeTools({})).stream("q")]
        assert any(e["type"] == "thinking" and e["text"] == "deliberating" for e in events)


async def test_reasoning_content_surfaced_not_refed_back() -> None:
    model = FakeModel([
        {"role": "assistant", "content": "Answer.", "reasoning_content": "secret chain of thought"},
    ])
    session = ChatSession(model, FakeTools({}))
    out = await session.ask("q")
    assert out["reasoning"] == "secret chain of thought"
    # Reasoning must NOT be stored in history (not fed back as context).
    assert all("reasoning_content" not in m for m in session.history)
