"""Agentic chat orchestrator.

Runs the tool-calling loop: send history + tools to vLLM, execute any tool
calls via the MCP client, feed results back, repeat until the model produces a
grounded answer. Keeps multi-turn history per session and extracts a normalized
citation list from the tools it ran (source documents; for law, binding vs
persuasive authority).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from case_chat.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful legal-research assistant for a fictional Arkansas \
family-law guardianship matter (the Holcomb case) plus a domain-knowledge corpus of \
Arkansas/federal law, behavioral-science patterns, professional standards, and scripture.

Ground every factual claim in the tools — do not answer from memory or invent facts, \
citations, dates, or quotes. Use:
- kb_law_search / kb_law_lookup for statutes, case-law, constitutions. Respect the \
binding vs persuasive vs non-authority labels: binding authority controls in Arkansas; \
persuasive only informs. Never present persuasive or out-of-state authority as binding.
- kb_pattern_search (+ kb_behavioral_source_search), kb_standards_search for behavioral \
and professional-standards questions. REQUIRED whenever you name or characterize a \
behavior pattern — abuse, coercive control, manipulation, gaslighting, parental \
alienation, isolation, intimidation, love-bombing, high-control religion, etc. Do NOT \
label behavior from your own knowledge: pull the relevant text with corpus_search, then \
call kb_pattern_search with the observed conduct to match it to the framework's pattern \
cards, and cite those cards (by card_name) alongside the source quotes. The framework — \
not your judgment — is the authority for what a pattern is.
- kb_scripture_search (by theme) or kb_scripture_lookup (by exact reference).
- corpus_search for the raw case documents (notes, emails, messages, court filings, \
transcripts, the visitation log).
- The structured case record:
  * case_overview — the case number, jurisdiction, and the FULL participant roster.
    Call this for ANY question about the case as a whole or its people: 'who are the
    participants/parties', 'what is this case', 'who are the respondents'. Do NOT ask
    the user to supply names first — look them up.
  * case_participants — the roster filtered by role (respondent, petitioner, minor,
    child, attorney, witness, judge) or type.
  * case_entity_lookup — one person by name/alias → role, DOB, aliases, and
    relationships RESOLVED TO NAMES (both their stated relationships and who refers to
    them), so you can answer 'how is X related to Y'.
  * case_timeline_query / case_facts_query / case_flags / case_observations.

Use the conversation history for context and follow-up questions. When you make a
factual claim, ground it in tool results and cite the sources — call the tools that
support your answer so it carries fresh, clickable Sources (don't just assert it from
an earlier turn). History is for continuity and resolving references ("that case",
"his messages"); tools are for grounding.

Be proactive: if a tool can answer, call it — never refuse for lack of a name when
case_overview or case_participants would list them. When the user names an exact
citation or verse, use the *_lookup tools, not search. Any time your answer would
*characterize* behavior, you must have called kb_pattern_search first and ground the
characterization in its cards — surfacing quotes without grounding the labels is
incomplete. Cite the documents/authorities you relied on. If the tools genuinely don't
support a claim, say so.

Attribute inline: after each specific claim, quote, or characterization, add a bracketed
source marker using the source's EXACT title, filename, citation, or pattern-card name as
shown in the tool results — e.g. [Text thread: Gerald & Ryan], [A.C.A. § 9-13-101],
[Guilt-tripping and induced shame as control]. Never use a bare number like [1], and do
not write your own numbered Sources list — the system renders one. Put one source per
bracket, or separate several with a semicolon: [Megan Trask Statement; Affidavit Peggy Prater]."""


class _Model(Protocol):
    async def chat(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = ...,
        temperature: float = ..., tool_choice: str = ...,
    ) -> dict[str, Any]: ...


class _Tools(Protocol):
    def openai_tools(self) -> list[dict[str, Any]]: ...
    async def call(self, name: str, arguments: dict[str, Any]) -> str: ...


def _parse_result(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _reasoning(msg: dict[str, Any]) -> str | None:
    """The model's reasoning, across provider field names.

    vLLM (gemma4 reasoning-parser) → ``reasoning_content``; Ollama's OpenAI
    endpoint → ``reasoning``; Ollama native → ``thinking``.
    """
    return msg.get("reasoning_content") or msg.get("reasoning") or msg.get("thinking")


_CITATION_TEXT_CAP = 6000


def _text(hit: dict[str, Any]) -> str:
    return (hit.get("text") or hit.get("text_preview") or "")[:_CITATION_TEXT_CAP]


def _citations_from(name: str, result: Any) -> list[dict[str, Any]]:
    """Normalize a tool result into citation descriptors for the UI.

    Domain-knowledge citations carry the retrieved passage ``text`` so the UI can
    show the material on click without a second fetch. Document citations carry
    ``source_path`` and open the raw file.
    """
    if not isinstance(result, dict):
        return []
    out: list[dict[str, Any]] = []
    for hit in result.get("hits", []):
        if name == "kb_law_search":
            out.append({"kind": "law", "binding_class": hit.get("binding_class"),
                        "citation": hit.get("citation"), "doc_type": hit.get("doc_type"),
                        "jurisdiction": hit.get("jurisdiction"), "source_url": hit.get("source_url"),
                        "text": _text(hit)})
        elif name == "corpus_search":
            out.append({"kind": "document", "source_path": hit.get("source_path"),
                        "title": hit.get("title"), "source_type": hit.get("source_type")})
        elif name == "kb_scripture_search":
            out.append({"kind": "scripture", "reference": hit.get("reference"),
                        "translation": hit.get("translation"), "text": _text(hit)})
        elif name == "kb_pattern_search":
            out.append({"kind": "pattern", "card_id": hit.get("card_id"),
                        "card_name": hit.get("card_name"),
                        "framework": hit.get("framework"), "text": _text(hit)})
        elif name in ("kb_behavioral_source_search", "kb_standards_search"):
            out.append({"kind": "reference", "title": hit.get("title") or hit.get("name"),
                        "citation": hit.get("citation"), "text": _text(hit)})
    for v in result.get("verses", []):
        out.append({"kind": "scripture", "reference": v.get("reference"),
                    "translation": v.get("translation"), "text": _text(v)})
    for p in result.get("passages", []):
        out.append({"kind": "law", "citation": p.get("citation"), "doc_type": p.get("doc_type"),
                    "jurisdiction": p.get("jurisdiction"), "text": _text(p)})
    return out


def _citation_key(c: dict[str, Any]) -> tuple:
    """Identity for de-duplicating citations (same source cited by many chunks)."""
    kind = c.get("kind")
    if kind == "document":
        return ("document", c.get("source_path"), None)
    if kind == "law":
        return ("law", c.get("citation"), None)
    if kind == "scripture":
        return ("scripture", c.get("reference"), c.get("translation"))
    if kind == "pattern":
        return ("pattern", c.get("card_id") or c.get("card_name"), None)
    return (kind, c.get("title") or c.get("citation"), None)


# -- inline citation-marker resolution -------------------------------------
# The model writes bracketed markers in many forms — [messages/x.rsmf],
# [Email: Pastor Whitfield to Ryan], [pattern: Foo], [simon-guilt-tripping],
# [Megan Trask Statement; Affidavit Peggy Prater]. We resolve each against the
# CONVERSATION-WIDE citation pool and rewrite it to a canonical [[CITE:n]] token,
# so the UI just renders tokens (no matching logic = no client/server drift).
_REF_PREFIX = re.compile(
    r"^(patterns?|sources?|docs?|documents?|citations?|cite|refs?|references?|email|e-mail|"
    r"transcript|notes?|apple note|visitation log|visitation|court document|court doc|court filing|"
    r"witness statement|statement|messages?|text thread|thread|scripture|verse|bible|exhibit)\s*:\s*",
    re.IGNORECASE,
)
_BRACKET = re.compile(r"\[([^\]\n]{1,120})\]")
_CITE_TOKEN = "[[CITE:{}]]"


def _norm_ref(s: str | None) -> str:
    s = _REF_PREFIX.sub("", (s or "").lower())
    s = re.sub(r"\.\w{1,5}$", "", s)  # drop a trailing file extension
    s = s.replace("-", " ").replace("_", " ").replace("/", " ")
    return re.sub(r"\s+", " ", s).strip()


def _cite_labels(c: dict[str, Any]) -> list[str]:
    """All strings a marker might use to name this citation."""
    kind = c.get("kind")
    out: list[str] = []
    if kind == "document":
        out += [c.get("source_path"), c.get("title")]
    elif kind == "law":
        out += [c.get("citation"), c.get("title"), c.get("case_name")]
    elif kind == "scripture":
        out += [c.get("reference")]
    elif kind == "pattern":
        out += [c.get("card_name"), c.get("card_id")]
    else:
        out += [c.get("title"), c.get("citation"), c.get("name")]
    return [x for x in out if x]


class _CiteResolver:
    """Indexes a citation pool by every label form and resolves a marker to it."""

    def __init__(self, pool: list[dict[str, Any]]) -> None:
        self._exact: dict[str, int] = {}
        self._fuzzy: list[tuple[str, int]] = []
        for i, c in enumerate(pool):
            for label in _cite_labels(c):
                key = _norm_ref(label)
                if key:
                    self._exact.setdefault(key, i)
                    self._fuzzy.append((key, i))

    def find(self, part: str) -> int | None:
        key = _norm_ref(part)
        if not key:
            return None
        if key in self._exact:
            return self._exact[key]
        if len(key) >= 10:  # substring only for distinctive labels
            for k, idx in self._fuzzy:
                if len(k) >= 6 and (key in k or k in key):
                    return idx
        return None


def _annotate_citations(answer: str, pool: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Rewrite inline markers to [[CITE:n]] tokens; return (answer, ordered cites)."""
    if not answer or not pool:
        return answer, []
    resolver = _CiteResolver(pool)
    ordered: list[int] = []
    local: dict[int, int] = {}

    def repl(m: re.Match) -> str:
        tokens = []
        for part in re.split(r"\s*;\s*", m.group(1)):  # multiple sources per bracket
            idx = resolver.find(part)
            if idx is None:
                continue
            if idx not in local:
                local[idx] = len(ordered)
                ordered.append(idx)
            tokens.append(_CITE_TOKEN.format(local[idx]))
        return "".join(tokens) if tokens else m.group(0)

    rewritten = _BRACKET.sub(repl, answer)
    return rewritten, [pool[i] for i in ordered]


class ChatSession:
    """One multi-turn conversation. Not thread-safe; one per user session."""

    def __init__(
        self,
        model: _Model,
        tools: _Tools,
        *,
        system: str = SYSTEM_PROMPT,
        max_iterations: int | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._system = {"role": "system", "content": system}
        self._history: list[dict[str, Any]] = []
        self._max_iterations = max_iterations or settings.vllm_max_tool_iterations
        # Conversation-wide citation pool, so an answer that leans on an earlier
        # turn's tool results still resolves its inline markers and shows Sources.
        self._cite_pool: list[dict[str, Any]] = []
        self._cite_pool_keys: set[tuple] = set()

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._history

    def _pool_add(self, cites: list[dict[str, Any]]) -> None:
        for c in cites or []:
            key = _citation_key(c)
            if key not in self._cite_pool_keys:
                self._cite_pool_keys.add(key)
                self._cite_pool.append(c)

    def _annotate(self, answer: str, this_turn: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Rewrite inline markers to [[CITE:n]] tokens and pick the Sources: the
        pool citations referenced inline (so prior turns count), falling back to
        this turn's tool citations when the answer named nothing recognizable."""
        rewritten, cited = _annotate_citations(answer, self._cite_pool)
        if cited:
            return rewritten, cited
        return answer, this_turn

    def seed_history(self, turns: list[tuple[str, str]]) -> None:
        """Restore conversational context from saved (role, content) turns.

        Only user/assistant text is replayed (tool internals are not needed for
        continuity), so a reloaded conversation keeps its running context.
        """
        self._history = [
            {"role": role, "content": content}
            for role, content in turns
            if role in ("user", "assistant") and content
        ]

    def seed_pool(self, citation_lists: list[list[dict[str, Any]] | None]) -> None:
        """Restore the citation pool from a reloaded conversation's saved turns."""
        for cites in citation_lists:
            self._pool_add(cites or [])

    async def stream(self, user_text: str):
        """Run the agentic loop, yielding live events:

        {type: 'tool_call', name, arguments}
        {type: 'tool_result', name, result}
        {type: 'answer', answer, reasoning, tool_calls, citations}   (terminal)
        """
        self._history.append({"role": "user", "content": user_text})
        tool_calls_made: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        seen_cites: set[tuple] = set()

        for _ in range(self._max_iterations):
            msg = await self._model.chat(
                [self._system, *self._history], tools=self._tools.openai_tools()
            )
            # Surface the model's reasoning (gemma4 reasoning-parser) for display,
            # on every turn (before tool calls and before the final answer). It is
            # never written back into history — not fed to the model as context.
            reasoning = _reasoning(msg)
            if reasoning:
                yield {"type": "thinking", "text": reasoning}
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                answer = msg.get("content") or ""
                self._history.append({"role": "assistant", "content": answer})  # store raw
                shown, cited = self._annotate(answer, citations)
                yield {"type": "answer", "answer": shown, "reasoning": reasoning,
                       "tool_calls": tool_calls_made, "citations": cited}
                return

            self._history.append({
                "role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield {"type": "tool_call", "name": name, "arguments": args}
                raw = await self._tools.call(name, args)
                parsed = _parse_result(raw)
                tool_calls_made.append({"name": name, "arguments": args, "result": parsed})
                for cit in _citations_from(name, parsed):
                    key = _citation_key(cit)
                    if key not in seen_cites:
                        seen_cites.add(key)
                        citations.append(cit)
                self._pool_add(citations)
                yield {"type": "tool_result", "name": name, "result": parsed}
                self._history.append({
                    "role": "tool", "tool_call_id": tc.get("id"), "name": name, "content": raw,
                })

        limit_msg = "I wasn't able to finish researching that within the tool-call limit."
        shown, cited = self._annotate(limit_msg, citations)
        yield {"type": "answer", "answer": shown, "reasoning": None,
               "tool_calls": tool_calls_made, "citations": cited}

    async def ask(self, user_text: str) -> dict[str, Any]:
        """Non-streaming convenience wrapper: returns the terminal answer event."""
        final: dict[str, Any] = {
            "answer": "", "reasoning": None, "tool_calls": [], "citations": []
        }
        async for event in self.stream(user_text):
            if event["type"] == "answer":
                final = {k: event[k] for k in ("answer", "reasoning", "tool_calls", "citations")}
        return final
