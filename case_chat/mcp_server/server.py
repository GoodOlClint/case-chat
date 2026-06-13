"""Standalone stdio MCP server exposing case-chat retrieval tools.

Run:  uv run python -m case_chat.mcp_server.server   (speaks MCP over stdio)

Tool names use underscores (not dots) so they're valid OpenAI/gemma4 function
names when the orchestrator advertises them to vLLM. Each tool's docstring is
the description the model sees — keep them about *when* to call the tool.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from case_chat.mcp_server import tools

mcp = FastMCP("case-chat")


# ---- domain knowledge: law -----------------------------------------------
@mcp.tool(name="kb_law_search")
def kb_law_search(
    query: str, jurisdiction: str | None = None, doc_type: str | None = None, limit: int = 8
) -> dict:
    """Search Arkansas + federal statutes, case-law, and constitutions by concept.

    Use for legal questions ("what is the standard for guardianship of a minor").
    jurisdiction: 'ar', 'federal', or omit for all. doc_type: 'statute',
    'opinion', 'constitution', or omit. Results are labeled binding / persuasive /
    non-authority for Arkansas.
    """
    return tools.law_search(query, jurisdiction=jurisdiction, doc_type=doc_type, limit=limit)


@mcp.tool(name="kb_law_lookup")
def kb_law_lookup(citation: str, jurisdiction: str | None = None, limit: int = 10) -> dict:
    """Fetch a specific statute/section/case by its EXACT citation (no search).

    Use when the user names a citation, e.g. 'Ark. Code Ann. § 9-13-101'. For
    conceptual questions use kb_law_search instead.
    """
    return tools.law_lookup(citation, jurisdiction=jurisdiction, limit=limit)


# ---- domain knowledge: scripture -----------------------------------------
@mcp.tool(name="kb_scripture_search")
def kb_scripture_search(
    query: str, translation: str | None = None, book: str | None = None, limit: int = 8
) -> dict:
    """Search the Bible (KJV + WEB) by theme/concept. translation: 'kjv'/'web'/omit."""
    return tools.scripture_search(query, translation=translation, book=book, limit=limit)


@mcp.tool(name="kb_scripture_lookup")
def kb_scripture_lookup(reference: str, translation: str | None = None) -> dict:
    """Fetch exact verse(s) by reference: 'James 2:3', 'James 2:1-4', or 'James 2'.

    Use when the user cites a specific passage. translation omitted → both KJV+WEB.
    """
    return tools.scripture_lookup(reference, translation=translation)


# ---- domain knowledge: behavioral / standards ----------------------------
@mcp.tool(name="kb_pattern_search")
def kb_pattern_search(query: str, wing: str | None = None, limit: int = 8) -> dict:
    """Search behavioral-pattern cards (coercive control, abuse dynamics,
    high-control religion, psychology). wing optionally narrows the framework."""
    return tools.pattern_search(query, wing=wing, limit=limit)


@mcp.tool(name="kb_behavioral_source_search")
def kb_behavioral_source_search(query: str, limit: int = 8) -> dict:
    """Search the source notes / academic papers behind the behavioral frameworks."""
    return tools.behavioral_source_search(query, limit=limit)


@mcp.tool(name="kb_standards_search")
def kb_standards_search(query: str, limit: int = 8) -> dict:
    """Search professional ethics / practice standards for family-law professionals."""
    return tools.standards_search(query, limit=limit)


# ---- raw case documents ---------------------------------------------------
@mcp.tool(name="corpus_search")
def corpus_search(query: str, source_type: str | None = None, limit: int = 8) -> dict:
    """Search the raw case documents (apple_note, email, messages, court_document,
    transcript, witness_statement, visitation_log, attachment_ocr). source_type
    optionally narrows to one document type."""
    return tools.corpus_search(query, source_type=source_type, limit=limit)


# ---- structured fake-case dataset ----------------------------------------
@mcp.tool(name="case_timeline_query")
def case_timeline_query(
    date_from: str | None = None, date_to: str | None = None, entity: str | None = None,
    category: str | None = None, text: str | None = None, limit: int = 25
) -> dict:
    """Query the case timeline of events (structured). Filter by ISO date range,
    entity (name/alias/id), category, or text. Use for 'when did X happen'."""
    return tools.case_timeline(date_from=date_from, date_to=date_to, entity=entity,
                               category=category, text=text, limit=limit)


@mcp.tool(name="case_overview")
def case_overview() -> dict:
    """Overview of the case: case number, jurisdiction, dataset counts, and the FULL
    roster of participants (id, name, role, DOB). Call this for any question about
    the case as a whole or its people — 'who are the participants/parties', 'what is
    this case', 'who are the respondents' — instead of asking the user for names."""
    return tools.case_overview()


@mcp.tool(name="case_participants")
def case_participants(role: str | None = None, entity_type: str | None = None) -> dict:
    """List case participants, optionally filtered by role substring (e.g.
    'respondent', 'petitioner', 'minor', 'child', 'attorney', 'witness', 'judge')
    or type. Use for 'who are the respondents', 'list the children', etc."""
    return tools.case_participants(role=role, entity_type=entity_type)


@mcp.tool(name="case_entity_lookup")
def case_entity_lookup(name: str, limit: int = 5) -> dict:
    """Look up ONE case participant by name/alias/id → role, DOB, aliases, and
    relationships resolved to names (both their stated relationships and who
    references them). Use for 'who is X' or 'how is X related to Y'."""
    return tools.case_entity(name, limit=limit)


@mcp.tool(name="case_facts_query")
def case_facts_query(
    subject: str | None = None, predicate: str | None = None, obj: str | None = None,
    category: str | None = None, text: str | None = None, limit: int = 25
) -> dict:
    """Query established case facts (subject-predicate-object). Filter by any part or text."""
    return tools.case_facts(subject=subject, predicate=predicate, obj=obj,
                            category=category, text=text, limit=limit)


@mcp.tool(name="case_flags")
def case_flags(
    flag_type: str | None = None, severity: str | None = None, text: str | None = None, limit: int = 25
) -> dict:
    """Query flagged issues/allegations in the case. Filter by type, severity, or text."""
    return tools.case_flags(flag_type=flag_type, severity=severity, text=text, limit=limit)


@mcp.tool(name="case_observations")
def case_observations(
    observer: str | None = None, subject: str | None = None, text: str | None = None, limit: int = 25
) -> dict:
    """Query recorded observations/claims. Filter by observer, referent subject, or text."""
    return tools.case_observations(observer=observer, subject=subject, text=text, limit=limit)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
