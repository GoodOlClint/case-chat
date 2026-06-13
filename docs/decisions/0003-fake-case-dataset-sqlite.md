# 0003 — Fake-case structured dataset in SQLite

Status: Accepted · 2026-06-13

## Context
The synthetic test corpus ships ground-truth JSON (`entities.json`,
`timeline.json`, `master_facts.json`, `expected_flags.json`,
`expected_observations.json`, `manifest.json`). These are evaluation artifacts
and are **not** indexed for RAG. But they describe the fictional Holcomb case
the way case-project's extraction pipeline will eventually describe a real case,
and they make exact questions ("when was the guardianship petition filed?",
"who is Kaylee Holcomb?") trivially answerable.

## Decision
Build a small **SQLite** "fake-case" dataset from the synthetic ground-truth
JSON and expose exact-lookup MCP tools over it:
`case.timeline.query`, `case.entity.lookup`, `case.facts.query`, `case.flags`,
`case.observations`.

## Boundary check
This does **not** violate the "no extracted data" rule. That rule forbids
reading case-project's **real `casedb`** (timeline events, evidence,
observations, resolved participants from the extraction pipeline). Here we
synthesize a *parallel, fictional* structured layer from the **synthetic**
corpus's ground-truth — fully fictional, safe to ship, a faithful stand-in for
the future capability. It stays out of the RAG vector index; it backs exact
lookups only.

## Caveat (accepted)
Exposing ground-truth as a queryable tool means it can no longer serve as a
clean eval *oracle* for those same facts (circularity). Success criteria for
the POC are informal, so this is acceptable.

## Why SQLite
- Models the shape of a future relational `casedb` without standing up a server.
- Supports range/filter queries (date ranges, entity joins) cleanly — better
  than in-memory JSON scans for the timeline/entity tools.
- Zero infra; file lives beside the code.
