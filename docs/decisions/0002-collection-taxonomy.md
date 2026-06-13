# 0002 — Collection taxonomy: by content domain, jurisdiction as payload

Status: Accepted · 2026-06-13

## Context
knowledgedb organizes the legal corpus as `family_law` (AR Code + AR
Constitution + AR appellate opinions + federal family statutes ICWA/PKPA/VAWA +
multi-state caselaw) and a separate `constitutional` (US Constitution + SCOTUS).
Two naming/structure problems surfaced:
1. "family_law" undersells a collection that is really broad Arkansas + federal
   law.
2. We expect to add other states later (e.g. Missouri). Question raised: does
   mixing federal into an "Arkansas" collection cause problems, or should all
   statute go into one collection filtered by jurisdiction?

## Decision
Organize Qdrant collections **by content domain**, and make **jurisdiction and
doc-type payload fields filtered at query time** — not collection boundaries.

Collections:

| Collection | Holds | Tool | Key payload filters |
|---|---|---|---|
| `law` | statutes + caselaw + constitutions, **all jurisdictions** | `kb.law.search` | `jurisdiction` (ar/federal/…), `doc_type` (statute/opinion/constitution), `binding_jurisdictions[]` |
| `behavioral_patterns` | pattern cards | `kb.pattern.search` | `wing` |
| `behavioral_sources` | framework sources / papers | `kb.behavioral_source.search` | — |
| `professional_standards` | ethics / practice codes | `kb.standards.search` | — |
| `scripture` | KJV + WEB | `kb.scripture.search` | `translation`, `book` |
| `synthetic_corpus` | raw case documents | `corpus.search` | `source_type` |

The old `constitutional` table folds into `law` (`doc_type=constitution`,
`jurisdiction=federal` for the US Constitution / `ar` for the AR Constitution).

## Why
- **Jurisdiction is data, not structure.** Each chunk carries `jurisdiction` +
  `binding_jurisdictions[]`. "Federal" is just another jurisdiction value; the
  binding logic already treats federal as binding-on-states. Adding Missouri is
  additive rows (`jurisdiction=mo`) — no new collections, no re-embed of
  existing data.
- **Unified ranking.** Family-law questions routinely span AR + federal (ICWA
  preemption, PKPA/UCCJEA interstate jurisdiction) and statute + the case
  interpreting it. One collection returns them in a single ranked pass;
  `partition_by_binding(active_jurisdiction)` then labels each hit
  binding/persuasive/non-authority for whichever state is active.
- **A payload filter is strictly more flexible than a split** — "AR only",
  "federal only", or "both + partition" all come from one index by toggling a
  filter. A hard jurisdiction split only gives the rigid version and forces the
  model to merge across collections.

## Consequences
- Re-embed routes knowledgedb rows into collections by content type (see
  [0005](0005-knowledgedb-reembed.md) when written): `family_law` rows split by
  `chunk_type` into `law` with `doc_type` derived; `constitutional` likewise;
  `bible_kjv`/`bible_web` → `scripture` with `translation`.
- `kb.law.search` takes optional `jurisdiction` and `doc_type` params backed by
  Qdrant payload filters, plus always returns the binding partition.

## Alternatives rejected
- **Split by jurisdiction** (`ar_code`, `federal_code`, `mo_code`, …) — explodes
  collection count with multi-state and forces cross-collection merge.
- **Split by content type** (`statutes`, `caselaw`, `constitutions`) — separates
  a statute from the case interpreting it; rejected for the unified-ranking
  reason. `doc_type` payload preserves the option to narrow without the split.

## Amendment (2026-06-13) — dual access mode: search + exact lookup
Some questions name an **exact reference/citation** ("James 2:3",
"A.C.A. § 9-13-101") where semantic kNN is the wrong tool. Each such corpus
therefore gets a second, non-semantic tool that resolves the identifier via a
**Qdrant payload scroll** (filter on identifier fields, no embedding):

- `kb.scripture.lookup` — verse reference (single, range, or whole chapter);
  scrolls `scripture` on `translation`/`book`/`chapter`/`verse`.
- `kb.law.lookup` — statute/section citation; scrolls `law` on `citation`.

This generalizes case-project's non-vector `fetch_by_column` read. Requirement:
the re-embed must carry the identifier fields (book/chapter/verse for scripture;
`citation` for law) into the Qdrant payload, or the lookup tools have nothing to
filter on. The model chooses `*.search` for conceptual questions and `*.lookup`
when an exact reference is named.
