# case-chat — Design Document

Status: **Draft for operator review** · 2026-06-13

A proof-of-concept conversational chat over two read-only knowledge sources,
backed by DiffusionGemma on vLLM. This document consolidates the interview
outcomes and the decision records ([docs/decisions/](decisions/)) into one
reviewable design. **Implementation pauses here for operator review.**

---

## 1. Goal & boundaries
Answer questions via RAG over (1) the synthetic raw-document corpus and (2) the
domain-knowledge corpus, with an agentic tool-calling loop and full citations.

Hard boundaries (unchanged from project brief):
- **No Athena.** LLM = DiffusionGemma on vLLM `/v1`; embeddings = Qwen3-Embedding-4B `/embeddings`.
- **No extracted data.** RAG reads only raw source docs + domain-knowledge text.
  The fake-case structured dataset is synthesized from the *synthetic* corpus's
  fictional ground-truth, not case-project's real `casedb` ([ADR 0003](decisions/0003-fake-case-dataset-sqlite.md)).
- **Sovereignty.** Only fictional synthetic data + non-sensitive domain-knowledge
  *reference text* leave local hardware. Real `case-data/` never does.

## 2. Interview outcomes (settled)
| Topic | Decision |
|---|---|
| Tool calling | DiffusionGemma supports OpenAI `tools` → **agentic MCP loop** |
| RAG scope | **Both** indexes: new synthetic index + re-embedded domain corpus |
| Domain vectors | **Re-embed required** (Athena bug invalidated existing vectors); keep chunk text/metadata |
| Embedder | Qwen3-Embedding-4B via **TEI** on the box; local fallback on Mac; **2560-dim contract fixed** ([ADR 0004](decisions/0004-no-athena-embedding-seam.md)) |
| Vector store | **Qdrant** ([ADR 0001](decisions/0001-vector-store-qdrant.md)) |
| Collections | **By content domain; jurisdiction + doc_type as payload** ([ADR 0002](decisions/0002-collection-taxonomy.md)) |
| Packaging | **Standalone stdio MCP server + separate web app** |
| Conversation | **Multi-turn** with history |
| Citations | **Full**: source docs + binding/persuasive partition |
| Tool surface | All domain collections + synthetic + structured case tools; **no participants / no extracted data** |
| Case dataset | **SQLite** from ground-truth ([ADR 0003](decisions/0003-fake-case-dataset-sqlite.md)) |
| Success bar | **Informal** — answers grounded questions with citations in a live demo |
| Dev | **Locally**; deploy to 5090 later |

## 3. Topology
```
[Web app + chat orchestrator] ──MCP stdio──▶ [MCP retrieval server] ──▶ Qdrant
        │  (MCP client)                              │
        │ OpenAI /v1 (tools)                         ├─ /embeddings ─▶ Qwen3-Embedding-4B (TEI / local fallback)
        ▼                                            │
   DiffusionGemma (vLLM, 4-bit)                      └─ SQLite (fake-case dataset)
```
Three processes: vLLM (model), the stdio MCP retrieval server (tools), and the
web app (chat orchestrator + UI, acting as the MCP client and exposing the
tools to vLLM).

## 4. Data flow
**Index build (offline, run locally now / on box later):**
1. *Domain corpus:* stage the `knowledgedb` `pg_dump -Fc` in a throwaway Docker
   Postgres → read `full_text` + metadata (vectors ignored — invalid) → embed
   bare → upsert into Qdrant, routing rows to collections by `chunk_type`.
2. *Synthetic corpus:* per-type load + chunk the raw docs → embed bare → upsert
   into `synthetic_corpus`. Ground-truth JSON is **excluded** from the index.
3. *Fake-case dataset:* load ground-truth JSON into SQLite tables.

**Query (online):**
1. User message → orchestrator → vLLM with the MCP tools advertised.
2. Model emits tool calls → orchestrator runs them via the MCP client →
   retrieval server queries Qdrant / SQLite → results returned to the model.
3. Model produces a grounded answer; orchestrator renders citations (source
   docs; for `law` hits, binding/persuasive/non-authority for the active
   jurisdiction). Multi-turn history retained per session.

## 5. Collections & tools
| Collection | Holds | Tool | Filters |
|---|---|---|---|
| `law` | statutes + caselaw + constitutions, all jurisdictions | `kb.law.search` | `jurisdiction`, `doc_type` + binding partition |
| `behavioral_patterns` | pattern cards | `kb.pattern.search` | `wing` |
| `behavioral_sources` | framework sources/papers | `kb.behavioral_source.search` | — |
| `professional_standards` | ethics/practice codes | `kb.standards.search` | — |
| `scripture` | KJV + WEB | `kb.scripture.search` | `translation`, `book` |
| `synthetic_corpus` | raw case docs | `corpus.search` | `source_type` |
| *(SQLite)* | fake-case structured | `case.timeline.query`, `case.entity.lookup`, `case.facts.query`, `case.flags`, `case.observations` | per-tool |

**Two access modes per corpus — semantic search *and* exact keyed lookup.**
Some questions cite an exact reference ("James 2:3", "A.C.A. § 9-13-101") where
semantic kNN is the wrong tool. These resolve via a **Qdrant payload scroll**
(filter on identifier fields, *no embedding*), returning the exact passage(s) in
order:

| Lookup tool | Resolves | Mechanism |
|---|---|---|
| `kb.scripture.lookup` | a scripture reference — `James 2:3`, ranges `James 2:1-4`, whole chapters `James 2` | scroll `scripture` on `translation`/`book`/`chapter`/`verse` |
| `kb.law.lookup` | a statute/section citation — `A.C.A. § 9-13-101` | scroll `law` on `citation` (+ optional `jurisdiction`) |

The model picks `*.search` for conceptual questions and `*.lookup` when the user
names an exact reference/citation. Both lookups require the identifier fields
(book/chapter/verse; citation) to be carried into the Qdrant payload at
re-embed time — see §8.

## 5a. Access & auth ([ADR 0005](decisions/0005-web-auth-boundary.md))
The web UI is shared externally (operator + a reviewer in TN), so it sits behind
an auth boundary; everything else stays internal.
- **Per-user magic-link tokens** → httpOnly/Secure session cookie; revocable per
  user. Token CLI issues/revokes shareable links.
- **Cloudflare Tunnel** publishes only the web port over HTTPS; no inbound ports.
- vLLM, Qdrant, MCP server, SQLite all bind localhost on the box — the
  authenticated web app is the sole externally-reachable process.
- **Source viewer:** `GET /api/document?path=…` serves raw **test-corpus**
  documents only (clickable citations in the UI). Domain-knowledge text and
  ground-truth JSON are never served — viewability = membership in the exact set
  of indexer-ingested files, which also makes path traversal impossible. The
  reader renders by type: **messages → chat transcript** (vendored
  `rsmf_viewer` HTML in a sandboxed iframe), **notes → markdown**, everything
  else → raw text. Answers render markdown client-side (escaped-then-transformed,
  no injection). A left sidebar nav switches Chat ↔ Source-data browser, and the
  model's reasoning streams into a collapsible "Thinking" disclosure.

## 6. Embedding contract (load-bearing)
`Qwen/Qwen3-Embedding-4B` · 2560-dim · cosine · L2-normalized · asymmetric
(queries wrapped `Instruct: <task>\nQuery: …`, documents bare). Enforced in
[`case_chat/embeddings/client.py`](../case_chat/embeddings/client.py); dim is
asserted on every response. Identical to domain-knowledge's build side so
vectors converge. ([ADR 0004](decisions/0004-no-athena-embedding-seam.md))

## 7. Synthetic corpus — per-type chunking (proposed)
| Source | Count | Strategy |
|---|---|---|
| Apple notes (.md) | 13 | Split on `##` headings, ~1200 chars; frontmatter → metadata |
| OCR stubs (.jpeg.txt) | 6 | Indexed as-is |
| Messages (**RSMF**) | 7 | Parsed via the vendored `rsmf_viewer` (canonical format, matches case-project); windowed ~15-msg / ~1200-char turns with resolved participants + timestamps. The `.xml`/CSV siblings are ignored. |
| Emails (.eml) | 11 | One chunk/message; headers → metadata; split body >1500 chars |
| Court docs (.txt) | 7 | Split on section headings (`I.`, `II.`, numbered); breadcrumb → metadata; ~2000 cap |
| Visitation (CSV + .schema.yaml) | 1 | Row-group chunks using schema field labels; long `observer_notes` per-visit |
| Transcripts / witness (.txt) | 3 / 2 | Paragraph/speaker-turn windows, ~1500 chars |
| PDFs (court) / .m4a (transcript) | 2 / 1 | **Out of scope** (confirmed) — skipped for the POC |

Ground-truth JSON (`manifest/entities/timeline/master_facts/expected_flags/expected_observations`)
is **never indexed** — it backs the SQLite case tools only.

## 8. knowledgedb → Qdrant routing
| Source table | → Collection | Notes |
|---|---|---|
| `family_law` | `law` | `doc_type` from `chunk_type` (statute/opinion); `jurisdiction`, `binding_jurisdictions[]`, **`citation`** carried (for `kb.law.lookup`) |
| `constitutional` | `law` | `doc_type=constitution` / `opinion`; `jurisdiction=federal`(US) or `ar`; `citation` |
| `behavioral_patterns` | `behavioral_patterns` | `wing`, `card_id`, `framework` |
| `behavioral_sources` | `behavioral_sources` | — |
| `professional_standards` | `professional_standards` | — |
| `bible_kjv` / `bible_web` | `scripture` | `translation=kjv` / `web`; **`book`/`chapter`/`verse`** carried (for `kb.scripture.lookup`) |

> Re-embed must verify the source bible tables expose verse coordinates
> (book/chapter/verse) and that `family_law`/`constitutional` expose `citation`,
> mapping them into Qdrant payload — these back the exact-lookup tools (§5).
Row counts ≈ 113k total (family_law 47.9k; bibles 62k). Local dev **skips the
two bibles by default** (toggleable); full set re-embedded on the box.

## 9. Build status
All components built and tested (46 tests green). End-to-end validated locally
against `gemma4:e4b-mlx` via Ollama and through the web app's auth + SSE path.
- ✅ embedding client (contract + retry/truncation) · Qdrant retriever + binding partition
- ✅ docker-compose infra · knowledgedb re-embed pipeline (hardened) · synthetic indexer
- ✅ SQLite fake-case dataset + query layer · stdio MCP server (13 tools)
- ✅ chat orchestrator (agentic loop, streaming) · web UI (magic-link auth, SSE, citations)
- ✅ Makefile · scripts · RUNBOOK · `.env.box.example` · ADRs 0001–0006
- ⏳ **Full domain re-embed runs in the background** (Ollama ~8 chunks/s; family_law
  dominates). Bibles (`make reembed-bibles`) optional locally. On the box (TEI) the
  full build is fast.
- 🔜 **Box bring-up only:** stand up vLLM (DiffusionGemma) + TEI; set `.env`; expose
  via Cloudflare Tunnel. See [RUNBOOK.md](RUNBOOK.md).

## 10. Open items — review status
1. ~~vLLM specifics~~ — **resolved.** `google/diffusiongemma-26B-A4B-it`, port
   8000, `/v1`, api key `EMPTY`. Tool calling: `--enable-auto-tool-choice
   --tool-call-parser gemma4`. Thinking: `--reasoning-parser gemma4` (responses
   carry `reasoning_content`, kept out of conversation context). Recipe's
   `--max-model-len 262144` targets a B200; the 5090/4-bit deployment will use a
   quant repo id + smaller context (both overridable in config).
2. ~~PDFs + .m4a~~ — **resolved: out of scope** for the POC.
3. ~~Local-dev embedder~~ — **resolved: run Qwen3-Embedding-4B locally** (e.g.
   sentence-transformers) over the same `/embeddings` contract.
4. ~~Scripture by reference~~ — **resolved: added `kb.scripture.lookup`** (exact,
   payload scroll) alongside `kb.scripture.search`; same dual-mode added to law
   as `kb.law.lookup` (§5).

Remaining blocker to a *runnable* end-to-end demo: vLLM details (#1). Build of
indexing/retrieval/MCP can proceed without them.
