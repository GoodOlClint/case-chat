# case-chat

A proof-of-concept conversational chat interface backed by **DiffusionGemma served
via vLLM** (OpenAI-compatible `/v1`) on a cloud-rented RTX 5090. It answers
questions using RAG over two read-only knowledge sources:

1. the **synthetic test corpus** of raw source documents (fictional Holcomb
   family Arkansas guardianship case), and
2. the existing **domain-knowledge corpus** (legal / behavioral / scripture),
   re-embedded into Qdrant and reachable via **MCP**.

It also exposes a **structured fake-case dataset** built from the synthetic
corpus ground-truth (timeline / entities / facts / flags / observations) so
exact questions like "when was the guardianship petition filed?" resolve
against structured data — a stand-in for what case-project's extraction
pipeline will eventually provide.

## Hard boundaries
- **No Athena.** LLM + embeddings go through vLLM / a Qwen3-Embedding-4B
  `/embeddings` endpoint, never the Athena daemon.
- **No extracted data.** RAG reads only raw source documents + the
  domain-knowledge corpus. case-project's `casedb` (timeline events, evidence,
  observations, resolved participants, …) is off-limits. The fake-case dataset
  here is synthesized from the *synthetic* corpus's ground-truth, which is
  fictional — not the real `casedb`.
- **Data sovereignty.** Only the fictional synthetic corpus and the
  non-sensitive domain-knowledge *reference* text leave local hardware. Real
  `case-data/` never does.

## Architecture
```
[Web app + chat orchestrator] ──MCP stdio──▶ [MCP retrieval server] ──▶ Qdrant
        │                                            │
        │ OpenAI /v1 (tools)                         ├─ /embeddings ─▶ Qwen3-Embedding-4B
        ▼                                            │                 (TEI on box / local fallback)
   DiffusionGemma (vLLM, 4-bit)                      └─ SQLite (fake-case dataset)
```

## Embedding contract (load-bearing)
`Qwen/Qwen3-Embedding-4B` · 2560-dim · cosine · L2-normalized · asymmetric
(queries wrapped `Instruct: …\nQuery: …`, documents bare). Kept identical to
domain-knowledge's build side so vectors converge. See
[`case_chat/embeddings/client.py`](case_chat/embeddings/client.py).

## Decisions
Architecture decisions are recorded under [`docs/decisions/`](docs/decisions/).

## Status
POC under construction. See the implementation plan / todo list.
