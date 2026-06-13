# 0004 — Embedding seam: OpenAI /embeddings, no Athena

Status: Accepted · 2026-06-13

## Context
case-project's `legal_corpus/embeddings.py` POSTs to Athena's `/v1/embeddings`
with a bearer token. This POC must not use Athena. The embedding *contract*,
however, is load-bearing: it must match the side that built (now: re-builds) the
index, or cosine ranks corrupt silently (a wrong-dim or wrong-wrapping vector is
not an error, just nonsense ranking).

## Decision
Port the embedding client Athena-free, talking to an OpenAI-compatible
`/embeddings` endpoint (config `CASECHAT_EMBEDDINGS_BASE_URL`). Keep the
contract identical:

- Model `Qwen/Qwen3-Embedding-4B`, **2560-dim**, cosine.
- **L2-normalized** — enforced client-side regardless of backend, so a server
  that forgets to normalize cannot corrupt ranks.
- **Asymmetric retrieval**: queries wrapped `Instruct: <QWEN3_QUERY_TASK>\nQuery:
  <text>`; documents embedded **bare**. `QWEN3_QUERY_TASK` is pinned in code and
  kept verbatim from case-project / domain-knowledge.
- Dimension is asserted on every response; a mismatch is a hard error.

## Serving
- **On the 5090:** HF Text-Embeddings-Inference (TEI) serving Qwen3-Embedding-4B
  (leaner than a second vLLM beside DiffusionGemma on 32GB).
- **Local dev (Mac):** a fallback server exposing the *same* model over the same
  `/embeddings` route (sentence-transformers / infinity). Same contract; only
  the base URL differs.

## Consequences
- Both the re-embedded domain corpus and the synthetic index embed documents
  bare through `embed_documents`, so build/query parity is true by construction.
- Vectors remain portable to pgvector for later convergence with `knowledgedb`.

See [`case_chat/embeddings/client.py`](../../case_chat/embeddings/client.py) and
its contract tests.
