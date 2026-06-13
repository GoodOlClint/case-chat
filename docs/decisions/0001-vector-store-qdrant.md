# 0001 — Vector store: Qdrant (not pgvector)

Status: Accepted · 2026-06-13

## Context
case-chat is greenfield. The existing domain-knowledge corpus lives in a
pgvector `knowledgedb` (`halfvec(2560)` HNSW). History: the project ran Qdrant,
moved to pgvector to simplify the data layer, then hit pgvector's HNSW
dimension cap (`vector` ≤ 2000-dim, `halfvec` ≤ 4000-dim), which forced the
Qwen3-8B/4096 → Qwen3-4B/2560 + `halfvec` compromise. Separately, the Athena
embedding bug means every existing `knowledgedb` vector is invalid and must be
regenerated — so this POC re-embeds everything regardless.

## Decision
Use **Qdrant** as the vector store for both the re-embedded domain corpus and
the new synthetic index.

## Why
- Full re-embed is happening anyway → **zero migration cost** to choose freely.
- **No HNSW dimension cap** — keeps a future move back to Qwen3-8B/4096 open
  without re-architecting storage.
- Native f16 storage + scalar/binary quantization handle index growth better
  than the pgvector setup that bit us before.
- Rich payload filtering cleanly replaces pgvector's typed metadata columns;
  `partition_by_binding` is app-side logic either way.
- One `docker run` on the single-box POC.

## Consequences
- The ported `LegalCorpusRetriever` SQL is rewritten thin against the Qdrant
  client; binding/persuasive partitioning moves fully app-side.
- **Convergence preserved:** the load-bearing contract is the *embedding*
  (Qwen3-4B / 2560 / cosine / L2 / asymmetric wrap — see
  [0004](0004-no-athena-embedding-seam.md)), and vectors are portable between
  Qdrant and pgvector. Storage engine choice does not block converging with
  `knowledgedb` later.

## Alternatives rejected
- **pgvector** — would technically work at 2560-dim (the dim cap doesn't bite
  there), and would let us reuse the retriever SQL. Rejected: it is the engine
  that hit the scaling wall, and greenfield + full re-embed removes its only
  real advantage (no-migration reuse).
