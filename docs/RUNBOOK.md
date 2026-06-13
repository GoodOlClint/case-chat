# case-chat — Runbook

How to build the indexes and run the app, locally and on the 5090 box. See
[DESIGN.md](DESIGN.md) for what it all is and [decisions/](decisions/) for why.

## Prerequisites
- Docker (Qdrant + a throwaway Postgres for the dump)
- `uv` (Python 3.12 env)
- Ollama with the local models: `make ollama-pull`
  (`qwen3-embedding:4b` for embeddings, `gemma4:e4b-mlx` for chat)
- The knowledgedb dump at `~/Source/domain-knowledge/dist/knowledgedb-*.dump`
- The synthetic corpus at `~/Source/synthetic-test-corpus/`

## Local build (one time)
```bash
uv sync --group dev          # create the venv
make up                      # Qdrant + staging Postgres
make restore                 # stage the knowledgedb dump (reads good text/metadata)
make reembed                 # re-embed domain corpus → Qdrant (skips bibles; ~1.5–2h on Ollama)
make reembed-bibles          # OPTIONAL: add scripture (62k verses; another ~2h)
make index-synthetic         # raw-doc index (214 chunks; fast)
make build-casedata          # SQLite fake-case dataset (instant)
make test                    # full suite should be green
```
Notes:
- Re-embedding throughput on Ollama is ~8 chunks/s; `family_law` (47.9k) dominates.
  The smaller domain tables populate after it. The build is resilient (retries
  transient errors, skips a persistently-bad batch, logs it).
- A handful of anomalous chunks (>32k chars) are truncated to the embedder
  context — expected, logged by the contract, not a failure.

## Run locally
```bash
make web                                   # http://127.0.0.1:8080
make link SUBJECT="Me"                     # prints a magic-link; open it to sign in
```
Then chat. The UI streams tool calls live and shows sources (and, for legal
hits, binding vs persuasive authority). Local chat uses `gemma4:e4b-mlx` — good
enough to validate the loop; the box model is far stronger.

## Deploy on the 5090 box — Docker Compose stack
The whole thing ships as a compose stack (vLLM + TEI + Qdrant + web + Cloudflare
Tunnel, plus a one-time build profile). Full steps in **[../deploy/README.md](../deploy/README.md)**.

Quick version:
```bash
cd deploy && cp .env.example .env   # set VLLM_MODEL, TUNNEL_TOKEN, secrets, mount paths
docker compose --env-file .env up -d qdrant tei
docker compose --env-file .env --profile build run --rm builder   # populate Qdrant + SQLite
docker compose --env-file .env up -d                              # serve everything
docker compose --env-file .env exec web python -m case_chat.web.auth issue --subject "Friend in TN"
```
Only the web app is reachable, and only through the Cloudflare Tunnel — vLLM,
TEI, Qdrant, and the MCP server (in the web container) stay internal.
Sovereignty: only the fictional synthetic corpus + non-sensitive reference text
are mounted; real `case-data/` is never deployed.

## Operational notes
- **Reset a conversation**: the "New conversation" button (or `POST /api/reset`).
- **Health**: `GET /healthz` → `{"ok":true,"mcp_tools":13}`.
- **Switching the chat model** is pure config (`CASECHAT_VLLM_*`) — the
  orchestrator only needs an OpenAI-compatible `/v1` with tool-calling.
- **Embedding contract is load-bearing**: keep Qwen3-Embedding-4B / 2560-dim /
  cosine / L2 / `Instruct:`/`Query:` wrapping identical on build and query sides
  or cosine ranks corrupt silently. The client asserts dim on every call.
