# case-chat — Cloud deploy (Docker Compose)

A self-contained stack for the RTX 5090 box. Only the web app is exposed, and
only through a Cloudflare Tunnel — vLLM, TEI, Qdrant, and the MCP server (which
runs inside the web container) all stay on the internal compose network.

```
                         ┌─────────────── compose network ───────────────┐
 Internet ─ Cloudflare ─►│ cloudflared ─► web ─┬─► vllm   (DiffusionGemma)│
            Tunnel       │                     ├─► tei    (Qwen3-Embed-4B) │
                         │                     └─► qdrant (vectors)        │
                         │   web also runs the MCP server + reads SQLite   │
                         └────────────────────────────────────────────────┘
```

## Prerequisites (on the box)
- Docker + Docker Compose, and the **NVIDIA Container Toolkit** (for `vllm`/`tei`).
- The **knowledgedb dump** and the **synthetic corpus** on disk.
- A **Cloudflare Tunnel** with a public hostname routed to `http://web:8080`,
  and its connector **token**.

## 1. Configure
```bash
cd deploy
cp .env.example .env
# set: VLLM_MODEL (the 4-bit quant you can fit), VLLM_MAX_MODEL_LEN, VLLM_GPU_FRACTION,
#      HF_TOKEN, TUNNEL_TOKEN, DUMP_PATH, SYNTHETIC_CORPUS_PATH,
#      CASECHAT_WEB_AUTH_SECRET (openssl rand -hex 32), CASECHAT_WEB_PUBLIC_BASE_URL
```
VRAM note: vLLM and TEI share the 32GB card. Tune `VLLM_GPU_FRACTION` (e.g. 0.6)
and `VLLM_MAX_MODEL_LEN` so both fit alongside the embedder.

## 2. One-time index build
Brings up the GPU embedder + a throwaway Postgres, restores the dump, and
populates Qdrant + the fake-case SQLite:
```bash
docker compose --env-file .env up -d qdrant tei
docker compose --env-file .env --profile build run --rm builder
```
(Embedding the full corpus + bibles takes a while on first run, but TEI on the
5090 is far faster than local Ollama.)

## 3. Serve
```bash
docker compose --env-file .env up -d        # qdrant, tei, vllm, web, cloudflared
docker compose --env-file .env ps
docker compose --env-file .env logs -f web vllm
```

## 4. Issue access links
The web container is the only thing behind the tunnel. Mint per-user magic links:
```bash
docker compose --env-file .env exec web python -m case_chat.web.auth issue --subject "Friend in TN"
docker compose --env-file .env exec web python -m case_chat.web.auth revoke --subject "Friend in TN"
```
Send the printed URL. First click sets an httpOnly session cookie.

## Notes
- **Switching the chat model** is just `VLLM_MODEL` (+ a restart of `vllm`/`web`).
  The app only needs an OpenAI-compatible `/v1` with `gemma4` tool-calling.
- **Embedding parity**: the box builds *and* queries with TEI, so vectors are
  self-consistent. The 2560-dim / cosine / L2 / `Instruct:`/`Query:` contract is
  identical to local dev (see ADR 0004).
- **Sovereignty**: only the fictional synthetic corpus and the non-sensitive
  domain-knowledge reference text are mounted into the stack. Real `case-data/`
  is never deployed.
- **TEI image tag**: `turing-1.5` is a CUDA build; pick the tag matching the
  box's GPU architecture (Blackwell/sm_120 may need a newer tag — check the TEI
  releases) if the default doesn't start.
