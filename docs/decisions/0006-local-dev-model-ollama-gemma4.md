# 0006 — Local-dev chat model via Ollama (gemma4:e4b-mlx)

Status: Accepted · 2026-06-13

## Context
The box runs DiffusionGemma-26B-A4B on vLLM. During local development on a Mac
we want to exercise the full agentic loop (tool selection → MCP → citations →
streaming UI) without waiting for the box. Question raised: install vLLM locally
and run a different Gemma4 model?

## Decision
**Do not install vLLM on the Mac.** Point the chat orchestrator at **Ollama**
serving **`gemma4:e4b-mlx`** for local dev. Keep embeddings on Ollama
(`qwen3-embedding:4b`). The box overrides via env to real vLLM + DiffusionGemma.

## Why
- vLLM has no real Apple-Silicon/Metal backend — a CPU-only source build of a
  Gemma4 model would be slow and fragile.
- The orchestrator only needs OpenAI-compatible `/v1/chat/completions` with
  `tools`. Ollama provides exactly that, and **`gemma4:e4b-mlx` does OpenAI
  tool-calling correctly** (verified). It's the *same model family* (gemma4) as
  the box, MLX-accelerated.
- This validates the wire-level plumbing the loop depends on with near-zero
  setup. What it does *not* reproduce — DiffusionGemma's exact tool-selection
  quality and gemma4 `reasoning_content` — are box-only concerns checked at
  deploy.

## Config
Local defaults (in `config.py`): `vllm_base_url=http://localhost:11434/v1`,
`vllm_model=gemma4:e4b-mlx`, `vllm_send_chat_template_kwargs=false` (the gemma4
thinking toggle is a vLLM extension Ollama rejects). Box env overrides:
`CASECHAT_VLLM_BASE_URL=http://localhost:8000/v1`,
`CASECHAT_VLLM_MODEL=google/diffusiongemma-26B-A4B-it`,
`CASECHAT_VLLM_SEND_CHAT_TEMPLATE_KWARGS=true`.
