"""Central configuration for the case-chat POC.

All hosts/ports/keys live here, env-overridable (prefix ``CASECHAT_``).
Nothing in this repo talks to Athena — the LLM is DiffusionGemma on vLLM's
OpenAI-compatible ``/v1`` server and embeddings come from a Qwen3-Embedding-4B
``/embeddings`` endpoint (TEI on the box, a local fallback server on the Mac).
The embedding *contract* (model id, 2560-dim, cosine, L2-norm, asymmetric
Instruct/Query wrap) is fixed regardless of which server answers — see
``case_chat.embeddings.client``.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CASECHAT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Chat model (OpenAI-compatible /v1) --------------------------------
    # LOCAL DEV defaults point at Ollama serving `gemma4:e4b-mlx` — same model
    # family as the box (gemma4) and it does OpenAI tool-calling, so the agentic
    # loop is validated locally without installing vLLM on the Mac.
    #
    # ON THE BOX (5090), override via env to real vLLM serving DiffusionGemma:
    #   CASECHAT_VLLM_BASE_URL=http://localhost:8000/v1
    #   CASECHAT_VLLM_MODEL=google/diffusiongemma-26B-A4B-it   (or the 4-bit quant repo id)
    #   CASECHAT_VLLM_SEND_CHAT_TEMPLATE_KWARGS=true           (enables gemma4 thinking toggle)
    # vLLM serves tool-calling via --enable-auto-tool-choice --tool-call-parser
    # gemma4 and thinking via --reasoning-parser gemma4 (recipe ctx 262144 on B200;
    # smaller on the 5090/4-bit).
    vllm_base_url: str = "http://localhost:11434/v1"  # Ollama (local dev)
    vllm_api_key: str = "EMPTY"
    # gemma4:26b-mlx is the closest local proxy for DiffusionGemma-26B-A4B.
    # (gemma4:e4b-mlx is a lighter fallback if 26b isn't pulled.)
    vllm_model: str = "gemma4:26b-mlx"  # box: google/diffusiongemma-26B-A4B-it
    vllm_context_window: int = 32768
    # chat_template_kwargs is a vLLM/gemma4 extension; Ollama rejects it, so it's
    # off locally and enabled on the box.
    vllm_send_chat_template_kwargs: bool = False
    vllm_enable_thinking: bool = True  # only sent when send_chat_template_kwargs
    vllm_max_tool_iterations: int = 6  # agentic loop safety bound

    # --- Embeddings: Qwen3-Embedding-4B, OpenAI /embeddings ----------------
    # Local dev: Ollama (verified 2560-dim). On the 5090: TEI serving
    # Qwen/Qwen3-Embedding-4B at the same /embeddings contract — override
    # base_url + model. The client L2-normalizes + asserts dim regardless.
    embeddings_base_url: str = "http://localhost:11434/v1"  # Ollama
    embeddings_api_key: str = "EMPTY"
    embeddings_model: str = "qwen3-embedding:4b"  # box: "Qwen/Qwen3-Embedding-4B"
    embeddings_dim: int = 2560  # LOAD-BEARING: must match knowledgedb vectors
    embeddings_batch_size: int = 16  # small batches: faster per-request, resilient
    embeddings_timeout_secs: float = 120.0
    embeddings_max_retries: int = 4  # retry transient timeouts/5xx during long builds
    # Defensive cap: a few dozen anomalous knowledgedb chunks are 100k+ chars,
    # overrunning the embedder context. Truncate to a safe budget (well under
    # Ollama's 40960-token ctx). Only the rare oversized outlier is clipped; the
    # original Athena build truncated at model ctx too. The TEI box build should
    # honor the same cap for parity on those outliers.
    embeddings_max_chars: int = 32000

    # --- Qdrant ------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_search_hnsw_ef: int = 100  # query-time ef (mirrors legal_corpus_hnsw_ef_search)

    # --- Jurisdiction / authority -----------------------------------------
    # Active jurisdiction for binding-vs-persuasive partitioning. Today "ar";
    # multi-state later is just a different value (e.g. "mo") — no re-embed.
    active_jurisdiction: str = "ar"

    # --- knowledgedb dump staging (re-embed source) ------------------------
    # The pg_dump -Fc artifact holding GOOD chunk text + metadata. Its vectors
    # are INVALID (Athena embedding bug) and are never read — we regenerate.
    knowledgedb_dump_path: str = (
        "~/Source/domain-knowledge/dist/"
        "knowledgedb-71123c1dirty_2026-05-27T181856Z.dump"
    )
    # Throwaway Docker Postgres used only to read the dump's text/metadata.
    knowledgedb_staging_dsn: str = (
        "postgresql://postgres:postgres@localhost:5544/knowledgedb"
    )

    # --- Web app + auth ([ADR 0005]) --------------------------------------
    # The web app is the ONLY externally-reachable process (behind auth, via
    # Cloudflare Tunnel). Everything else binds localhost. Set a real secret in
    # prod: CASECHAT_WEB_AUTH_SECRET. web_public_base_url is used to mint links.
    web_host: str = "127.0.0.1"
    web_port: int = 8080
    web_auth_secret: str = "dev-insecure-change-me"  # OVERRIDE in prod
    web_session_ttl_days: int = 14
    web_public_base_url: str = "http://localhost:8080"
    web_session_cookie: str = "casechat_session"
    web_cookie_secure: bool = False  # True in prod (HTTPS via the tunnel)
    web_revoked_path: str = "./data/auth_revoked.json"

    # --- Source corpora paths ---------------------------------------------
    synthetic_corpus_path: str = "~/Source/synthetic-test-corpus"
    # Fictional ground-truth JSON → SQLite fake-case dataset. These are NEVER
    # added to the vector index; they back exact-lookup case.* tools.
    casedata_sqlite_path: str = "./case_chat/casedata/fake_case.sqlite3"
    # Writable store for saved chat conversations (per authenticated subject).
    conversations_sqlite_path: str = "./data/conversations.sqlite3"


settings = Settings()


# ---------------------------------------------------------------------------
# Qdrant collection names. Collections are organized BY CONTENT DOMAIN;
# jurisdiction + doc_type live in the payload and are filtered at query time
# (so adding Missouri later is additive rows, not new collections).
# ---------------------------------------------------------------------------
COLLECTION_LAW = "law"  # statutes + caselaw + constitutions, all jurisdictions
COLLECTION_BEHAVIORAL_PATTERNS = "behavioral_patterns"
COLLECTION_BEHAVIORAL_SOURCES = "behavioral_sources"
COLLECTION_PROFESSIONAL_STANDARDS = "professional_standards"
COLLECTION_SCRIPTURE = "scripture"  # bible_kjv + bible_web (translation in payload)
COLLECTION_SYNTHETIC = "synthetic_corpus"  # raw case documents

ALL_COLLECTIONS = (
    COLLECTION_LAW,
    COLLECTION_BEHAVIORAL_PATTERNS,
    COLLECTION_BEHAVIORAL_SOURCES,
    COLLECTION_PROFESSIONAL_STANDARDS,
    COLLECTION_SCRIPTURE,
    COLLECTION_SYNTHETIC,
)
