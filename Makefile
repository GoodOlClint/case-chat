# case-chat — common tasks. See docs/RUNBOOK.md for the full story.
.DEFAULT_GOAL := help
PY := uv run

.PHONY: help up down restore reembed reembed-full reembed-bibles index-synthetic \
        build-casedata build-all mcp web link revoke test ollama-pull

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

ollama-pull: ## Pull the local models (embedder + gemma4 chat)
	ollama pull qwen3-embedding:4b
	ollama pull gemma4:26b-mlx   # closest DiffusionGemma proxy (gemma4:e4b-mlx = lighter fallback)

up: ## Start Qdrant + staging Postgres
	docker compose up -d

down: ## Stop infra (keeps volumes)
	docker compose down

restore: ## Stage the knowledgedb dump into the throwaway Postgres
	bash scripts/restore_knowledgedb.sh

reembed: ## Re-embed domain corpus into Qdrant, skipping bibles (fast local default)
	$(PY) python -m case_chat.knowledgedb.reembed --recreate --skip-bibles

reembed-full: ## Re-embed the entire domain corpus including bibles
	$(PY) python -m case_chat.knowledgedb.reembed --recreate

reembed-bibles: ## Re-embed ONLY the bibles into the scripture collection
	$(PY) python -m case_chat.knowledgedb.reembed --tables bible_kjv,bible_web --recreate

index-synthetic: ## Build the synthetic raw-doc index
	$(PY) python -m case_chat.synthetic.indexer --recreate

build-casedata: ## Build the SQLite fake-case dataset from ground-truth
	$(PY) python -m case_chat.casedata.dataset

build-all: reembed index-synthetic build-casedata ## Build all indexes (no bibles)

mcp: ## Run the stdio MCP server (for manual inspection)
	$(PY) python -m case_chat.mcp_server.server

web: ## Run the web app
	$(PY) uvicorn case_chat.web.app:app --host $${CASECHAT_WEB_HOST:-127.0.0.1} --port $${CASECHAT_WEB_PORT:-8080}

link: ## Issue a magic-link  (make link SUBJECT="Friend in TN")
	$(PY) python -m case_chat.web.auth issue --subject "$(SUBJECT)"

revoke: ## Revoke a subject's links  (make revoke SUBJECT="Friend in TN")
	$(PY) python -m case_chat.web.auth revoke --subject "$(SUBJECT)"

test: ## Run the test suite
	$(PY) pytest -q
