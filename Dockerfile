# case-chat application image (web app + MCP server + index-build CLIs).
# CPU-only; the GPU services (vLLM, TEI) run as separate official images.
# postgresql-client is included so the build profile can pg_restore the
# knowledgedb dump into the staging Postgres over the network.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-client bash \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install deps first (better layer caching). No dev group, no GPU/torch extras.
COPY pyproject.toml README.md ./
COPY case_chat ./case_chat
RUN uv sync --no-dev

COPY deploy/web_entrypoint.sh deploy/build_indexes.sh ./deploy/
RUN chmod +x deploy/web_entrypoint.sh deploy/build_indexes.sh

EXPOSE 8080
CMD ["bash", "deploy/web_entrypoint.sh"]
