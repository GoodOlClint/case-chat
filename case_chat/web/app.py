"""FastAPI web app — the only externally-reachable process ([ADR 0005]).

Magic-link auth → httpOnly session cookie; SSE streaming of the agentic loop so
the UI shows tool calls live and renders citations. vLLM (or Ollama locally),
Qdrant, the MCP server, and SQLite all stay internal.

Run:  uv run uvicorn case_chat.web.app:app --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from case_chat.chat.mcp_client import MCPToolClient
from case_chat.chat.orchestrator import ChatSession
from case_chat.chat.vllm_client import VLLMClient
from case_chat.config import settings
from case_chat.web.auth import verify_token
from case_chat.web.documents import list_documents, render_document

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"

GATE_HTML = """<!doctype html><html><head><meta charset=utf-8><title>case-chat</title>
<style>body{font-family:system-ui;max-width:34rem;margin:18vh auto;padding:0 1rem;color:#222}
code{background:#f0f0f0;padding:.1rem .3rem;border-radius:4px}</style></head>
<body><h1>case-chat</h1><p>This is a private demo. Access is by invite link only.</p>
<p>Ask the owner for a magic link, then open it to sign in.</p></body></html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vllm = VLLMClient()
    app.state.mcp = MCPToolClient()
    await app.state.mcp.connect()
    app.state.sessions = {}  # conversation_id -> ChatSession (isolated per chat)
    from case_chat.web.conversations import ConversationStore

    app.state.store = ConversationStore()
    logger.info("web app ready (%d MCP tools)", len(app.state.mcp.openai_tools()))
    try:
        yield
    finally:
        await app.state.mcp.aclose()
        await app.state.vllm.aclose()


app = FastAPI(title="case-chat", lifespan=lifespan)


def _subject(request: Request) -> str | None:
    payload = verify_token(request.cookies.get(settings.web_session_cookie))
    return payload.get("sub") if payload else None


def _session_for_conv(request: Request, conv_id: str, subject: str) -> ChatSession:
    """Per-conversation session. Rebuilds context from the store on first use so a
    reloaded chat keeps its running history (survives restarts; isolated per chat)."""
    sessions = request.app.state.sessions
    sess = sessions.get(conv_id)
    if sess is None:
        sess = ChatSession(request.app.state.vllm, request.app.state.mcp)
        saved = request.app.state.store.get(conv_id, subject)
        if saved:
            sess.seed_history([(m["role"], m["content"]) for m in saved["messages"]])
            sess.seed_pool([m.get("citations") for m in saved["messages"]])
        sessions[conv_id] = sess
    return sess


@app.get("/healthz")
async def healthz(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "mcp_tools": len(request.app.state.mcp.openai_tools())})


@app.get("/auth")
async def auth(request: Request, token: str = "") -> Response:
    payload = verify_token(token)
    if not payload:
        return HTMLResponse(GATE_HTML, status_code=401)
    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie(
        settings.web_session_cookie, token,
        max_age=settings.web_session_ttl_days * 86400,
        httponly=True, secure=settings.web_cookie_secure, samesite="lax",
    )
    return resp


@app.get("/")
async def index(request: Request) -> Response:
    if not _subject(request):
        return HTMLResponse(GATE_HTML, status_code=401)
    return FileResponse(STATIC / "index.html")


@app.get("/api/whoami")
async def whoami(request: Request) -> JSONResponse:
    sub = _subject(request)
    if not sub:
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    return JSONResponse({"subject": sub})


def _dataset(request: Request):
    ds = getattr(request.app.state, "dataset", None)
    if ds is None:
        from case_chat.casedata.queries import CaseDataset

        ds = CaseDataset()
        request.app.state.dataset = ds
    return ds


@app.get("/api/casefacts")
async def casefacts(request: Request) -> Response:
    """The structured fake-case record (participants, timeline, facts, flags,
    observations) for the 'Case facts' browse view."""
    if not _subject(request):
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    try:
        return JSONResponse(_dataset(request).case_facts_view())
    except FileNotFoundError:
        return JSONResponse({"error": "case dataset not built"}, status_code=503)


@app.get("/api/documents")
async def documents(request: Request) -> Response:
    """List viewable test-corpus source documents for the sidebar browser."""
    if not _subject(request):
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    return JSONResponse({"documents": list_documents()})


@app.get("/api/passage")
async def passage(request: Request, kind: str = "", key: str = "") -> Response:
    """Assemble the full text of a domain-knowledge source (pattern card / statute)
    for the citation viewer."""
    if not _subject(request):
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    from case_chat.web.passages import get_passage

    result = get_passage(kind, key)
    if result is None:
        return JSONResponse({"error": "passage not found"}, status_code=404)
    return JSONResponse(result)


@app.get("/api/document")
async def document(request: Request, path: str = "") -> Response:
    """Return a raw TEST-CORPUS source document for the viewer. Test corpus
    only — domain knowledge and ground-truth are not served (see documents.py)."""
    if not _subject(request):
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    doc = render_document(path)
    if doc is None:
        return JSONResponse({"error": "not a viewable source document"}, status_code=404)
    return JSONResponse(doc)


# ---- conversations (saved chat history) -----------------------------------
@app.get("/api/conversations")
async def list_conversations(request: Request) -> Response:
    sub = _subject(request)
    if not sub:
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    return JSONResponse({"conversations": request.app.state.store.list(sub)})


@app.post("/api/conversations/new")
async def new_conversation(request: Request) -> Response:
    sub = _subject(request)
    if not sub:
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    conv_id = request.app.state.store.new_id()
    request.app.state.store.create(conv_id, sub)
    return JSONResponse({"id": conv_id})


@app.get("/api/conversations/{conv_id}")
async def get_conversation(request: Request, conv_id: str) -> Response:
    sub = _subject(request)
    if not sub:
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    conv = request.app.state.store.get(conv_id, sub)
    if conv is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(conv)


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(request: Request, conv_id: str) -> Response:
    sub = _subject(request)
    if not sub:
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    request.app.state.sessions.pop(conv_id, None)
    ok = request.app.state.store.delete(conv_id, sub)
    return JSONResponse({"ok": ok}, status_code=200 if ok else 404)


@app.post("/api/chat")
async def chat(request: Request) -> Response:
    sub = _subject(request)
    if not sub:
        return JSONResponse({"error": "unauthenticated"}, status_code=401)
    body = await request.json()
    message = (body.get("message") or "").strip()
    conv_id = body.get("conversation_id") or ""
    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)
    store = request.app.state.store
    if not conv_id or not store.owns(conv_id, sub):
        return JSONResponse({"error": "unknown conversation"}, status_code=400)
    session = _session_for_conv(request, conv_id, sub)
    store.add_turn(conv_id, sub, "user", message)

    async def event_stream():
        answer, citations, thinking_parts = "", [], []
        try:
            async for event in session.stream(message):
                if event["type"] == "answer":
                    answer, citations = event["answer"], event.get("citations") or []
                elif event["type"] == "thinking":
                    thinking_parts.append(event["text"])
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # surface to the client, don't 500 mid-stream
            logger.exception("chat stream failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        try:
            store.add_turn(conv_id, sub, "assistant", answer, citations=citations,
                           thinking="\n".join(thinking_parts) or None)
        except Exception:
            logger.exception("failed to persist assistant turn")
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
