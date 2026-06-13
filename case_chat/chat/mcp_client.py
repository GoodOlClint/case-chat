"""MCP client wrapper: spawns the stdio retrieval server and exposes its tools.

The orchestrator is an MCP *client*. This wrapper starts the case-chat MCP
server as a subprocess over stdio, lists its tools, converts them to OpenAI
tool schemas (to advertise to vLLM), and dispatches tool calls.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPToolClient:
    def __init__(self, command: str | None = None, args: list[str] | None = None) -> None:
        # Default: run our own server module with the current interpreter.
        self._params = StdioServerParameters(
            command=command or sys.executable,
            args=args or ["-m", "case_chat.mcp_server.server"],
        )
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._openai_tools: list[dict[str, Any]] = []
        # One stdio session is shared across user sessions; serialize calls.
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(self._params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        listed = await self._session.list_tools()
        self._openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (t.description or "").strip(),
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                },
            }
            for t in listed.tools
        ]
        logger.info("MCP connected; %d tools", len(self._openai_tools))

    def openai_tools(self) -> list[dict[str, Any]]:
        return self._openai_tools

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        if self._session is None:
            raise RuntimeError("MCPToolClient not connected")
        async with self._lock:
            result = await self._session.call_tool(name, arguments)
        parts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        text = "\n".join(parts) if parts else ""
        if result.isError:
            return f"TOOL ERROR: {text}"
        return text

    async def aclose(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None
