"""Async client for DiffusionGemma served by vLLM (OpenAI-compatible /v1).

Minimal on purpose — just chat completions with tools. Handles the gemma4
reasoning parser: the assistant message may carry ``reasoning_content`` separate
from ``content``; we surface it for display but never feed it back as context.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from case_chat.config import settings

logger = logging.getLogger(__name__)


class VLLMClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self._api_key = api_key or settings.vllm_api_key
        self._model = model or settings.vllm_model
        self._client = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        if self._api_key and self._api_key != "EMPTY":
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """One chat-completions turn. Returns the assistant message dict
        (``content``, optional ``tool_calls``, optional ``reasoning_content``)."""
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        # gemma4 thinking toggle — a vLLM extension Ollama doesn't accept, so it
        # is only sent when targeting real vLLM (box).
        if settings.vllm_send_chat_template_kwargs:
            body["chat_template_kwargs"] = {"enable_thinking": settings.vllm_enable_thinking}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice

        resp = await self._client.post(
            f"{self._base_url}/chat/completions", json=body, headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]

    async def aclose(self) -> None:
        await self._client.aclose()
