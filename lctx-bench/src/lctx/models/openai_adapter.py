"""OpenAI-compatible adapter.

Covers hosted OpenAI models and, via ``base_url``, any server speaking the same
Chat Completions protocol — vLLM, SGLang, llama.cpp, TGI's OpenAI shim, LM Studio.
That is the supported path for evaluating a locally served custom-attention model
at scale: serve it, point ``base_url`` at it, and nothing else changes.
"""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

from .base import FatalError, GenerationResult, ModelAdapter, TransientError, Usage
from .tokenizers import BaseTokenizer, Message, TiktokenTokenizer

_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAIAdapter(ModelAdapter):
    def __init__(
        self,
        name: str,
        model_id: str,
        tokenizer: BaseTokenizer | None = None,
        context_limit: int | None = None,
        pricing: tuple[float, float] = (0.0, 0.0),
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 0,
        require_key: bool = True,
        **params: Any,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the openai package is required: pip install 'lctx-bench[openai]'"
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            if require_key and not base_url:
                raise FatalError(f"{api_key_env} is not set")
            # Local servers usually accept any non-empty key.
            api_key = "not-needed"
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": max_retries}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self.model_id = model_id
        self.extra_body = dict(params.get("extra_body", {}))
        #: Some reasoning models reject an explicit temperature.
        self.send_temperature = bool(params.get("send_temperature", True))

        super().__init__(name, tokenizer or TiktokenTokenizer(), context_limit, pricing)

    async def agenerate(
        self,
        messages: Sequence[Message],
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
        trial: Any | None = None,
    ) -> GenerationResult:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": max_tokens,
            **self.extra_body,
        }
        if self.send_temperature:
            kwargs["temperature"] = temperature
        if stop:
            kwargs["stop"] = list(stop)

        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc
        latency = time.perf_counter() - t0

        choice = resp.choices[0]
        usage = Usage(
            getattr(resp.usage, "prompt_tokens", 0) or 0,
            getattr(resp.usage, "completion_tokens", 0) or 0,
        ) if resp.usage else Usage()
        return GenerationResult(
            text=choice.message.content or "",
            usage=usage,
            latency_s=latency,
            meta={"finish_reason": choice.finish_reason, "response_id": resp.id},
        )

    async def aclose(self) -> None:
        await self._client.close()


def _classify(exc: Exception) -> Exception:
    status = getattr(exc, "status_code", None)
    if status in _TRANSIENT_STATUS:
        return TransientError(f"{type(exc).__name__}: {exc}")
    name = type(exc).__name__
    if name in {"AuthenticationError", "PermissionDeniedError", "BadRequestError",
                "NotFoundError", "UnprocessableEntityError"}:
        return FatalError(f"{name}: {exc}")
    return TransientError(f"{name}: {exc}")
