"""Anthropic Messages API adapter."""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

from .base import FatalError, GenerationResult, ModelAdapter, TransientError, Usage
from .tokenizers import AnthropicTokenizer, BaseTokenizer, Message

_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class AnthropicAdapter(ModelAdapter):
    """Hosted Anthropic models. Authenticates from ``ANTHROPIC_API_KEY``.

    Token counting defaults to :class:`AnthropicTokenizer`, which pairs a local
    BPE proxy with the provider's authoritative ``count_tokens`` endpoint so the
    packer stays token-exact without spending an API call per search step.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        tokenizer: BaseTokenizer | None = None,
        context_limit: int | None = None,
        pricing: tuple[float, float] = (0.0, 0.0),
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 0,  # the runner owns retries
        **params: Any,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the anthropic package is required: pip install 'lctx-bench[anthropic]'"
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise FatalError(
                f"{api_key_env} is not set; export it or point api_key_env at another variable"
            )
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": max_retries}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self.model_id = model_id
        self.extra_body = dict(params.get("extra_body", {}))

        if tokenizer is None:
            # The tokenizer needs a *sync* client for its count_tokens calls.
            from anthropic import Anthropic

            sync_kwargs = dict(kwargs)
            tokenizer = AnthropicTokenizer(model_id, client=Anthropic(**sync_kwargs))
        super().__init__(name, tokenizer, context_limit, pricing)

    async def agenerate(
        self,
        messages: Sequence[Message],
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
        trial: Any | None = None,
    ) -> GenerationResult:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] != "system"
        ]
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": chat,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self.extra_body,
        }
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = list(stop)

        t0 = time.perf_counter()
        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - classified and re-raised below
            raise _classify(exc) from exc
        latency = time.perf_counter() - t0

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        return GenerationResult(
            text=text,
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
            latency_s=latency,
            meta={"stop_reason": resp.stop_reason, "response_id": resp.id},
        )

    async def aclose(self) -> None:
        await self._client.close()


def _classify(exc: Exception) -> Exception:
    """Map a provider exception onto the runner's retry policy."""
    status = getattr(exc, "status_code", None)
    if status in _TRANSIENT_STATUS:
        return TransientError(f"{type(exc).__name__}: {exc}")
    name = type(exc).__name__
    if name in {"APIConnectionError", "APITimeoutError", "RateLimitError",
                "InternalServerError", "APIStatusError"}:
        return TransientError(f"{name}: {exc}")
    if name in {"AuthenticationError", "PermissionDeniedError", "BadRequestError",
                "NotFoundError", "UnprocessableEntityError"}:
        return FatalError(f"{name}: {exc}")
    return TransientError(f"{name}: {exc}")
