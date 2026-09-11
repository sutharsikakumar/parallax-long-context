"""Local ``transformers`` adapter, with optional attention capture for probes."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Sequence

from .base import AttentionCapture, GenerationResult, ModelAdapter, Usage
from .tokenizers import BaseTokenizer, HFTokenizer, Message


class HFAdapter(ModelAdapter):
    """A local causal LM loaded with ``transformers``.

    Generation is greedy whenever ``temperature == 0``, so runs are reproducible.
    Because ``transformers`` is synchronous, calls are dispatched to a thread so
    the async runner is not blocked; concurrency should normally be set to 1 for
    a single GPU.

    Probes: :meth:`capture_attention` needs attention weights, which fused
    kernels do not materialise. Load with ``attn_implementation: eager`` to use
    them — see the README section on probes.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        tokenizer: BaseTokenizer | None = None,
        context_limit: int | None = None,
        pricing: tuple[float, float] = (0.0, 0.0),
        device: str | None = None,
        dtype: str = "auto",
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        **params: Any,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers and torch are required: pip install 'lctx-bench[hf]'"
            ) from exc

        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        kwargs: dict[str, Any] = dict(model_kwargs or {})
        kwargs["trust_remote_code"] = trust_remote_code
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        else:
            kwargs["torch_dtype"] = "auto"
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.attn_implementation = attn_implementation

        self.model = self._load_model(AutoModelForCausalLM, model_id, kwargs)
        self.model.eval()
        if self.device != "auto" and "device_map" not in kwargs:
            self.model.to(self.device)

        tok = tokenizer or HFTokenizer(model_id)
        self._hf_tok = getattr(tok, "_tok", None)
        if self._hf_tok is None:  # a non-HF tokenizer was supplied explicitly
            from transformers import AutoTokenizer

            self._hf_tok = AutoTokenizer.from_pretrained(model_id)
        if self._hf_tok.pad_token_id is None:
            self._hf_tok.pad_token = self._hf_tok.eos_token

        if context_limit is None:
            context_limit = getattr(self.model.config, "max_position_embeddings", None)
        super().__init__(name, tok, context_limit, pricing)

    def _load_model(self, auto_cls: Any, model_id: str, kwargs: dict[str, Any]) -> Any:
        """Overridden by :class:`CustomAttnAdapter` to inject a custom mechanism."""
        return auto_cls.from_pretrained(model_id, **kwargs)

    # -- prompt rendering -------------------------------------------------

    def render(self, messages: Sequence[Message]) -> str:
        if getattr(self._hf_tok, "chat_template", None):
            return self._hf_tok.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        # No chat template: fall back to a plain, explicit transcript.
        parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        return "\n\n".join(parts) + "\n\nASSISTANT:"

    def _encode(self, messages: Sequence[Message]) -> Any:
        text = self.render(messages)
        return self._hf_tok(text, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )

    # -- generation -------------------------------------------------------

    def _generate_sync(
        self,
        messages: Sequence[Message],
        max_tokens: int,
        temperature: float,
        stop: Sequence[str] | None,
    ) -> GenerationResult:
        torch = self.torch
        inputs = self._encode(messages)
        n_in = int(inputs["input_ids"].shape[-1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._hf_tok.pad_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)  # greedy: deterministic by default

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - t0

        new_ids = out[0][n_in:]
        text = self._hf_tok.decode(new_ids, skip_special_tokens=True)
        for s in stop or []:
            idx = text.find(s)
            if idx >= 0:
                text = text[:idx]
        return GenerationResult(
            text=text,
            usage=Usage(n_in, int(new_ids.shape[-1])),
            latency_s=latency,
            meta={"device": str(self.model.device), "attn_implementation": self.attn_implementation},
        )

    async def agenerate(
        self,
        messages: Sequence[Message],
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
        trial: Any | None = None,
    ) -> GenerationResult:
        return await asyncio.to_thread(
            self._generate_sync, messages, max_tokens, temperature, stop
        )

    # -- probes -----------------------------------------------------------

    def capture_attention(self, messages: Sequence[Message]) -> AttentionCapture | None:
        """One forward pass with ``output_attentions=True``.

        Returns ``None`` when the loaded attention implementation does not
        materialise weights, which is the normal case for fused kernels.
        """
        torch = self.torch
        inputs = self._encode(messages)
        try:
            with torch.inference_mode():
                out = self.model(**inputs, output_attentions=True, use_cache=False)
        except (ValueError, TypeError, NotImplementedError):
            return None
        attentions = getattr(out, "attentions", None)
        if not attentions or attentions[0] is None:
            return None
        weights = [a[0].to(torch.float32).cpu().numpy() for a in attentions]
        return AttentionCapture(
            weights=weights,
            input_token_count=int(inputs["input_ids"].shape[-1]),
            meta={"attn_implementation": self.attn_implementation, "n_layers": len(weights)},
        )

    def capture_logits(self, messages: Sequence[Message]) -> Any | None:
        torch = self.torch
        inputs = self._encode(messages)
        with torch.inference_mode():
            out = self.model(**inputs, use_cache=False)
        return out.logits[0, -1].to(torch.float32).cpu().numpy()

    async def aclose(self) -> None:
        self.model = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
