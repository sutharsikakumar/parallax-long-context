"""Adapter for models whose attention mechanism has been replaced.

This is the entry point for evaluating a *custom attention implementation* —
linear attention, local-linear / covariance-corrected attention, sparse or
sliding-window variants, a hand-written Triton or Helion kernel — under the same
grid, packing, scoring, and statistics as any hosted model.

There are three supported routes, in increasing order of invasiveness.

Route 1 — a registered attention function (preferred for kernel swaps)
----------------------------------------------------------------------
``transformers`` dispatches attention through a registry, so a custom kernel can
be published under a name and selected per-model::

    # myproj/attn.py
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def my_attention(module, query, key, value, attention_mask=None, scaling=None,
                     dropout=0.0, **kwargs):
        # ... your kernel ...
        return attn_output, attn_weights  # attn_weights may be None

    ALL_ATTENTION_FUNCTIONS["my_attn"] = my_attention

    def register() -> None:
        '''Called by lctx-bench before the model is loaded.'''

Config::

    - name: my-model-custom-attn
      adapter: custom_attn
      model_id: org/my-model
      params:
        register_hook: myproj.attn:register
        attn_implementation: my_attn

Route 2 — a patch function (module surgery)
-------------------------------------------
When the mechanism changes module structure rather than just the kernel, supply
a callable that receives the loaded model and returns the patched model::

    # myproj/attn.py
    def patch_model(model, **kwargs):
        for layer in model.model.layers:
            layer.self_attn = MyAttention.from_existing(layer.self_attn, **kwargs)
        return model

Config::

    params:
      attention_patch: myproj.attn:patch_model
      patch_kwargs: {window: 512}

Route 3 — a complete custom loader
----------------------------------
When the model is not an ``AutoModelForCausalLM`` at all, supply a loader that
returns a ``transformers``-compatible module exposing ``.generate`` and
``.config``::

    params:
      model_loader: myproj.loading:load_my_model
      loader_kwargs: {checkpoint: /path/to/ckpt.pt}

A worked, runnable stub lives in ``examples/custom_attention_stub.py``.

Serving note: for long-context sweeps at scale, serving the patched model behind
an OpenAI-compatible endpoint (vLLM/SGLang) and using ``adapter: openai`` with
``base_url`` is usually faster than in-process ``generate``. This adapter is the
right choice when you need probes, or when the mechanism has no serving path yet.
"""

from __future__ import annotations

from typing import Any, Sequence

from ._util import load_object
from .base import AttentionCapture
from .hf_adapter import HFAdapter
from .tokenizers import BaseTokenizer, Message


class CustomAttnAdapter(HFAdapter):
    """A local model with a user-supplied attention mechanism."""

    def __init__(
        self,
        name: str,
        model_id: str,
        tokenizer: BaseTokenizer | None = None,
        context_limit: int | None = None,
        pricing: tuple[float, float] = (0.0, 0.0),
        register_hook: str | None = None,
        attention_patch: str | None = None,
        patch_kwargs: dict[str, Any] | None = None,
        model_loader: str | None = None,
        loader_kwargs: dict[str, Any] | None = None,
        **params: Any,
    ) -> None:
        self._register_hook = register_hook
        self._attention_patch = attention_patch
        self._patch_kwargs = dict(patch_kwargs or {})
        self._model_loader = model_loader
        self._loader_kwargs = dict(loader_kwargs or {})
        self._patch_applied = False

        # Route 1: register the kernel *before* the model is constructed, so that
        # `attn_implementation` can name it.
        if register_hook:
            load_object(register_hook)()

        super().__init__(
            name=name,
            model_id=model_id,
            tokenizer=tokenizer,
            context_limit=context_limit,
            pricing=pricing,
            **params,
        )

    def _load_model(self, auto_cls: Any, model_id: str, kwargs: dict[str, Any]) -> Any:
        # Route 3: a complete custom loader replaces from_pretrained.
        if self._model_loader:
            loader = load_object(self._model_loader)
            model = loader(model_id, **{**kwargs, **self._loader_kwargs})
        else:
            model = auto_cls.from_pretrained(model_id, **kwargs)

        # Route 2: module surgery on the loaded model.
        if self._attention_patch:
            patch = load_object(self._attention_patch)
            patched = patch(model, **self._patch_kwargs)
            if patched is None:
                raise ValueError(
                    f"attention_patch {self._attention_patch!r} returned None; "
                    "it must return the patched model"
                )
            model = patched
            self._patch_applied = True
        return model

    def capture_attention(self, messages: Sequence[Message]) -> AttentionCapture | None:
        """Custom mechanisms frequently have no dense attention matrix to report.

        A mechanism that *can* expose one should either return ``attn_weights``
        from its attention function (route 1) or implement
        ``model.capture_attention(input_ids)`` returning a list of per-layer
        arrays shaped ``(heads, queries, keys)``.
        """
        hook = getattr(self.model, "capture_attention", None)
        if callable(hook):
            inputs = self._encode(messages)
            weights = hook(inputs["input_ids"])
            if weights is None:
                return None
            return AttentionCapture(
                weights=list(weights),
                input_token_count=int(inputs["input_ids"].shape[-1]),
                meta={"source": "model.capture_attention", "custom": True},
            )
        return super().capture_attention(messages)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "custom_attention": {
                    "register_hook": self._register_hook,
                    "attn_implementation": self.attn_implementation,
                    "attention_patch": self._attention_patch,
                    "patch_applied": self._patch_applied,
                    "model_loader": self._model_loader,
                }
            }
        )
        return info
