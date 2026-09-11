"""Adapter construction from config."""

from __future__ import annotations

from typing import Any, Callable

from ._util import load_object
from .base import ModelAdapter
from .tokenizers import build_tokenizer

_BUILDERS: dict[str, Callable[..., ModelAdapter]] = {}


def register_adapter(name: str, builder: Callable[..., ModelAdapter]) -> None:
    _BUILDERS[name] = builder


def _mock(**kw: Any) -> ModelAdapter:
    from .mock import MockAdapter

    return MockAdapter(**kw)


def _echo(**kw: Any) -> ModelAdapter:
    from .mock import EchoAdapter

    kw.pop("model_id", None)
    return EchoAdapter(**kw)


def _anthropic(**kw: Any) -> ModelAdapter:
    from .anthropic_adapter import AnthropicAdapter

    return AnthropicAdapter(**kw)


def _openai(**kw: Any) -> ModelAdapter:
    from .openai_adapter import OpenAIAdapter

    return OpenAIAdapter(**kw)


def _hf(**kw: Any) -> ModelAdapter:
    from .hf_adapter import HFAdapter

    return HFAdapter(**kw)


def _custom_attn(**kw: Any) -> ModelAdapter:
    from .custom_attn import CustomAttnAdapter

    return CustomAttnAdapter(**kw)


for _name, _builder in [
    ("mock", _mock), ("echo", _echo), ("anthropic", _anthropic),
    ("openai", _openai), ("hf", _hf), ("custom_attn", _custom_attn),
]:
    register_adapter(_name, _builder)


#: Adapters that need a provider-side model identifier.
_NEEDS_MODEL_ID = {"anthropic", "openai", "hf", "custom_attn"}


def build_adapter(cfg: Any) -> ModelAdapter:
    """Instantiate the adapter described by a :class:`~lctx.config.ModelConfig`.

    ``adapter`` is either a registered name or a ``package.module:callable`` path,
    so a third-party backend can be plugged in without modifying lctx-bench.
    """
    kind = cfg.adapter
    builder = _BUILDERS.get(kind)
    if builder is None:
        if ":" not in kind:
            raise KeyError(
                f"unknown adapter {kind!r}; available: {sorted(_BUILDERS)} "
                "(or give a 'package.module:callable' path)"
            )
        builder = load_object(kind)

    if kind in _NEEDS_MODEL_ID and not cfg.model_id:
        raise ValueError(f"model {cfg.name!r} uses adapter {kind!r} and must set model_id")

    # The Anthropic adapter builds its own calibrated tokenizer, because that
    # needs a client which only the adapter has.
    tokenizer = None
    if not (kind == "anthropic" and cfg.tokenizer.kind == "anthropic"):
        tokenizer = build_tokenizer(cfg.tokenizer, cfg.model_id)

    kwargs: dict[str, Any] = {
        "name": cfg.name,
        "tokenizer": tokenizer,
        "context_limit": cfg.context_limit,
        "pricing": (cfg.pricing.input_per_mtok, cfg.pricing.output_per_mtok),
        **cfg.params,
    }
    if cfg.model_id is not None:
        kwargs["model_id"] = cfg.model_id
    return builder(**kwargs)
