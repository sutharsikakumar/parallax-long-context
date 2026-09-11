"""Model adapters: one interface over hosted APIs, local weights, and custom attention."""

from .base import (  # noqa: F401
    AttentionCapture,
    FatalError,
    GenerationResult,
    ModelAdapter,
    TransientError,
    Usage,
)
from .registry import build_adapter, register_adapter  # noqa: F401
from .tokenizers import BaseTokenizer, build_tokenizer  # noqa: F401

__all__ = [
    "AttentionCapture", "FatalError", "GenerationResult", "ModelAdapter",
    "TransientError", "Usage", "build_adapter", "register_adapter",
    "BaseTokenizer", "build_tokenizer",
]
