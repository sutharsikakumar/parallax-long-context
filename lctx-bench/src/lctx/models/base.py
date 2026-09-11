"""The ModelAdapter interface.

Every backend — hosted API, local transformers model, or a model with a custom
attention implementation swapped in — implements this one interface. The runner,
the packer, and the probes know nothing else about a model.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from .tokenizers import BaseTokenizer, Message


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def cost(self, input_per_mtok: float, output_per_mtok: float) -> float:
        return (
            self.input_tokens * input_per_mtok + self.output_tokens * output_per_mtok
        ) / 1_000_000


@dataclass
class GenerationResult:
    text: str
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    #: Provider-specific extras (stop reason, response id, ...) for the run log.
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionCapture:
    """Per-layer attention for the probe suite.

    ``weights`` is a list of arrays shaped ``(heads, query_positions, key_positions)``,
    one entry per layer. Adapters that cannot expose attention return ``None``
    from :meth:`ModelAdapter.capture_attention` — probes are always optional.
    """

    weights: list[Any]
    input_token_count: int
    meta: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Base class for all model backends."""

    def __init__(
        self,
        name: str,
        tokenizer: BaseTokenizer,
        context_limit: int | None = None,
        pricing: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.name = name
        self.tokenizer = tokenizer
        self.context_limit = context_limit
        self.input_per_mtok, self.output_per_mtok = pricing

    # -- generation -------------------------------------------------------

    #: Simulator adapters receive ground truth in the trial context. Network and
    #: local-weights adapters must leave this False so they can never see it.
    is_simulator: bool = False

    @abstractmethod
    async def agenerate(
        self,
        messages: Sequence[Message],
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
        trial: Any | None = None,
    ) -> GenerationResult:
        """Generate a completion. Must be deterministic when ``temperature == 0``.

        ``trial`` carries read-only metadata about the grid cell (ids, task name,
        target length, depth). Real backends may use it for request tagging or
        logging and must otherwise ignore it.
        """

    def generate(
        self,
        messages: Sequence[Message],
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
        trial: Any | None = None,
    ) -> GenerationResult:
        """Blocking convenience wrapper around :meth:`agenerate`."""
        return asyncio.run(self.agenerate(messages, max_tokens, temperature, stop, trial))

    # -- token accounting -------------------------------------------------

    def count_message_tokens(self, messages: Sequence[Message]) -> int:
        return self.tokenizer.count_message_tokens(messages)

    def cost(self, usage: Usage) -> float:
        return usage.cost(self.input_per_mtok, self.output_per_mtok)

    # -- optional probe hooks ---------------------------------------------

    def capture_attention(self, messages: Sequence[Message]) -> AttentionCapture | None:
        """Return attention weights for a forward pass, or ``None`` if unsupported."""
        return None

    def capture_logits(self, messages: Sequence[Message]) -> Any | None:
        """Return next-token logits for a forward pass, or ``None`` if unsupported."""
        return None

    @property
    def supports_probes(self) -> bool:
        return type(self).capture_attention is not ModelAdapter.capture_attention

    # -- lifecycle --------------------------------------------------------

    async def aclose(self) -> None:
        """Release network or GPU resources. Safe to call more than once."""

    def describe(self) -> dict[str, Any]:
        """Metadata recorded in the run manifest."""
        return {
            "name": self.name,
            "adapter": type(self).__name__,
            "tokenizer": self.tokenizer.name,
            "context_limit": self.context_limit,
            "supports_probes": self.supports_probes,
            "is_simulator": self.is_simulator,
            "pricing": {
                "input_per_mtok": self.input_per_mtok,
                "output_per_mtok": self.output_per_mtok,
            },
        }


class TransientError(RuntimeError):
    """Raised by adapters for errors the runner should retry (429, 5xx, timeouts)."""


class FatalError(RuntimeError):
    """Raised by adapters for errors retrying cannot fix (auth, bad request)."""
