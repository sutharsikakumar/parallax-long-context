"""Tokenizer abstractions.

Context length in lctx-bench is *always* measured in tokens under the model's
own tokenizer, including chat-template overhead. Everything that needs to count
tokens goes through this interface, so the packer never has to know which
provider it is talking to.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

Message = dict[str, str]


class BaseTokenizer(ABC):
    """Encode/decode/count interface.

    ``count_message_tokens`` is the authoritative measurement: it must include
    whatever the provider's chat template adds around the raw content.
    """

    #: Human-readable identifier, recorded in run manifests.
    name: str = "base"
    #: Extra tokens the chat template adds per message.
    per_message_overhead: int = 4
    #: Extra tokens the chat template adds once per request.
    per_request_overhead: int = 3

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str: ...

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def count_message_tokens(self, messages: Sequence[Message]) -> int:
        """Token count of a full rendered request, template overhead included."""
        total = self.per_request_overhead
        for m in messages:
            total += self.per_message_overhead + self.count_tokens(m["content"])
        return total

    def chat_overhead(self, messages: Sequence[Message]) -> int:
        """Tokens attributable to the chat template rather than to content."""
        content = sum(self.count_tokens(m["content"]) for m in messages)
        return self.count_message_tokens(messages) - content

    def truncate_to(self, text: str, n_tokens: int) -> str:
        """Decode the first ``n_tokens`` tokens of ``text``.

        Round-tripping is not exact for every BPE, so callers that need an exact
        count must re-measure the result rather than trusting ``n_tokens``.
        """
        if n_tokens <= 0:
            return ""
        return self.decode(self.encode(text)[:n_tokens])


class WordTokenizer(BaseTokenizer):
    """Dependency-free, fully reversible tokenizer.

    A token is a run of non-whitespace plus its trailing whitespace, which makes
    ``decode(encode(x)) == x`` exactly. It exists so the entire harness — packing,
    tolerance checks, tests, CI — runs offline with no model files and no network.
    """

    name = "word"
    _SPLIT = re.compile(r"\S+\s*|\s+")

    def __init__(self, per_message_overhead: int = 4, per_request_overhead: int = 3) -> None:
        self._vocab: dict[str, int] = {}
        self._inv: list[str] = []
        self.per_message_overhead = per_message_overhead
        self.per_request_overhead = per_request_overhead

    def _id(self, piece: str) -> int:
        got = self._vocab.get(piece)
        if got is None:
            got = len(self._inv)
            self._vocab[piece] = got
            self._inv.append(piece)
        return got

    def encode(self, text: str) -> list[int]:
        return [self._id(p) for p in self._SPLIT.findall(text)]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self._inv[i] for i in ids)

    def count_tokens(self, text: str) -> int:
        # Avoid growing the vocab just to count.
        return sum(1 for _ in self._SPLIT.finditer(text))


class TiktokenTokenizer(BaseTokenizer):
    """OpenAI BPE tokenizer."""

    def __init__(
        self,
        encoding: str = "cl100k_base",
        per_message_overhead: int = 4,
        per_request_overhead: int = 3,
    ) -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "tiktoken is required for tokenizer kind 'tiktoken': pip install 'lctx-bench[tiktoken]'"
            ) from exc
        self._enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"
        self.per_message_overhead = per_message_overhead
        self.per_request_overhead = per_request_overhead

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, disallowed_special=())

    def decode(self, ids: Sequence[int]) -> str:
        return self._enc.decode(list(ids))


class HFTokenizer(BaseTokenizer):
    """``transformers`` tokenizer; uses the model's real chat template when present."""

    def __init__(self, name_or_path: str, per_message_overhead: int = 4,
                 per_request_overhead: int = 3, **kwargs: Any) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for tokenizer kind 'hf': pip install 'lctx-bench[hf]'"
            ) from exc
        self._tok = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
        self.name = f"hf:{name_or_path}"
        self.per_message_overhead = per_message_overhead
        self.per_request_overhead = per_request_overhead

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=True)

    def count_message_tokens(self, messages: Sequence[Message]) -> int:
        # Prefer the model's own chat template: it is the ground truth for overhead.
        if getattr(self._tok, "chat_template", None):
            rendered = self._tok.apply_chat_template(
                list(messages), tokenize=True, add_generation_prompt=True
            )
            return len(rendered)
        return super().count_message_tokens(messages)


class AnthropicTokenizer(BaseTokenizer):
    """Authoritative counts via the ``count_tokens`` API, with a calibrated local proxy.

    Counting exactly on every step of the packer's binary search would cost ~15
    API round-trips per prompt. Instead the search runs against a local BPE proxy
    scaled by an empirically calibrated ratio, and only the final candidates are
    verified — and refined — against the real API.
    """

    def __init__(
        self,
        model_id: str,
        client: Any | None = None,
        proxy_encoding: str = "cl100k_base",
        exact_refine_budget: int = 4,
        per_message_overhead: int = 4,
        per_request_overhead: int = 3,
    ) -> None:
        self._proxy = TiktokenTokenizer(proxy_encoding, per_message_overhead, per_request_overhead)
        self._model_id = model_id
        self._client = client
        self.name = f"anthropic:{model_id}"
        self.exact_refine_budget = exact_refine_budget
        self.per_message_overhead = per_message_overhead
        self.per_request_overhead = per_request_overhead
        #: Multiplicative correction from proxy counts to true counts.
        self.calibration: float = 1.0
        self._calibrated = False

    def encode(self, text: str) -> list[int]:
        return self._proxy.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self._proxy.decode(ids)

    def count_tokens(self, text: str) -> int:
        return int(round(self._proxy.count_tokens(text) * self.calibration))

    def count_exact(self, messages: Sequence[Message]) -> int:
        """One authoritative round-trip to the provider's token counter."""
        if self._client is None:
            raise RuntimeError("AnthropicTokenizer needs a client for exact counting")
        system = [m["content"] for m in messages if m["role"] == "system"]
        chat = [m for m in messages if m["role"] != "system"]
        kwargs: dict[str, Any] = {"model": self._model_id, "messages": chat}
        if system:
            kwargs["system"] = "\n\n".join(system)
        return int(self._client.messages.count_tokens(**kwargs).input_tokens)

    def calibrate(self, messages: Sequence[Message]) -> float:
        """Fit the proxy->true ratio once, on a representative prompt."""
        proxy = self._proxy.count_message_tokens(messages)
        if proxy <= 0:
            return self.calibration
        self.calibration = self.count_exact(messages) / proxy
        self._calibrated = True
        return self.calibration

    def count_message_tokens(self, messages: Sequence[Message]) -> int:
        return int(round(self._proxy.count_message_tokens(messages) * self.calibration))


def build_tokenizer(cfg: Any, model_id: str | None = None, client: Any | None = None) -> BaseTokenizer:
    """Construct a tokenizer from a :class:`~lctx.config.TokenizerConfig`."""
    kind = cfg.kind
    over = cfg.per_message_overhead
    kw = {"per_message_overhead": over} if over else {}
    if kind == "word":
        return WordTokenizer(**kw)
    if kind == "tiktoken":
        return TiktokenTokenizer(cfg.encoding, **kw)
    if kind == "hf":
        target = cfg.name_or_path or model_id
        if not target:
            raise ValueError("tokenizer kind 'hf' requires tokenizer.name_or_path or model_id")
        return HFTokenizer(target, **kw)
    if kind == "anthropic":
        target = cfg.name_or_path or model_id
        if not target:
            raise ValueError("tokenizer kind 'anthropic' requires a model_id")
        return AnthropicTokenizer(
            target,
            client=client,
            proxy_encoding=cfg.encoding,
            exact_refine_budget=cfg.exact_refine_budget,
            **kw,
        )
    raise ValueError(f"unknown tokenizer kind: {kind!r}")
