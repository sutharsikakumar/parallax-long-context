"""Offline simulator adapters.

These exist so the full pipeline — packing, running, checkpointing, scoring,
statistics, plotting — is exercisable with no network, no GPU, and no API key.
``MockAdapter`` does not pretend to be a language model: it is a *parameterised
degradation simulator* whose accuracy falls off with context length and varies
with needle depth. That makes it useful for two things and no others:

  1. end-to-end CI of the harness itself;
  2. sanity-checking that the statistics and plots respond correctly to a
     *known* ground-truth degradation curve.

Never report simulator numbers as model results.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import time
from typing import Any, Sequence

from .base import GenerationResult, ModelAdapter, Usage
from .tokenizers import BaseTokenizer, Message, WordTokenizer


def _unit_hash(*parts: Any) -> float:
    """A deterministic uniform draw in [0, 1) from arbitrary keys."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


class MockAdapter(ModelAdapter):
    """Simulates a model whose recall degrades with length and depth.

    Parameters (all from ``ModelConfig.params``):

    ``behavior``
        ``oracle`` (always correct), ``degrading`` (the default simulator),
        ``always_absent``, or ``random``.
    ``base_accuracy``
        Accuracy at negligible context length.
    ``half_length``
        Context length at which the length factor reaches one half.
    ``decay_sharpness``
        Slope of the logistic falloff in log-length. Larger is sharper.
    ``middle_penalty``
        Strength of the "lost in the middle" dip, peaking at depth 0.5.
    ``task_difficulty``
        Optional per-task multipliers, e.g. ``{multi_hop: 0.8}``.
    ``latency``
        Simulated seconds per call (0 by default, so CI stays fast).
    """

    #: Tells the runner it is safe to pass ground truth in the trial context.
    is_simulator = True

    def __init__(
        self,
        name: str,
        tokenizer: BaseTokenizer | None = None,
        context_limit: int | None = None,
        pricing: tuple[float, float] = (0.0, 0.0),
        **params: Any,
    ) -> None:
        super().__init__(name, tokenizer or WordTokenizer(), context_limit, pricing)
        self.behavior: str = params.get("behavior", "degrading")
        self.base_accuracy: float = float(params.get("base_accuracy", 0.98))
        self.half_length: float = float(params.get("half_length", 32_000))
        self.decay_sharpness: float = float(params.get("decay_sharpness", 2.0))
        self.middle_penalty: float = float(params.get("middle_penalty", 0.25))
        self.task_difficulty: dict[str, float] = dict(params.get("task_difficulty", {}))
        self.latency: float = float(params.get("latency", 0.0))

    # -- the simulated degradation curve ----------------------------------

    def p_correct(self, task: str, target_tokens: int, depth: float) -> float:
        if self.behavior == "oracle":
            return 1.0
        if self.behavior in ("always_absent", "random"):
            return 0.0
        length = max(1, int(target_tokens))
        # Logistic falloff in log-length: flat, then a knee near `half_length`.
        z = (math.log(length) - math.log(max(2.0, self.half_length))) * self.decay_sharpness
        length_factor = 1.0 / (1.0 + math.exp(z))
        # "Lost in the middle": worst at depth 0.5, best at either end. With no
        # haystack (target_tokens == 0) there is no middle to be lost in, so the
        # ceiling condition carries no positional penalty.
        depth_factor = 1.0
        if target_tokens > 0:
            depth_factor -= self.middle_penalty * math.sin(math.pi * min(1.0, max(0.0, depth)))
        p = self.base_accuracy * length_factor * depth_factor
        p *= self.task_difficulty.get(task, 1.0)
        return min(1.0, max(0.0, p))

    # -- answer synthesis -------------------------------------------------

    @staticmethod
    def _corrupt(answer: str, rng: random.Random) -> str:
        """A plausible *wrong* answer of the same shape as ``answer``."""
        if answer.strip().upper() == "ABSENT":
            # False positive: confidently invent a value instead of abstaining.
            return "-".join(
                "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(4))
                for _ in range(2)
            )
        if "," in answer:
            items = [x.strip() for x in answer.split(",") if x.strip()]
            keep = [x for x in items if rng.random() > 0.45]
            if keep and rng.random() < 0.3:
                keep.append(MockAdapter._corrupt(keep[0], rng))
            return ", ".join(keep)
        if re.fullmatch(r"-?\d+(\.\d+)?", answer.strip()):
            val = float(answer)
            drift = max(1.0, abs(val) * rng.uniform(0.02, 0.25))
            out = val + rng.choice([-1, 1]) * drift
            return str(int(out) if float(val).is_integer() else round(out, 3))
        chars = list(answer)
        idx = [i for i, c in enumerate(chars) if c.isalnum()]
        for i in rng.sample(idx, min(len(idx), rng.choice([1, 2]))):
            chars[i] = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
        return "".join(chars)

    async def agenerate(
        self,
        messages: Sequence[Message],
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
        trial: Any | None = None,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        if self.latency:
            import asyncio

            await asyncio.sleep(self.latency)

        gold = getattr(trial, "simulator_hint", None)
        if gold is None:
            # Without ground truth there is nothing to simulate; be explicit
            # rather than silently emitting something scorable.
            text = "<answer>ABSENT</answer>"
        elif self.behavior == "always_absent":
            text = "<answer>ABSENT</answer>"
        else:
            key = getattr(trial, "trial_id", "") or str(len(messages))
            rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
            p = self.p_correct(
                getattr(trial, "task", ""),
                getattr(trial, "target_tokens", 0),
                getattr(trial, "depth", 0.5),
            )
            draw = _unit_hash(key, "correct")
            if self.behavior == "random":
                text = f"<answer>{self._corrupt(str(gold), rng)}</answer>"
            elif draw < p:
                text = f"<answer>{gold}</answer>"
            else:
                text = f"<answer>{self._corrupt(str(gold), rng)}</answer>"

        usage = Usage(
            input_tokens=self.tokenizer.count_message_tokens(messages),
            output_tokens=self.tokenizer.count_tokens(text),
        )
        return GenerationResult(
            text=text, usage=usage, latency_s=time.perf_counter() - t0,
            meta={"behavior": self.behavior, "simulated": True},
        )


class EchoAdapter(ModelAdapter):
    """Returns a fixed string. Used by tests that must not depend on scoring."""

    is_simulator = True

    def __init__(self, name: str = "echo", tokenizer: BaseTokenizer | None = None,
                 reply: str = "<answer>ABSENT</answer>", **params: Any) -> None:
        super().__init__(name, tokenizer or WordTokenizer())
        self.reply = params.get("reply", reply)

    async def agenerate(
        self,
        messages: Sequence[Message],
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
        trial: Any | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            text=self.reply,
            usage=Usage(self.tokenizer.count_message_tokens(messages),
                        self.tokenizer.count_tokens(self.reply)),
        )
