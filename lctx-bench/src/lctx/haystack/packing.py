"""Token-exact prompt packing.

Core requirement #3: every cell must land on its target token count, measured
with the model's own tokenizer and *including chat-template overhead*.

The packer places needles at their requested relative depths inside a filler
tape, then solves for the filler length that makes the fully rendered request
hit ``target_tokens``. Because tokenization is not additive across concatenation
boundaries, the length is found by measuring the real rendered messages rather
than by summing parts: the search is a damped fixed-point iteration on

    F  <-  F + (target - measured(F))

which converges in two or three steps since one extra filler token adds almost
exactly one prompt token, with a bisection bracket as a fallback guard.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from ..models.tokenizers import BaseTokenizer, Message
from ..tasks.base import Needle, TaskInstance
from .filler import FillerSource

DOC_OPEN = "<document>\n"
DOC_CLOSE = "\n</document>\n\n"


@dataclass
class PackedPrompt:
    """A rendered prompt plus the measurements needed to trust it."""

    messages: list[Message]
    target_tokens: int
    actual_tokens: int
    filler_tokens: int
    chat_overhead_tokens: int
    #: Absolute token index of each needle's first token in the rendered prompt.
    needle_positions: list[int]
    #: Token length of each needle's own text, so probes can address its span.
    needle_token_lengths: list[int]
    needle_depths_requested: list[float]
    #: Realized depth = needle position / actual prompt length.
    needle_depths_actual: list[float]
    needle_roles: list[str] = field(default_factory=list)
    within_tolerance: bool = True
    #: False for conditions where the target length does not apply (no_haystack).
    tolerance_checked: bool = True
    search_iterations: int = 0
    context_start_token: int = 0

    @property
    def error_frac(self) -> float:
        if not self.target_tokens:
            return 0.0
        return (self.actual_tokens - self.target_tokens) / self.target_tokens


class Packer:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        filler: FillerSource,
        tolerance: float = 0.02,
        max_iters: int = 24,
    ) -> None:
        self.tok = tokenizer
        self.filler = filler
        self.tolerance = tolerance
        self.max_iters = max_iters

    # -- filler tape ------------------------------------------------------

    def _tape(self, n_tokens: int, seed: int, shuffled: bool) -> list[int]:
        """A deterministic tape of at least ``n_tokens`` filler token ids."""
        if n_tokens <= 0:
            return []
        rng = random.Random(seed)
        stream = self.filler.stream(rng, shuffled=shuffled)
        parts: list[str] = []
        # Characters-per-token varies by tokenizer; start pessimistic and extend.
        approx_cpt = 3.0
        needed_chars = int(n_tokens * approx_cpt * 1.3) + 256
        chars = 0
        while chars < needed_chars:
            s = next(stream)
            parts.append(s)
            chars += len(s) + 1
        ids = self.tok.encode(" ".join(parts))
        while len(ids) < n_tokens:
            extra = [next(stream) for _ in range(max(64, (n_tokens - len(ids)) // 4))]
            parts.extend(extra)
            ids = self.tok.encode(" ".join(parts))
        return ids

    # -- assembly ---------------------------------------------------------

    def _build(
        self,
        inst: TaskInstance,
        needles: Sequence[Needle],
        tape: Sequence[int],
        n_filler: int,
        include_document_wrapper: bool = True,
    ) -> tuple[list[Message], str, list[int]]:
        """Assemble messages; also return the user content and needle char offsets."""
        n_filler = max(0, min(int(n_filler), len(tape)))
        cuts = [int(round(n.depth * n_filler)) for n in needles]
        # Needles are sorted by depth, so cuts must be non-decreasing.
        for i in range(1, len(cuts)):
            cuts[i] = max(cuts[i], cuts[i - 1])

        pieces: list[str] = []
        offsets: list[int] = []
        cursor = 0
        running = 0  # character length of `pieces` joined so far

        def emit(text: str) -> None:
            nonlocal running
            pieces.append(text)
            running += len(text)

        for needle, cut in zip(needles, cuts):
            seg = self.tok.decode(tape[cursor:cut]) if cut > cursor else ""
            cursor = cut
            if seg:
                emit(seg.rstrip() + "\n")
            offsets.append(running)
            emit(needle.text + "\n")
        tail = self.tok.decode(tape[cursor:n_filler]) if n_filler > cursor else ""
        if tail:
            emit(tail.lstrip())

        context = "".join(pieces)
        if include_document_wrapper:
            user = DOC_OPEN + context + DOC_CLOSE + inst.question
            base = len(DOC_OPEN)
        else:
            user = context + "\n" + inst.question
            base = 0
        offsets = [o + base for o in offsets]

        messages = [
            {"role": "system", "content": inst.system},
            {"role": "user", "content": user},
        ]
        return messages, user, offsets

    # -- packing ----------------------------------------------------------

    def pack(
        self,
        inst: TaskInstance,
        target_tokens: int,
        condition: str = "standard",
        seed: int = 0,
    ) -> PackedPrompt:
        needles = inst.sorted_needles()
        shuffled = condition == "shuffled_haystack"
        no_haystack = condition == "no_haystack"

        if no_haystack:
            # Ceiling control: the task alone, with no filler at all. Target
            # length does not apply, so tolerance is not checked.
            messages, user, offsets = self._build(inst, needles, [], 0)
            return self._finalize(
                inst, messages, user, offsets, needles,
                target_tokens=target_tokens, n_filler=0, iterations=0,
                tolerance_checked=False,
            )

        tape = self._tape(int(target_tokens * 1.35) + 512, seed, shuffled)

        measured: dict[int, int] = {}

        def count(f: int) -> int:
            f = max(0, min(f, len(tape)))
            if f not in measured:
                msgs, _, _ = self._build(inst, needles, tape, f)
                measured[f] = self.tok.count_message_tokens(msgs)
            return measured[f]

        floor_tokens = count(0)
        best_f = 0
        best_err = abs(floor_tokens - target_tokens)
        iterations = 1

        if floor_tokens < target_tokens:
            lo, hi = 0, len(tape)
            f = min(len(tape), target_tokens - floor_tokens)
            for _ in range(self.max_iters):
                c = count(f)
                iterations += 1
                err = target_tokens - c
                if abs(err) < best_err:
                    best_err, best_f = abs(err), f
                if err == 0:
                    break
                # Maintain a bracket so a pathological tokenizer cannot diverge.
                if c < target_tokens:
                    lo = max(lo, f)
                else:
                    hi = min(hi, f)
                nxt = max(lo, min(hi, f + err))
                if nxt == f or nxt in measured:
                    # Fixed point or revisit: fall back to bisection, then stop.
                    nxt = (lo + hi) // 2
                    if nxt in measured or nxt == f:
                        break
                f = nxt

        messages, user, offsets = self._build(inst, needles, tape, best_f)
        return self._finalize(
            inst, messages, user, offsets, needles,
            target_tokens=target_tokens, n_filler=best_f, iterations=iterations,
            tolerance_checked=True,
        )

    def _finalize(
        self,
        inst: TaskInstance,
        messages: list[Message],
        user: str,
        offsets: list[int],
        needles: Sequence[Needle],
        *,
        target_tokens: int,
        n_filler: int,
        iterations: int,
        tolerance_checked: bool,
    ) -> PackedPrompt:
        actual = self.tok.count_message_tokens(messages)
        overhead = self.tok.chat_overhead(messages)

        # Tokens preceding the user content in the rendered request, so needle
        # positions can be reported on the same axis as the total length.
        prefix = self.tok.count_message_tokens([messages[0]])
        positions = [prefix + self.tok.count_tokens(user[:o]) for o in offsets]

        within = True
        if tolerance_checked and target_tokens > 0:
            within = abs(actual - target_tokens) <= self.tolerance * target_tokens

        return PackedPrompt(
            messages=messages,
            target_tokens=target_tokens,
            actual_tokens=actual,
            filler_tokens=n_filler,
            chat_overhead_tokens=overhead,
            needle_positions=positions,
            needle_token_lengths=[self.tok.count_tokens(n.text) for n in needles],
            needle_depths_requested=[n.depth for n in needles],
            needle_depths_actual=[p / actual if actual else 0.0 for p in positions],
            needle_roles=[n.role for n in needles],
            within_tolerance=within,
            tolerance_checked=tolerance_checked,
            search_iterations=iterations,
            context_start_token=prefix,
        )
