"""Filler ("haystack") sources.

Two sources are available, selectable from config:

``corpus``
    Neutral natural-language prose. The bundled source is a bank of mundane
    administrative sentence templates whose slots are filled at random, which
    yields effectively unlimited non-repeating text with no topical overlap with
    any needle — so filler can never accidentally answer, contradict, or cue a
    task. Point ``haystack.corpus_path`` at any plain-text file to use real prose
    instead (recommended when you want realism at the cost of bundling a corpus).

``random_tokens``
    An ablation: pronounceable nonsense words carrying no syntax or semantics.
    Comparing the two isolates how much degradation is driven by *competing
    meaningful content* rather than by raw sequence length.
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

_DATA = Path(__file__).parent / "data" / "neutral_corpus.txt"

_SLOTS: dict[str, list[str]] = {
    "dept": ["records", "facilities", "planning", "operations", "logistics", "compliance",
             "estates", "procurement", "scheduling", "archives"],
    "doc": ["procedures note", "handbook", "summary report", "reference guide",
            "working paper", "briefing note", "standing instruction", "review document"],
    "sys": ["booking", "reporting", "catalogue", "timetable", "directory", "monitoring",
            "submission", "archive"],
    "place": ["north wing", "annexe", "central office", "west building",
              "ground floor", "east block", "depot", "records room"],
    "role": ["coordinator", "duty officer", "site manager", "records officer",
             "team lead", "administrator", "supervisor", "convenor"],
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "month": ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"],
    "dur": ["fortnight", "quarter", "month", "six-week period", "term", "reporting period"],
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class FillerSource(ABC):
    """Produces an unbounded stream of filler sentences."""

    name = "base"

    @abstractmethod
    def stream(self, rng: random.Random, shuffled: bool = False) -> Iterator[str]:
        """Yield filler sentences forever."""


class CorpusFiller(FillerSource):
    name = "corpus"

    def __init__(self, corpus_path: str | Path | None = None) -> None:
        self.corpus_path = Path(corpus_path) if corpus_path else None
        if self.corpus_path is not None:
            text = self.corpus_path.read_text(encoding="utf-8")
            self._units = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
            if not self._units:
                raise ValueError(f"corpus at {self.corpus_path} produced no sentences")
            self._templated = False
            self.name = f"corpus:{self.corpus_path.name}"
        else:
            self._units = [
                ln.strip()
                for ln in _DATA.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
            self._templated = True
            self.name = "corpus:bundled"

    def _fill(self, template: str, rng: random.Random) -> str:
        if not self._templated:
            return template
        out = template
        for key, vocab in _SLOTS.items():
            token = "{" + key + "}"
            while token in out:
                out = out.replace(token, rng.choice(vocab), 1)
        while "{n}" in out:
            out = out.replace("{n}", str(rng.randint(2, 480)), 1)
        while "{pct}" in out:
            out = out.replace("{pct}", str(rng.randint(3, 97)), 1)
        return out

    def stream(self, rng: random.Random, shuffled: bool = False) -> Iterator[str]:
        n = len(self._units)
        order = list(range(n))
        if shuffled:
            # Control condition: destroy the document's running order so that any
            # coherence in the filler cannot be what the model is tracking.
            rng.shuffle(order)
            while True:
                for i in order:
                    yield self._fill(self._units[i], rng)
                rng.shuffle(order)
        else:
            # Sequential, from a seed-dependent offset: consecutive sentences keep
            # their natural thematic flow.
            start = rng.randrange(n)
            i = start
            while True:
                yield self._fill(self._units[i], rng)
                i = (i + 1) % n


class RandomTokenFiller(FillerSource):
    """Ablation filler: nonsense words with no syntax and no meaning."""

    name = "random_tokens"

    def __init__(self, vocab_size: int = 4096, words_per_sentence: tuple[int, int] = (8, 22)) -> None:
        self.vocab_size = max(16, int(vocab_size))
        self.words_per_sentence = words_per_sentence
        # A fixed vocabulary, so the ablation varies only word order.
        vrng = random.Random(0xC0FFEE)
        cons, vows = "bcdfgklmnprstvz", "aeiou"
        self._vocab = [
            "".join(vrng.choice(cons) + vrng.choice(vows) for _ in range(vrng.randint(2, 3)))
            for _ in range(self.vocab_size)
        ]

    def stream(self, rng: random.Random, shuffled: bool = False) -> Iterator[str]:
        lo, hi = self.words_per_sentence
        while True:
            k = rng.randint(lo, hi)
            words = [rng.choice(self._vocab) for _ in range(k)]
            yield " ".join(words).capitalize() + "."


def build_filler(cfg) -> FillerSource:
    """Construct a filler source from a :class:`~lctx.config.HaystackConfig`."""
    if cfg.source == "corpus":
        return CorpusFiller(cfg.corpus_path)
    if cfg.source == "random_tokens":
        return RandomTokenFiller(cfg.random_vocab_size)
    raise ValueError(f"unknown haystack source: {cfg.source!r}")
