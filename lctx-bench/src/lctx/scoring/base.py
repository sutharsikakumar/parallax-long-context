"""Scorer contract and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreResult:
    """Outcome of scoring one trial.

    ``score`` is the continuous quantity (equal to 0/1 for binary scorers) and
    ``correct`` is its binarization. Both are reported: Wilson intervals are
    computed over ``correct``, bootstrap intervals over ``score``.
    """

    score: float
    correct: bool
    parsed: Any = None
    detail: dict[str, Any] = field(default_factory=dict)


class Scorer(ABC):
    name: str = "base"
    #: True when ``score`` only ever takes the values 0.0 and 1.0.
    binary: bool = True

    @abstractmethod
    def score(self, prediction: str, ground_truth: Any, **params: Any) -> ScoreResult:
        """Score a raw model completion against the ground truth."""


_REGISTRY: dict[str, Scorer] = {}


def register_scorer(scorer: Scorer) -> Scorer:
    if scorer.name in _REGISTRY:
        raise ValueError(f"duplicate scorer registration: {scorer.name!r}")
    _REGISTRY[scorer.name] = scorer
    return scorer


def get_scorer(name: str) -> Scorer:
    if name not in _REGISTRY:
        raise KeyError(f"unknown scorer {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_scorers() -> dict[str, Scorer]:
    return dict(_REGISTRY)
