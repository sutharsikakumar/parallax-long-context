"""Confidence intervals.

Core requirement #4: every reported number carries an interval.

Wilson is used for accuracy (a binomial proportion) because the normal
approximation is badly behaved exactly where long-context evaluation lives —
small n and proportions near 0 or 1, where it can produce intervals that extend
past [0, 1] or collapse to zero width at p = 0.

The bootstrap (percentile, BCa-free for simplicity and determinism) is used for
continuous scores such as set-F1, where no closed form applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    n: int
    method: str

    @property
    def width(self) -> float:
        return self.high - self.low

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "point": self.point, "low": self.low, "high": self.high,
            "n": self.n, "method": self.method,
        }


def z_for(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return Interval(float("nan"), 0.0, 1.0, 0, "wilson")
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} out of range for n={n}")
    z = z_for(confidence)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(p, max(0.0, center - half), min(1.0, center + half), n, "wilson")


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 12345,
) -> Interval:
    """Percentile bootstrap CI for the mean of a continuous score.

    Deterministic given ``seed``, so a re-run of the analysis reproduces the
    same interval exactly.
    """
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, "bootstrap")
    point = float(arr.mean())
    if n == 1 or np.allclose(arr, arr[0]):
        # A degenerate sample has no spread to resample; say so rather than
        # reporting a spuriously tight interval.
        return Interval(point, point, point, n, "bootstrap(degenerate)")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return Interval(point, float(lo), float(hi), n, "bootstrap")


def interval_for(
    scores: Sequence[float],
    corrects: Sequence[bool],
    binary: bool,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 12345,
) -> Interval:
    """Pick the right interval: Wilson for binary outcomes, bootstrap otherwise."""
    if binary:
        return wilson_interval(int(sum(bool(c) for c in corrects)), len(corrects), confidence)
    return bootstrap_ci(scores, confidence, resamples, seed)
