"""Aggregation of trial records into cells, curves, and effective-context estimates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from ..runner.records import TrialRecord
from .intervals import Interval, interval_for, wilson_interval

#: Grid coordinates that identify a heatmap cell.
CELL_KEYS = ("model", "task", "condition", "distractor_type", "target_tokens", "depth")
#: Grid coordinates that identify a point on an accuracy-vs-length curve
#: (depths pooled).
CURVE_KEYS = ("model", "task", "condition", "distractor_type", "target_tokens")
#: A series whose effective context length can be estimated.
SERIES_KEYS = ("model", "task", "condition", "distractor_type")


@dataclass
class Summary:
    """Pooled statistics over a set of trials."""

    key: dict[str, Any]
    n: int
    n_errors: int
    accuracy: Interval           # Wilson, over the binarized outcome
    score: Interval              # bootstrap or Wilson, over the continuous score
    mean_actual_tokens: float
    tolerance_violations: int
    mean_latency_s: float
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    #: Raw per-trial outcomes, kept so plots and exports need no second pass.
    raw_correct: list[bool] = field(default_factory=list)
    raw_scores: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = dict(self.key)
        row.update(
            n=self.n,
            n_errors=self.n_errors,
            accuracy=self.accuracy.point,
            accuracy_lo=self.accuracy.low,
            accuracy_hi=self.accuracy.high,
            accuracy_method=self.accuracy.method,
            score=self.score.point,
            score_lo=self.score.low,
            score_hi=self.score.high,
            score_method=self.score.method,
            mean_actual_tokens=self.mean_actual_tokens,
            tolerance_violations=self.tolerance_violations,
            mean_latency_s=self.mean_latency_s,
            total_cost_usd=self.total_cost_usd,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
        )
        row.update(self.extra)
        return row


def _summarize(
    key: dict[str, Any],
    records: Sequence[TrialRecord],
    confidence: float,
    resamples: int,
    seed: int,
) -> Summary:
    usable = [r for r in records if r.ok]
    n_err = len(records) - len(usable)
    corrects = [r.correct for r in usable]
    scores = [r.score for r in usable]
    binary = all(r.binary_scorer for r in usable) if usable else True

    acc = wilson_interval(sum(corrects), len(corrects), confidence)
    sc = interval_for(scores, corrects, binary, confidence, resamples, seed)

    n = len(usable)
    return Summary(
        key=key,
        n=n,
        n_errors=n_err,
        accuracy=acc,
        score=sc,
        mean_actual_tokens=(sum(r.actual_tokens for r in usable) / n) if n else 0.0,
        tolerance_violations=sum(
            1 for r in usable if r.tolerance_checked and not r.within_tolerance
        ),
        mean_latency_s=(sum(r.latency_s for r in usable) / n) if n else 0.0,
        total_cost_usd=sum(r.cost_usd for r in records),
        total_input_tokens=sum(r.input_tokens for r in records),
        total_output_tokens=sum(r.output_tokens for r in records),
        raw_correct=corrects,
        raw_scores=scores,
    )


def group_by(records: Iterable[TrialRecord], keys: Sequence[str]) -> dict[tuple, list[TrialRecord]]:
    out: dict[tuple, list[TrialRecord]] = defaultdict(list)
    for r in records:
        out[tuple(getattr(r, k) for k in keys)].append(r)
    return dict(out)


def aggregate(
    records: Iterable[TrialRecord],
    keys: Sequence[str] = CELL_KEYS,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 12345,
) -> list[Summary]:
    """Pool trials into summaries at the granularity given by ``keys``."""
    groups = group_by(records, keys)
    out = [
        _summarize(dict(zip(keys, k)), v, confidence, resamples, seed)
        for k, v in groups.items()
    ]
    out.sort(key=lambda s: tuple(str(s.key[k]) for k in keys))
    return out


@dataclass
class EffectiveContext:
    """Longest context length at which a series still meets the accuracy bar.

    Two readings are reported, because they answer different questions:

    ``effective_strict``
        The largest tested length L such that *every* tested length up to and
        including L clears the bar. This is the headline number: it refuses to
        credit a model that fails at 32k and then recovers at 64k, which is
        almost always noise or a task artifact.

    ``effective_max``
        The largest tested length that clears the bar at all, ignoring dips
        below it. Reported alongside so that non-monotonic behaviour is visible
        rather than hidden.

    The bar is applied to the *lower* confidence bound, not the point estimate,
    so a cell only counts as passing if the evidence supports it.
    """

    key: dict[str, Any]
    threshold: float
    confidence: float
    effective_strict: int | None
    effective_max: int | None
    lengths: list[int]
    accuracy: list[float]
    accuracy_lo: list[float]
    n_per_length: list[int]
    monotonic: bool

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = dict(self.key)
        row.update(
            threshold=self.threshold,
            confidence=self.confidence,
            effective_context_strict=self.effective_strict,
            effective_context_max=self.effective_max,
            n_lengths_tested=len(self.lengths),
            min_length=min(self.lengths) if self.lengths else None,
            max_length=max(self.lengths) if self.lengths else None,
            monotonic=self.monotonic,
        )
        return row


def effective_context(
    records: Iterable[TrialRecord],
    threshold: float = 0.8,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 12345,
    series_keys: Sequence[str] = SERIES_KEYS,
) -> list[EffectiveContext]:
    """Estimate effective context length per series.

    Only length-varying conditions are considered: ``no_haystack`` sets the
    ceiling and has no length axis, so it is excluded here.
    """
    usable = [r for r in records if r.condition != "no_haystack"]
    out: list[EffectiveContext] = []

    for skey, group in group_by(usable, series_keys).items():
        by_len = group_by(group, ["target_tokens"])
        lengths = sorted(int(k[0]) for k in by_len)
        if not lengths:
            continue

        accs, los, ns = [], [], []
        for L in lengths:
            s = _summarize({}, by_len[(L,)], confidence, resamples, seed)
            accs.append(s.accuracy.point)
            los.append(s.accuracy.low)
            ns.append(s.n)

        passes = [lo >= threshold for lo in los]
        strict: int | None = None
        for L, ok in zip(lengths, passes):
            if not ok:
                break
            strict = L
        maxpass = max((L for L, ok in zip(lengths, passes) if ok), default=None)
        monotonic = all(accs[i] >= accs[i + 1] - 1e-9 for i in range(len(accs) - 1))

        out.append(
            EffectiveContext(
                key=dict(zip(series_keys, skey)),
                threshold=threshold,
                confidence=confidence,
                effective_strict=strict,
                effective_max=maxpass,
                lengths=lengths,
                accuracy=accs,
                accuracy_lo=los,
                n_per_length=ns,
                monotonic=monotonic,
            )
        )
    out.sort(key=lambda e: tuple(str(e.key[k]) for k in series_keys))
    return out


def control_summary(
    records: Iterable[TrialRecord],
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 12345,
) -> list[dict[str, Any]]:
    """Compare each length-varying condition against the no-haystack ceiling.

    Interpretation: the ceiling is what the model scores on the task with no
    filler at all. A low ceiling means the *task* is hard (or the prompt is
    unclear) and any length effect must be read relative to it — degradation is
    only attributable to context length once the ceiling is high.
    """
    rows: list[dict[str, Any]] = []
    by_model_task = group_by(records, ["model", "task"])
    for (model, task), group in by_model_task.items():
        ceiling = [r for r in group if r.condition == "no_haystack"]
        ceil_summary = (
            _summarize({}, ceiling, confidence, resamples, seed) if ceiling else None
        )
        for cond, crecs in group_by(group, ["condition"]).items():
            s = _summarize({}, crecs, confidence, resamples, seed)
            row = {
                "model": model,
                "task": task,
                "condition": cond[0],
                "n": s.n,
                "accuracy": s.accuracy.point,
                "accuracy_lo": s.accuracy.low,
                "accuracy_hi": s.accuracy.high,
                "mean_score": s.score.point,
                "ceiling_accuracy": ceil_summary.accuracy.point if ceil_summary else None,
                "gap_from_ceiling": (
                    ceil_summary.accuracy.point - s.accuracy.point if ceil_summary else None
                ),
            }
            rows.append(row)
    rows.sort(key=lambda r: (r["model"], r["task"], r["condition"]))
    return rows


def to_rows(summaries: Sequence[Summary]) -> list[dict[str, Any]]:
    return [s.as_row() for s in summaries]


def summary_asdict(e: EffectiveContext) -> dict[str, Any]:
    return asdict(e)
