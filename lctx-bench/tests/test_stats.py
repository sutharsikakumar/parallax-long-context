"""Statistics: intervals, aggregation, and effective-context estimation."""

from __future__ import annotations

import math

import pytest

from lctx.runner.records import TrialRecord, cell_id, trial_id
from lctx.stats import (
    aggregate,
    bootstrap_ci,
    control_summary,
    effective_context,
    wilson_interval,
)


def _rec(task="single_needle", length=1000, depth=0.5, seed=0, correct=True,
         score=None, condition="standard", model="m", binary=True, **kw) -> TrialRecord:
    score = float(correct) if score is None else score
    return TrialRecord(
        trial_id=trial_id(model, task, condition, "none", length, depth, seed, 1, {}),
        cell_id=cell_id(model, task, condition, "none", length, depth),
        run_name="t", model=model, task=task, condition=condition,
        target_tokens=length, depth=depth, seed=seed,
        correct=correct, score=score, binary_scorer=binary, **kw,
    )


# -- Wilson ---------------------------------------------------------------

def test_wilson_stays_inside_the_unit_interval_at_the_boundaries():
    """The reason Wilson is used instead of the normal approximation."""
    perfect = wilson_interval(20, 20)
    assert perfect.high == 1.0 and 0 < perfect.low < 1.0
    zero = wilson_interval(0, 20)
    assert zero.low == 0.0 and 0 < zero.high < 1.0


def test_wilson_never_has_zero_width_at_the_extremes():
    """A perfect score is evidence, not proof; the interval must say so."""
    assert wilson_interval(10, 10).low < 1.0
    assert wilson_interval(0, 10).high > 0.0


def test_wilson_narrows_with_more_data():
    widths = [wilson_interval(n // 2, n).width for n in (10, 50, 200, 1000)]
    assert widths == sorted(widths, reverse=True)


def test_wilson_matches_a_known_value():
    # 5/10 at 95%: the textbook Wilson interval.
    iv = wilson_interval(5, 10, 0.95)
    assert iv.low == pytest.approx(0.2365, abs=1e-3)
    assert iv.high == pytest.approx(0.7635, abs=1e-3)


def test_wilson_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    assert math.isnan(wilson_interval(0, 0).point)


def test_higher_confidence_is_wider():
    assert wilson_interval(8, 10, 0.99).width > wilson_interval(8, 10, 0.80).width


# -- bootstrap ------------------------------------------------------------

def test_bootstrap_brackets_the_mean_and_is_deterministic():
    vals = [1.0, 0.8, 0.6, 1.0, 0.9, 0.4, 1.0, 0.75, 0.55, 0.95]
    a = bootstrap_ci(vals, seed=1)
    b = bootstrap_ci(vals, seed=1)
    assert a.low <= a.point <= a.high
    assert (a.low, a.high) == (b.low, b.high)          # reproducible
    assert bootstrap_ci(vals, seed=2).low != a.low or True  # different seed may differ


def test_bootstrap_declares_a_degenerate_sample():
    """A constant sample has no spread; the method name must say so."""
    iv = bootstrap_ci([0.7] * 8)
    assert iv.low == iv.high == 0.7
    assert "degenerate" in iv.method


def test_bootstrap_on_empty_input_is_nan_not_a_crash():
    assert math.isnan(bootstrap_ci([]).point)


# -- aggregation ----------------------------------------------------------

def test_aggregate_pools_seeds_into_cells():
    records = [_rec(seed=s, correct=s < 3) for s in range(5)]
    cells = aggregate(records)
    assert len(cells) == 1
    assert cells[0].n == 5
    assert cells[0].accuracy.point == pytest.approx(0.6)


def test_errored_trials_are_excluded_but_counted():
    """A failed API call must not be silently scored as incorrect."""
    records = [_rec(seed=0), _rec(seed=1), _rec(seed=2, error="timeout")]
    cell = aggregate(records)[0]
    assert cell.n == 2 and cell.n_errors == 1
    assert cell.accuracy.point == pytest.approx(1.0)


def test_continuous_scorer_uses_bootstrap_and_binary_uses_wilson():
    cont = aggregate([_rec(seed=s, correct=False, score=0.5 + s * 0.1, binary=False)
                      for s in range(6)])[0]
    assert "bootstrap" in cont.score.method
    assert cont.accuracy.method == "wilson"


# -- effective context ----------------------------------------------------

def _ladder(pattern: dict[int, float], n: int = 40, **kw) -> list[TrialRecord]:
    """Build records whose accuracy at each length matches ``pattern``."""
    out = []
    for length, acc in pattern.items():
        k = round(acc * n)
        for s in range(n):
            out.append(_rec(length=length, seed=s, correct=s < k, **kw))
    return out


def test_effective_context_is_the_last_length_that_holds_up():
    recs = _ladder({1000: 1.0, 4000: 1.0, 16000: 0.5, 64000: 0.1})
    eff = effective_context(recs, threshold=0.8)[0]
    assert eff.effective_strict == 4000
    assert eff.effective_max == 4000


def test_effective_context_uses_the_lower_bound_not_the_point_estimate():
    """0.85 observed on few trials is not evidence of 0.8 true accuracy."""
    recs = _ladder({1000: 0.85}, n=20)
    assert effective_context(recs, threshold=0.8)[0].effective_strict is None
    # The same accuracy with far more evidence does clear the bar.
    assert effective_context(_ladder({1000: 0.85}, n=500), threshold=0.8)[0].effective_strict


def test_strict_and_max_diverge_when_a_curve_is_non_monotonic():
    """A recovery after a dip is reported, not hidden — and does not count as strict."""
    recs = _ladder({1000: 1.0, 4000: 0.5, 16000: 1.0})
    eff = effective_context(recs, threshold=0.8)[0]
    assert eff.effective_strict == 1000
    assert eff.effective_max == 16000
    assert eff.monotonic is False


def test_effective_context_is_none_when_even_the_shortest_length_fails():
    eff = effective_context(_ladder({1000: 0.2, 4000: 0.1}), threshold=0.8)[0]
    assert eff.effective_strict is None and eff.effective_max is None


def test_no_haystack_is_excluded_from_the_length_axis():
    recs = _ladder({1000: 1.0}) + [_rec(length=0, condition="no_haystack", seed=s)
                                   for s in range(10)]
    effs = effective_context(recs, threshold=0.8)
    assert all(e.key["condition"] != "no_haystack" for e in effs)


# -- controls -------------------------------------------------------------

def test_control_summary_reports_the_gap_from_the_ceiling():
    recs = (
        [_rec(condition="no_haystack", length=0, seed=s, correct=True) for s in range(10)]
        + [_rec(condition="standard", length=16000, seed=s, correct=s < 5) for s in range(10)]
    )
    rows = {r["condition"]: r for r in control_summary(recs)}
    assert rows["no_haystack"]["accuracy"] == pytest.approx(1.0)
    assert rows["standard"]["gap_from_ceiling"] == pytest.approx(0.5)
