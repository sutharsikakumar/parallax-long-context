"""Probe math, checked against distributions whose answers are known exactly."""

from __future__ import annotations

import numpy as np
import pytest

from lctx.probes.attention import (
    analyze_capture,
    layer_metrics,
    normalized_entropy,
    span_mass,
)
from lctx.models.base import AttentionCapture


def _uniform(heads=4, keys=64) -> np.ndarray:
    return np.ones((heads, 1, keys)) / keys


def _peaked(at: int, heads=4, keys=64, mass=0.9) -> np.ndarray:
    w = np.full((heads, 1, keys), (1 - mass) / (keys - 1))
    w[:, 0, at] = mass
    return w


def test_uniform_attention_has_maximal_entropy():
    assert normalized_entropy(_uniform()[:, 0, :]).mean() == pytest.approx(1.0)


def test_peaked_attention_has_low_entropy():
    assert normalized_entropy(_peaked(20)[:, 0, :]).mean() < 0.25


def test_entropy_is_normalised_across_context_lengths():
    """Comparing layers across lengths requires entropy to be length-invariant."""
    a = normalized_entropy(_uniform(keys=64)[:, 0, :]).mean()
    b = normalized_entropy(_uniform(keys=4096)[:, 0, :]).mean()
    assert a == pytest.approx(b) == pytest.approx(1.0)


def test_sink_mass_reads_position_zero():
    m = layer_metrics(_peaked(0), 0, [(10, 14)])
    assert m.sink_mass == pytest.approx(0.9, abs=1e-6)
    assert m.needle_mass < 0.01


def test_needle_mass_sums_the_span():
    w = np.zeros((2, 1, 32))
    w[:, 0, 5:9] = 0.2      # 0.8 inside the span
    w[:, 0, 20] = 0.2       # 0.2 outside
    m = layer_metrics(w, 0, [(5, 9)])
    assert m.needle_mass == pytest.approx(0.8)


def test_span_mass_clips_out_of_range_spans():
    """A span past the end of the sequence must be clipped, not crash."""
    d = np.ones((2, 16)) / 16
    assert span_mass(d, [(12, 999)])[0] == pytest.approx(4 / 16)
    assert span_mass(d, [(-5, 2)])[0] == pytest.approx(2 / 16)
    assert span_mass(d, [(10, 10)])[0] == pytest.approx(0.0)


def test_multiple_spans_are_additive():
    d = np.ones((1, 20)) / 20
    assert span_mass(d, [(0, 2), (10, 12)])[0] == pytest.approx(4 / 20)


def test_attention_distance_matches_the_peak_offset():
    keys = 64
    m = layer_metrics(_peaked(10, keys=keys, mass=1.0), 0, [(0, 1)])
    assert m.mean_attention_distance == pytest.approx(keys - 1 - 10)


def test_metrics_normalise_unnormalised_rows():
    """Some implementations hand back unnormalised weights; results must not shift."""
    w = _peaked(20) * 7.5
    m = layer_metrics(w, 0, [(20, 21)])
    assert m.needle_mass == pytest.approx(0.9, abs=1e-6)


def test_query_selection_picks_the_answering_position():
    w = np.zeros((1, 3, 8))
    w[0, 0, 0] = 1.0     # first query attends to the sink
    w[0, 2, 5] = 1.0     # last query attends to the needle
    assert layer_metrics(w, 0, [(5, 6)], query_index=-1).needle_mass == pytest.approx(1.0)
    assert layer_metrics(w, 0, [(5, 6)], query_index=0).needle_mass == pytest.approx(0.0)


def test_bad_shapes_are_rejected_clearly():
    with pytest.raises(ValueError, match="heads, queries, keys"):
        layer_metrics(np.zeros((2, 2, 2, 2)), 0, [(0, 1)])
    with pytest.raises(IndexError):
        layer_metrics(np.zeros((1, 2, 8)), 0, [(0, 1)], query_index=9)


def test_analyze_capture_walks_every_layer():
    capture = AttentionCapture(
        weights=[_peaked(20), _peaked(0), _uniform()],
        input_token_count=64,
        meta={"n_layers": 3},
    )
    result = analyze_capture(capture, [(20, 21)])
    assert [m.layer for m in result.layers] == [0, 1, 2]
    assert result.max_needle_mass == pytest.approx(0.9, abs=1e-6)
    assert result.layers[1].sink_mass > result.layers[0].sink_mass
    assert len(result.as_rows()) == 3


def test_recall_correlation_is_descriptive_and_guards_small_samples():
    from lctx.report.probe_plots import recall_correlation

    assert recall_correlation([])["correlation"] is None
    rows = [{"max_needle_mass": 0.9, "correct": 1.0}] * 5
    assert recall_correlation(rows)["correlation"] is None  # no variance

    mixed = (
        [{"max_needle_mass": 0.8, "correct": 1.0} for _ in range(5)]
        + [{"max_needle_mass": 0.05, "correct": 0.0} for _ in range(5)]
    )
    out = recall_correlation(mixed)
    assert out["correlation"] == pytest.approx(1.0)
    assert out["n"] == 10


def test_probes_are_optional_for_adapters_that_cannot_capture():
    """A hosted model must report "no probes", never fabricate them."""
    from lctx.models.mock import MockAdapter

    adapter = MockAdapter("sim")
    assert adapter.capture_attention([{"role": "user", "content": "hi"}]) is None
    assert adapter.supports_probes is False


def test_facet_grid_lays_panels_out_without_orphans():
    """Four panels must read as 2x2, not a row of three with one stranded below."""
    from lctx.report.style import facet_grid

    assert facet_grid(4) == (2, 2)
    # ...but absorbing an orphan must never cost an extra row.
    assert facet_grid(7) == (3, 3)
    for n in range(1, 13):
        rows, cols = facet_grid(n)
        assert rows * cols >= n
        assert cols <= 3
        # No layout should waste a whole column's worth of cells.
        assert rows * cols - n < cols
