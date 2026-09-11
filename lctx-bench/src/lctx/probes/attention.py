"""Attention probes: sink mass, entropy, and mass landing on the needle.

These are diagnostics, not metrics of quality. They answer "where did attention
go?" so that a recall failure can be told apart from a *retrieval* failure: a
model that puts no mass on the needle never saw it, while one that attends to the
needle and still answers wrong failed downstream of retrieval.

Three quantities per layer, all computed from the query position that produces
the answer (the final position) unless told otherwise:

``sink_mass``
    Attention mass on the first token. Large, content-independent mass on
    position 0 is the well-documented "attention sink"; when it grows with
    context length it is mass that is not being spent on the document.

``entropy``
    Shannon entropy of the attention distribution, normalised by ``log(k)`` so
    layers over different context lengths are comparable. Near 1 means diffuse
    attention spread over everything; near 0 means sharply peaked.

``needle_mass``
    Fraction of attention mass landing inside the needle's token span. This is
    the quantity that should correlate with recall, and the one to plot against
    depth.

All functions take plain numpy arrays shaped ``(heads, queries, keys)``, so they
are testable without a GPU or a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class LayerMetrics:
    layer: int
    sink_mass: float
    entropy: float
    needle_mass: float
    #: Mean distance (in tokens) from query to the attended key, mass-weighted.
    mean_attention_distance: float
    n_heads: int
    per_head_sink_mass: list[float] = field(default_factory=list)
    per_head_needle_mass: list[float] = field(default_factory=list)


@dataclass
class ProbeResult:
    layers: list[LayerMetrics]
    input_token_count: int
    needle_spans: list[tuple[int, int]]
    query_index: int
    meta: dict[str, Any] = field(default_factory=dict)

    def as_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "layer": m.layer,
                "sink_mass": m.sink_mass,
                "entropy": m.entropy,
                "needle_mass": m.needle_mass,
                "mean_attention_distance": m.mean_attention_distance,
                "n_heads": m.n_heads,
                **self.meta,
            }
            for m in self.layers
        ]

    @property
    def mean_needle_mass(self) -> float:
        return float(np.mean([m.needle_mass for m in self.layers])) if self.layers else 0.0

    @property
    def max_needle_mass(self) -> float:
        return float(np.max([m.needle_mass for m in self.layers])) if self.layers else 0.0


def _select_query(weights: np.ndarray, query_index: int) -> np.ndarray:
    """Reduce ``(heads, queries, keys)`` to ``(heads, keys)`` for one query."""
    if weights.ndim == 2:
        return weights
    if weights.ndim != 3:
        raise ValueError(f"expected attention shaped (heads, queries, keys), got {weights.shape}")
    q = weights.shape[1]
    idx = query_index if query_index >= 0 else q + query_index
    if not 0 <= idx < q:
        raise IndexError(f"query index {query_index} out of range for {q} query positions")
    return weights[:, idx, :]


def normalized_entropy(dist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Entropy of each row, divided by ``log(n_keys)`` so it lands in [0, 1]."""
    d = np.clip(np.asarray(dist, dtype=np.float64), eps, None)
    d = d / d.sum(axis=-1, keepdims=True)
    ent = -(d * np.log(d)).sum(axis=-1)
    k = d.shape[-1]
    return ent / np.log(k) if k > 1 else np.zeros_like(ent)


def span_mass(dist: np.ndarray, spans: Sequence[tuple[int, int]]) -> np.ndarray:
    """Attention mass falling inside the given key spans, per row."""
    d = np.asarray(dist, dtype=np.float64)
    total = np.zeros(d.shape[0], dtype=np.float64)
    k = d.shape[-1]
    for start, end in spans:
        lo, hi = max(0, int(start)), min(k, int(end))
        if hi > lo:
            total += d[:, lo:hi].sum(axis=-1)
    return total


def layer_metrics(
    weights: np.ndarray,
    layer: int,
    needle_spans: Sequence[tuple[int, int]],
    query_index: int = -1,
) -> LayerMetrics:
    dist = _select_query(np.asarray(weights, dtype=np.float64), query_index)
    dist = dist / np.clip(dist.sum(axis=-1, keepdims=True), 1e-12, None)

    sink = dist[:, 0]
    ent = normalized_entropy(dist)
    needle = span_mass(dist, needle_spans)

    k = dist.shape[-1]
    q_pos = k - 1 if query_index == -1 else query_index
    positions = np.arange(k, dtype=np.float64)
    distance = (dist * np.abs(q_pos - positions)).sum(axis=-1)

    return LayerMetrics(
        layer=layer,
        sink_mass=float(sink.mean()),
        entropy=float(ent.mean()),
        needle_mass=float(needle.mean()),
        mean_attention_distance=float(distance.mean()),
        n_heads=int(dist.shape[0]),
        per_head_sink_mass=[float(x) for x in sink],
        per_head_needle_mass=[float(x) for x in needle],
    )


def analyze_capture(
    capture: Any,
    needle_spans: Sequence[tuple[int, int]],
    query_index: int = -1,
    meta: dict[str, Any] | None = None,
) -> ProbeResult:
    """Compute per-layer metrics for an :class:`~lctx.models.base.AttentionCapture`."""
    layers = [
        layer_metrics(w, i, needle_spans, query_index)
        for i, w in enumerate(capture.weights)
    ]
    return ProbeResult(
        layers=layers,
        input_token_count=capture.input_token_count,
        needle_spans=[(int(a), int(b)) for a, b in needle_spans],
        query_index=query_index,
        meta={**(capture.meta or {}), **(meta or {})},
    )
