"""Optional attention and logit probes."""

from .attention import (  # noqa: F401
    LayerMetrics,
    ProbeResult,
    analyze_capture,
    layer_metrics,
    normalized_entropy,
    span_mass,
)
from .run import locate_needle_spans, probe_plan, run_probes, sample_plans  # noqa: F401

__all__ = [
    "LayerMetrics", "ProbeResult", "analyze_capture", "layer_metrics",
    "normalized_entropy", "span_mass",
    "locate_needle_spans", "probe_plan", "run_probes", "sample_plans",
]
