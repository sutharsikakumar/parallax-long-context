"""Statistics: intervals, cell aggregation, effective-context estimation."""

from .aggregate import (  # noqa: F401
    CELL_KEYS,
    CURVE_KEYS,
    SERIES_KEYS,
    EffectiveContext,
    Summary,
    aggregate,
    control_summary,
    effective_context,
    group_by,
    to_rows,
)
from .intervals import Interval, bootstrap_ci, interval_for, wilson_interval  # noqa: F401

__all__ = [
    "CELL_KEYS", "CURVE_KEYS", "SERIES_KEYS", "EffectiveContext", "Summary",
    "aggregate", "control_summary", "effective_context", "group_by", "to_rows",
    "Interval", "bootstrap_ci", "interval_for", "wilson_interval",
]
