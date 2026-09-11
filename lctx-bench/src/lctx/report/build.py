"""Report orchestration: turn a run directory into figures and tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..config import ExperimentConfig, StatsConfig
from ..runner.records import TrialRecord, read_jsonl
from ..stats.aggregate import (
    CELL_KEYS,
    CURVE_KEYS,
    aggregate,
    control_summary,
    effective_context,
)
from .curves import plot_curve_set
from .heatmap import plot_all_heatmaps
from .probe_plots import build_probe_report
from .tables import (
    cost_latency_rows,
    effective_context_rows,
    failure_mode_rows,
    packing_rows,
    summary_payload,
    write_csv,
    write_json,
)

REPORT_DIR = "report"


@dataclass
class ReportResult:
    report_dir: Path
    figures: list[Path] = field(default_factory=list)
    tables: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


def _integrity_warnings(records: Sequence[TrialRecord], threshold: float) -> list[str]:
    """Checks worth surfacing before anyone reads the numbers."""
    warns: list[str] = []
    checked = [r for r in records if r.tolerance_checked]
    viol = [r for r in checked if not r.within_tolerance]
    if viol:
        warns.append(
            f"{len(viol)}/{len(checked)} prompts missed the token-count tolerance; "
            "length comparisons across those cells are not exact"
        )
    errs = [r for r in records if not r.ok]
    if errs:
        warns.append(
            f"{len(errs)}/{len(records)} trials errored and are excluded from all statistics"
        )

    # A low ceiling means the task, not the context length, is the binding constraint.
    for row in control_summary(records):
        if row["condition"] == "no_haystack" and row["n"] and row["accuracy"] < threshold:
            warns.append(
                f"ceiling for {row['model']}/{row['task']} is only {row['accuracy']:.0%} "
                "with no haystack at all — degradation at length cannot be attributed "
                "to context length until the ceiling is high"
            )

    # Wilson lower bounds are wide at small n. Effective context is computed on
    # trials pooled over depth, so the power question is about *that* n, not the
    # per-cell n behind a heatmap square.
    from ..stats.intervals import wilson_interval

    # no_haystack has no length axis and is excluded from effective context,
    # so it must not drive the power check either.
    pooled = aggregate([r for r in records if r.condition != "no_haystack"], CURVE_KEYS)
    if pooled:
        min_n = min(s.n for s in pooled if s.n)
        if min_n and wilson_interval(min_n, min_n, 0.95).low < threshold:
            need = 1
            while wilson_interval(need, need, 0.95).low < threshold and need < 10_000:
                need += 1
            warns.append(
                f"underpowered: the smallest length has n={min_n} trials pooled over "
                f"depth, and even a perfect result at that n has a 95% Wilson lower "
                f"bound below the {threshold:.2f} threshold. No length can be reported "
                f"as effective below n={need}; raise run.n_trials accordingly"
            )
    return warns


def build_report(
    run_dir: Path | str,
    stats_cfg: StatsConfig | None = None,
    out_dir: Path | str | None = None,
) -> ReportResult:
    """Generate every figure and table for a completed (or partial) run."""
    run_dir = Path(run_dir)
    records = read_jsonl(run_dir / "trials.jsonl")
    if not records:
        raise FileNotFoundError(f"no trials found in {run_dir / 'trials.jsonl'}")

    if stats_cfg is None:
        cfg_path = run_dir / "config.yaml"
        stats_cfg = (
            ExperimentConfig.from_yaml(cfg_path).stats if cfg_path.exists() else StatsConfig()
        )

    out = Path(out_dir) if out_dir else run_dir / REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)

    kw = dict(
        confidence=stats_cfg.ci,
        resamples=stats_cfg.bootstrap_resamples,
        seed=stats_cfg.bootstrap_seed,
    )
    cells = aggregate(records, CELL_KEYS, **kw)
    curves = aggregate(records, CURVE_KEYS, **kw)
    effs = effective_context(
        records, threshold=stats_cfg.effective_context_threshold, **kw
    )
    controls = control_summary(records, **kw)

    figures: list[Path] = []
    figures += plot_all_heatmaps(cells, out)
    figures += plot_curve_set(curves, out, threshold=stats_cfg.effective_context_threshold)

    tables = [
        write_csv(out / "summary_effective_context.csv", effective_context_rows(effs)),
        write_csv(out / "cells.csv", [s.as_row() for s in cells]),
        write_csv(out / "curves.csv", [s.as_row() for s in curves]),
        write_csv(out / "controls.csv", list(controls)),
        write_csv(out / "cost_latency.csv", cost_latency_rows(records)),
        write_csv(out / "packing_audit.csv", packing_rows(records)),
        write_csv(out / "failure_modes.csv", failure_mode_rows(records)),
    ]

    # Probes are optional; absent probes.jsonl simply contributes nothing.
    probe_figs, probe_summary = build_probe_report(run_dir, out)
    figures += probe_figs

    payload = summary_payload(cells, curves, effs, controls, records)
    if probe_summary:
        payload["probes"] = probe_summary
    warnings = _integrity_warnings(records, stats_cfg.effective_context_threshold)
    payload["warnings"] = warnings
    tables.append(write_json(out / "summary.json", payload))

    return ReportResult(
        report_dir=out, figures=figures, tables=tables, warnings=warnings, payload=payload
    )
