"""CSV and JSON exports.

Every figure has a table behind it. Besides being the accessible form of each
chart, the tables are what you cite: a heatmap shows a shape, a CSV shows n and
the interval that shape is resting on.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..runner.records import TrialRecord
from ..stats.aggregate import EffectiveContext, Summary, group_by


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    # Union of keys, first-seen order, so ragged rows still export cleanly.
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})
    return path


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, dict)):
        return json.dumps(v, default=str)
    return v


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def cost_latency_rows(records: Iterable[TrialRecord]) -> list[dict[str, Any]]:
    """Per-model cost and latency, with the tail latencies that matter for planning."""
    rows: list[dict[str, Any]] = []
    for (model,), recs in group_by(records, ["model"]).items():
        ok = [r for r in recs if r.ok]
        lats = sorted(r.latency_s for r in ok)

        def pct(p: float) -> float:
            if not lats:
                return 0.0
            idx = min(len(lats) - 1, max(0, int(round(p * (len(lats) - 1)))))
            return lats[idx]

        rows.append(
            {
                "model": model,
                "n_trials": len(recs),
                "n_errors": len(recs) - len(ok),
                "n_retried": sum(1 for r in recs if r.attempts > 1),
                "total_input_tokens": sum(r.input_tokens for r in recs),
                "total_output_tokens": sum(r.output_tokens for r in recs),
                "total_cost_usd": sum(r.cost_usd for r in recs),
                "cost_per_trial_usd": (sum(r.cost_usd for r in recs) / len(recs)) if recs else 0.0,
                "mean_latency_s": (sum(lats) / len(lats)) if lats else 0.0,
                "p50_latency_s": pct(0.5),
                "p95_latency_s": pct(0.95),
                "max_latency_s": lats[-1] if lats else 0.0,
            }
        )
    rows.sort(key=lambda r: r["model"])
    return rows


def packing_rows(records: Iterable[TrialRecord]) -> list[dict[str, Any]]:
    """Token-exactness audit: did every cell actually land on its target length?

    This table is the evidence for core requirement #3. A run with violations
    here has not measured what it claims to have measured.
    """
    rows: list[dict[str, Any]] = []
    for (model, target), recs in group_by(records, ["model", "target_tokens"]).items():
        checked = [r for r in recs if r.tolerance_checked]
        if not checked:
            continue
        errs = [
            (r.actual_tokens - r.target_tokens) / r.target_tokens
            for r in checked
            if r.target_tokens
        ]
        rows.append(
            {
                "model": model,
                "target_tokens": target,
                "n": len(checked),
                "mean_actual_tokens": sum(r.actual_tokens for r in checked) / len(checked),
                "mean_abs_error_frac": (sum(abs(e) for e in errs) / len(errs)) if errs else 0.0,
                "max_abs_error_frac": max((abs(e) for e in errs), default=0.0),
                "violations": sum(1 for r in checked if not r.within_tolerance),
                "mean_chat_overhead_tokens": sum(
                    r.chat_overhead_tokens for r in checked
                ) / len(checked),
                "mean_pack_iterations": sum(r.pack_iterations for r in checked) / len(checked),
            }
        )
    rows.sort(key=lambda r: (r["model"], r["target_tokens"]))
    return rows


def failure_mode_rows(records: Iterable[TrialRecord]) -> list[dict[str, Any]]:
    """How wrong answers were wrong — abstention vs. copying a decoy vs. fabrication."""
    rows: list[dict[str, Any]] = []
    for (model, task, condition), recs in group_by(
        records, ["model", "task", "condition"]
    ).items():
        ok = [r for r in recs if r.ok]
        wrong = [r for r in ok if not r.correct]
        if not ok:
            continue
        rows.append(
            {
                "model": model,
                "task": task,
                "condition": condition,
                "n": len(ok),
                "n_wrong": len(wrong),
                "wrong_abstained": sum(1 for r in wrong if r.score_detail.get("abstained")),
                "wrong_copied_decoy": sum(1 for r in wrong if r.score_detail.get("copied_decoy")),
                "wrong_fabricated": sum(1 for r in wrong if r.score_detail.get("fabricated")),
                "mean_precision": _mean(
                    [r.score_detail["precision"] for r in ok if "precision" in r.score_detail]
                ),
                "mean_recall": _mean(
                    [r.score_detail["recall"] for r in ok if "recall" in r.score_detail]
                ),
            }
        )
    rows.sort(key=lambda r: (r["model"], r["task"], r["condition"]))
    return rows


def _mean(xs: Sequence[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def effective_context_rows(effs: Sequence[EffectiveContext]) -> list[dict[str, Any]]:
    return [e.as_row() for e in effs]


def summary_payload(
    cells: Sequence[Summary],
    curves: Sequence[Summary],
    effs: Sequence[EffectiveContext],
    controls: Sequence[dict[str, Any]],
    records: Sequence[TrialRecord],
) -> dict[str, Any]:
    """The machine-readable form of the whole report."""
    return {
        "effective_context": [asdict(e) for e in effs],
        "cells": [s.as_row() for s in cells],
        "curves": [s.as_row() for s in curves],
        "controls": list(controls),
        "cost_latency": cost_latency_rows(records),
        "packing_audit": packing_rows(records),
        "failure_modes": failure_mode_rows(records),
        "totals": {
            "n_trials": len(records),
            "n_errors": sum(1 for r in records if not r.ok),
            "total_cost_usd": sum(r.cost_usd for r in records),
            "tolerance_violations": sum(
                1 for r in records if r.tolerance_checked and not r.within_tolerance
            ),
        },
    }
