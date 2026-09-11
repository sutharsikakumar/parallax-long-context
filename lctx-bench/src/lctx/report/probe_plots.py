"""Plots for attention probes: sink mass, entropy, and needle mass vs. recall."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .style import (
    TEXT_MUTED,
    apply_style,
    facet_grid,
    format_tokens,
    series_color,
    strip_axes,
)


def load_probes(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _layer_curve(rows: Sequence[dict[str, Any]], field: str) -> tuple[list[int], list[float]]:
    """Mean of ``field`` per layer across the given probe rows."""
    by_layer: dict[int, list[float]] = {}
    for r in rows:
        for m in r["layers"]:
            by_layer.setdefault(int(m["layer"]), []).append(float(m[field]))
    layers = sorted(by_layer)
    return layers, [float(np.mean(by_layer[l])) for l in layers]


def plot_layer_profiles(rows: Sequence[dict[str, Any]], out_path: Path) -> Path | None:
    """Sink mass, entropy, and needle mass by layer, one line per context length."""
    import matplotlib.pyplot as plt

    if not rows:
        return None
    apply_style()
    lengths = sorted({int(r["target_tokens"]) for r in rows})
    fields = [
        ("sink_mass", "attention mass on first token", "attention sink"),
        ("entropy", "normalised entropy (0 = peaked, 1 = diffuse)", "attention entropy"),
        ("needle_mass", "attention mass on needle span", "needle attention"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), squeeze=False)
    for ax, (field, ylabel, title) in zip(axes[0], fields):
        for i, L in enumerate(lengths[: 8]):
            subset = [r for r in rows if int(r["target_tokens"]) == L]
            layers, vals = _layer_curve(subset, field)
            if not layers:
                continue
            ax.plot(layers, vals, color=series_color(i), linewidth=2,
                    marker="o", markersize=4, markeredgecolor="#fcfcfb",
                    markeredgewidth=1.2, label=format_tokens(L))
        ax.set_title(title, loc="left")
        ax.set_xlabel("layer")
        ax.set_ylabel(ylabel, fontsize=8.5)
        if field == "entropy":
            ax.set_ylim(-0.02, 1.02)
        strip_axes(ax)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(labels) >= 2:
        fig.legend(handles, labels, loc="lower center", ncol=min(8, len(labels)),
                   bbox_to_anchor=(0.5, -0.06), title="context length")

    h_in = fig.get_size_inches()[1]
    fig.text(0.006, 1 - 0.12 / h_in, "Attention probes by layer", ha="left", va="top",
             fontsize=13, fontweight="semibold")
    fig.text(0.006, 1 - 0.36 / h_in,
             "means over probed cells · diagnostics, not quality metrics",
             ha="left", va="top", fontsize=8.5, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1 - 0.58 / h_in))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_needle_mass_vs_depth(rows: Sequence[dict[str, Any]], out_path: Path) -> Path | None:
    """Needle attention mass against depth, faceted by context length."""
    import matplotlib.pyplot as plt

    if not rows:
        return None
    apply_style()
    lengths = sorted({int(r["target_tokens"]) for r in rows})
    r_, c_ = facet_grid(len(lengths))
    fig, axes = plt.subplots(r_, c_, figsize=(4.3 * c_, 3.2 * r_), squeeze=False, sharey=True)

    for i, L in enumerate(lengths):
        ax = axes[i // c_][i % c_]
        subset = [r for r in rows if int(r["target_tokens"]) == L]
        by_depth: dict[float, list[float]] = {}
        for r in subset:
            by_depth.setdefault(float(r["depth"]), []).append(float(r["max_needle_mass"]))
        depths = sorted(by_depth)
        ax.plot(depths, [float(np.mean(by_depth[d])) for d in depths],
                color=series_color(0), linewidth=2, marker="o", markersize=5,
                markeredgecolor="#fcfcfb", markeredgewidth=1.4)
        ax.set_title(format_tokens(L), loc="left")
        ax.set_xlabel("needle depth")
        if i % c_ == 0:
            ax.set_ylabel("max needle attention mass")
        ax.set_xlim(-0.03, 1.03)
        strip_axes(ax)

    for j in range(len(lengths), r_ * c_):
        axes[j // c_][j % c_].set_visible(False)

    h_in = fig.get_size_inches()[1]
    fig.text(0.006, 1 - 0.12 / h_in, "Needle attention mass by depth", ha="left", va="top",
             fontsize=13, fontweight="semibold")
    fig.text(0.006, 1 - 0.36 / h_in, "peak over layers · faceted by context length",
             ha="left", va="top", fontsize=8.5, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1 - 0.58 / h_in))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def recall_correlation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Correlate needle attention mass with whether the answer was correct.

    Reported as a point-biserial correlation (Pearson with a binary outcome).
    It is descriptive only: probed cells are a non-random sample of the grid, and
    a correlation here is not evidence that attention mass *causes* recall.
    """
    paired = [
        (float(r["max_needle_mass"]), float(r["correct"]))
        for r in rows
        if r.get("correct") is not None
    ]
    if len(paired) < 3:
        return {"n": len(paired), "correlation": None,
                "note": "too few probed trials with a recall outcome to correlate"}
    x = np.array([p[0] for p in paired])
    y = np.array([p[1] for p in paired])
    if x.std() == 0 or y.std() == 0:
        return {"n": len(paired), "correlation": None,
                "note": "no variance in attention mass or in outcome"}
    return {
        "n": len(paired),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "mean_needle_mass_correct": float(x[y == 1].mean()) if (y == 1).any() else None,
        "mean_needle_mass_incorrect": float(x[y == 0].mean()) if (y == 0).any() else None,
        "note": "point-biserial; descriptive only, probed cells are not a random sample",
    }


def build_probe_report(run_dir: Path, out_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    """Every probe figure plus the correlation summary, for one run."""
    rows = load_probes(Path(run_dir) / "probes.jsonl")
    if not rows:
        return [], {}
    figs: list[Path] = []
    for model in sorted({r["model"] for r in rows}):
        subset = [r for r in rows if r["model"] == model]
        p = plot_layer_profiles(subset, out_dir / f"probe_layers__{model}.png")
        if p:
            figs.append(p)
        p = plot_needle_mass_vs_depth(subset, out_dir / f"probe_needle_depth__{model}.png")
        if p:
            figs.append(p)
    summary = {
        model: recall_correlation([r for r in rows if r["model"] == model])
        for model in sorted({r["model"] for r in rows})
    }
    return figs, summary
