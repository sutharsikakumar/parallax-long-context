"""NIAH-style heatmaps: accuracy over (context length x needle depth)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..stats.aggregate import Summary
from .style import (
    MISSING,
    TEXT_MUTED,
    TEXT_PRIMARY,
    accuracy_cmap,
    apply_style,
    format_tokens,
)


def _grid(summaries: Sequence[Summary]) -> tuple[list[int], list[float], np.ndarray, np.ndarray]:
    lengths = sorted({int(s.key["target_tokens"]) for s in summaries})
    depths = sorted({float(s.key["depth"]) for s in summaries})
    acc = np.full((len(depths), len(lengths)), np.nan)
    ns = np.zeros((len(depths), len(lengths)), dtype=int)
    li = {v: i for i, v in enumerate(lengths)}
    di = {v: i for i, v in enumerate(depths)}
    for s in summaries:
        r, c = di[float(s.key["depth"])], li[int(s.key["target_tokens"])]
        acc[r, c] = s.accuracy.point
        ns[r, c] = s.n
    return lengths, depths, acc, ns


def plot_heatmap(
    summaries: Sequence[Summary],
    title: str,
    out_path: Path,
    subtitle: str = "",
    annotate: bool = True,
) -> Path | None:
    """Render one heatmap. Returns the path, or ``None`` if there is nothing to plot."""
    import matplotlib.pyplot as plt

    if not summaries:
        return None
    apply_style()
    lengths, depths, acc, ns = _grid(summaries)
    if not lengths or not depths:
        return None

    # Size scales with the grid so cells stay roughly square and labels fit.
    fig_w = max(4.2, 1.05 * len(lengths) + 2.0)
    fig_h = max(3.0, 0.62 * len(depths) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    masked = np.ma.masked_invalid(acc)
    mesh = ax.pcolormesh(
        np.arange(len(lengths) + 1),
        np.arange(len(depths) + 1),
        masked,
        cmap=accuracy_cmap(),
        vmin=0.0,
        vmax=1.0,
        edgecolors="#fcfcfb",  # 2px surface gap between cells
        linewidth=2,
    )

    ax.set_xticks(np.arange(len(lengths)) + 0.5)
    ax.set_xticklabels([format_tokens(v) for v in lengths])
    ax.set_yticks(np.arange(len(depths)) + 0.5)
    ax.set_yticklabels([f"{d:.0%}" for d in depths])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("needle depth (position in document)")
    ax.grid(False)
    ax.invert_yaxis()  # depth 0% (document start) at the top, as read
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)

    if annotate:
        for r in range(len(depths)):
            for c in range(len(lengths)):
                v = acc[r, c]
                if np.isnan(v):
                    ax.text(c + 0.5, r + 0.5, "–", ha="center", va="center",
                            color=TEXT_MUTED, fontsize=8)
                    continue
                # Label ink flips on the dark end of the ramp so it stays legible.
                ax.text(c + 0.5, r + 0.5, f"{v:.2f}", ha="center", va="center",
                        color="#ffffff" if v > 0.62 else TEXT_PRIMARY,
                        fontsize=8.5, fontweight="medium")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("accuracy", fontsize=9)
    cbar.outline.set_visible(False)

    nonzero = ns[ns > 0]
    if nonzero.size == 0:
        # Every cell errored; the grid exists but carries no observations.
        n_note = "no successful trials"
    elif int(ns.max()) == int(nonzero.min()):
        n_note = f"n = {int(ns.max())} trials per cell"
    else:
        n_note = f"n = {int(nonzero.min())}–{int(ns.max())} trials per cell"
    full_sub = " · ".join(x for x in (subtitle, n_note) if x)
    ax.set_title(title, loc="left", pad=14)
    ax.text(0, 1.012, full_sub, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color=TEXT_MUTED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_all_heatmaps(
    summaries: Iterable[Summary],
    out_dir: Path,
    group_keys: Sequence[str] = ("model", "task", "condition", "distractor_type"),
) -> list[Path]:
    """One heatmap per series. Conditions with no length axis are skipped."""
    groups: dict[tuple, list[Summary]] = {}
    for s in summaries:
        if s.key.get("condition") == "no_haystack":
            continue  # ceiling control: no length or depth axis to plot
        groups.setdefault(tuple(s.key[k] for k in group_keys), []).append(s)

    paths: list[Path] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        meta = dict(zip(group_keys, key))
        title = f"{meta['task']} — {meta['model']}"
        subtitle = f"condition: {meta['condition']} · distractors: {meta['distractor_type']}"
        slug = "__".join(str(meta[k]).replace("/", "-") for k in group_keys)
        p = plot_heatmap(items, title, out_dir / f"heatmap__{slug}.png", subtitle)
        if p:
            paths.append(p)
    return paths
