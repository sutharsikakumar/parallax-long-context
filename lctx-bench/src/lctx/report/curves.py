"""Accuracy-vs-length curves with confidence bands, faceted by task and distractor.

One rule governs the layout: **a line may only connect points that measure the
same thing.** ``standard`` measures retrieval accuracy, ``needle_absent`` measures
correct abstention, ``no_haystack`` has no length axis at all. Drawing them as one
series would produce a curve whose every segment changes meaning, so conditions
are separated into their own series and panels rather than pooled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..stats.aggregate import Summary
from .style import (
    MAX_SERIES,
    TEXT_MUTED,
    apply_style,
    dashes_for,
    facet_grid,
    format_tokens,
    series_color,
    strip_axes,
)

#: What the accuracy axis means in each condition — the panels are not comparable
#: without it.
CONDITION_MEANING = {
    "standard": "retrieval accuracy",
    "shuffled_haystack": "retrieval accuracy, filler order shuffled",
    "needle_absent": "correct abstention rate",
    "no_haystack": "ceiling (no filler)",
}


def _points(items: Sequence[Summary]) -> tuple[list[int], list[float], list[float], list[float]]:
    pts = sorted(items, key=lambda s: int(s.key["target_tokens"]))
    return (
        [int(s.key["target_tokens"]) for s in pts],
        [s.accuracy.point for s in pts],
        [s.accuracy.low for s in pts],
        [s.accuracy.high for s in pts],
    )


def _draw_panel(
    ax,
    grouped: dict[tuple[str, str], list[Summary]],
    color_of: dict[str, int],
    threshold: float | None,
    direct_labels: bool,
) -> None:
    if threshold is not None:
        ax.axhline(threshold, color=TEXT_MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)

    for (model, distractor), items in sorted(grouped.items()):
        xs, ys, lo, hi = _points(items)
        if not xs:
            continue
        if len(set(xs)) != len(xs):
            raise ValueError(
                f"series ({model}, {distractor}) has duplicate lengths {xs}; "
                "a series must be split further before plotting"
            )
        color = series_color(color_of[model])

        # The band is the interval, not decoration: it is what makes a dip
        # readable as a real effect rather than noise.
        ax.fill_between(xs, lo, hi, color=color, alpha=0.14, linewidth=0, zorder=2)
        line, = ax.plot(
            xs, ys, color=color, linewidth=2, zorder=3, marker="o", markersize=4.5,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5,
        )
        dashes = dashes_for(distractor)
        if dashes:
            line.set_dashes(dashes)

        if direct_labels:
            label = model if distractor == "none" else f"{model} ({distractor})"
            ax.annotate(
                label, xy=(xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=8, va="center", fontweight="medium",
            )


def plot_curves(
    summaries: Sequence[Summary],
    out_path: Path,
    facet_by: str = "task",
    conditions: Sequence[str] | None = ("standard",),
    threshold: float | None = 0.8,
    title: str = "Accuracy vs. context length",
    subtitle: str = "",
) -> Path | None:
    """Faceted curves: one panel per value of ``facet_by``, one line per model.

    Distractor type is a dash pattern rather than a second hue, so a model keeps
    one identity across every condition shown.
    """
    import matplotlib.pyplot as plt

    usable = [s for s in summaries if s.n > 0 and s.key.get("condition") != "no_haystack"]
    if conditions is not None:
        usable = [s for s in usable if str(s.key.get("condition")) in set(conditions)]
    if not usable:
        return None
    apply_style()

    facets = sorted({str(s.key[facet_by]) for s in usable})
    models = sorted({str(s.key["model"]) for s in usable})
    if len(models) > MAX_SERIES:
        # Never cycle hues: keep one palette-sized group and say so on the figure.
        models = models[:MAX_SERIES]
        usable = [s for s in usable if str(s.key["model"]) in models]
        subtitle = (subtitle + " · " if subtitle else "") + (
            f"first {MAX_SERIES} models only — facet the rest"
        )
    color_of = {m: i for i, m in enumerate(models)}

    rows, cols = facet_grid(len(facets))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False, sharey=True
    )
    n_series = len({(str(s.key["model"]), str(s.key["distractor_type"])) for s in usable})

    for i, facet in enumerate(facets):
        ax = axes[i // cols][i % cols]
        items = [s for s in usable if str(s.key[facet_by]) == facet]
        grouped: dict[tuple[str, str], list[Summary]] = {}
        for s in items:
            grouped.setdefault(
                (str(s.key["model"]), str(s.key["distractor_type"])), []
            ).append(s)

        _draw_panel(ax, grouped, color_of, threshold, direct_labels=n_series <= 4)
        panel_title = facet
        if facet_by == "condition":
            panel_title = f"{facet} — {CONDITION_MEANING.get(facet, facet)}"
        ax.set_title(panel_title, loc="left")
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "", "50%", "", "100%"])

        lengths = sorted({int(s.key["target_tokens"]) for s in items})
        if len(lengths) > 1 and max(lengths) / max(1, min(lengths)) >= 8:
            ax.set_xscale("log")  # lengths are a geometric ladder
        ax.set_xticks(lengths)
        ax.set_xticklabels([format_tokens(v) for v in lengths])
        ax.minorticks_off()
        if i // cols == rows - 1:
            ax.set_xlabel("context length (tokens)")
        if i % cols == 0:
            ax.set_ylabel("accuracy")
        strip_axes(ax)

    for j in range(len(facets), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    # A legend is always present for two or more series: identity never rests on
    # color alone.
    if n_series >= 2:
        handles = [
            plt.Line2D([], [], color=series_color(color_of[m]), linewidth=2, label=m)
            for m in models
        ]
        distractors = sorted({str(s.key["distractor_type"]) for s in usable})
        if len(distractors) > 1:
            handles += [
                plt.Line2D([], [], color=TEXT_MUTED, linewidth=1.6,
                           dashes=dashes_for(d) or (1, 0), label=f"distractors: {d}")
                for d in distractors
            ]
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(4, len(handles)), bbox_to_anchor=(0.5, -0.04))

    bits = [subtitle]
    if conditions is not None and facet_by != "condition":
        bits.append("condition: " + ", ".join(conditions))
    if threshold is not None:
        bits.append(f"dashed line: {threshold:.0%} threshold")
    bits.append("bands: Wilson intervals")

    # Header offsets are expressed in inches and converted, so the title block
    # keeps the same spacing whatever the facet count does to figure height.
    h_in = fig.get_size_inches()[1]
    fig.text(0.008, 1 - 0.12 / h_in, title, ha="left", va="top",
             fontsize=13, fontweight="semibold")
    fig.text(0.008, 1 - 0.36 / h_in, " · ".join(b for b in bits if b), ha="left",
             va="top", fontsize=8.5, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1 - 0.58 / h_in))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_curve_set(
    summaries: Sequence[Summary], out_dir: Path, threshold: float = 0.8
) -> list[Path]:
    """The standard curve figures for a run."""
    out: list[Path] = []
    present = {str(s.key["condition"]) for s in summaries}
    length_conditions = sorted(present - {"no_haystack"})
    tasks = sorted({str(s.key["task"]) for s in summaries})
    distractors = {str(s.key["distractor_type"]) for s in summaries}

    # 1. The headline: retrieval accuracy per task, standard condition.
    main_cond = "standard" if "standard" in present else (
        length_conditions[0] if length_conditions else None
    )
    if main_cond:
        p = plot_curves(
            summaries, out_dir / "curves_by_task.png", facet_by="task",
            conditions=(main_cond,), threshold=threshold,
            title="Accuracy vs. context length, by task",
        )
        if p:
            out.append(p)

    # 2. Faceted by distractor type, one figure per task.
    if len(distractors) > 1 and main_cond:
        for task in tasks:
            items = [s for s in summaries if str(s.key["task"]) == task]
            p = plot_curves(
                items, out_dir / f"curves_by_distractor__{task}.png",
                facet_by="distractor_type", conditions=(main_cond,), threshold=threshold,
                title=f"Accuracy vs. context length — {task}",
                subtitle="faceted by distractor type",
            )
            if p:
                out.append(p)

    # 3. Control conditions, faceted so that each panel's axis has one meaning.
    if len(length_conditions) > 1:
        for task in tasks:
            items = [s for s in summaries if str(s.key["task"]) == task]
            p = plot_curves(
                items, out_dir / f"curves_by_condition__{task}.png",
                facet_by="condition", conditions=tuple(length_conditions),
                threshold=threshold,
                title=f"Conditions — {task}",
                subtitle="panels measure different quantities and are not directly comparable",
            )
            if p:
                out.append(p)
    return out
