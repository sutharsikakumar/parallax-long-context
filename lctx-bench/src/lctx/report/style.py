"""Shared plotting style.

One palette, applied by role, so every figure in a report reads as one system.

Color assignments follow the job the color does:

* **Model identity** is categorical — a fixed hue order, assigned by series and
  never cycled. A model keeps its color when other models are filtered out.
* **Accuracy in a heatmap** is magnitude — a single-hue sequential ramp, light to
  dark. Not a rainbow: a rainbow implies category boundaries that accuracy does
  not have, and its perceived lightness is non-monotonic, so it misreports order.

The categorical order below was validated (adjacent-pair CVD ΔE ≥ 8, normal-vision
ΔE ≥ 15, chroma and lightness bands) before use. Three of the slots fall below 3:1
contrast on the light surface, so every figure carries a legend, direct labels on
series ends, and an exported CSV table — identity is never conveyed by color alone.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # report generation must work headless

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

#: Categorical hues, assigned to series in this fixed order.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
MAX_SERIES = len(CATEGORICAL)

#: Single-hue sequential ramp for magnitude (accuracy), light to dark.
SEQUENTIAL_STEPS: tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#83817c"
GRID = "#e6e5e1"
MISSING = "#ededea"

#: Line styles used as a secondary (non-color) encoding for distractor type.
DISTRACTOR_DASHES: dict[str, tuple] = {
    "none": (),
    "near_duplicate_key": (5, 2),
    "semantic_lure": (1.5, 1.5),
    "repeated_decoy": (7, 2, 1.5, 2),
}


def accuracy_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list("lctx_accuracy", list(SEQUENTIAL_STEPS))
    cmap.set_bad(MISSING)
    return cmap


def series_color(index: int) -> str:
    """Color for series ``index``, by position in the fixed order.

    Beyond the palette the caller must facet or fold into "other" rather than
    cycling, which would give two series the same identity.
    """
    if index >= MAX_SERIES:
        raise IndexError(
            f"series index {index} exceeds the {MAX_SERIES}-slot categorical palette; "
            "facet the figure instead of reusing a hue"
        )
    return CATEGORICAL[index]


def dashes_for(distractor: str) -> tuple:
    return DISTRACTOR_DASHES.get(distractor, ())


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "font.family": ["DejaVu Sans"],
        }
    )


def strip_axes(ax) -> None:
    """Recessive frame: keep the data, lose the box."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def format_tokens(n: float) -> str:
    """Compact token-count label: 128000 -> '128k'."""
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1000:
        return f"{n / 1000:g}k"
    return f"{n:g}"


def facet_grid(n: int, max_cols: int = 3) -> tuple[int, int]:
    """Rows and columns for ``n`` facets.

    Avoids a final row holding a single panel — four facets read as 2x2, not as
    a row of three with one orphan below and a large empty region beside it.
    """
    if n <= 0:
        return 1, 1
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    # Drop a column to absorb a lone trailing panel, but only when that does not
    # cost an extra row (7 facets stay 3x3 rather than becoming a tall 4x2).
    if cols > 1 and n > cols and n % cols == 1:
        narrower = cols - 1
        if (n + narrower - 1) // narrower == rows:
            cols = narrower
    return rows, cols


def chunk_series(names: Sequence[str], size: int = MAX_SERIES) -> list[list[str]]:
    """Split series into palette-sized groups so hues are never reused in a figure."""
    return [list(names[i : i + size]) for i in range(0, len(names), size)] or [[]]
