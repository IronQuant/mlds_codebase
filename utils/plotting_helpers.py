"""Shared matplotlib helpers for the thesis notebooks (styling + common charts)."""

import numpy as np
import matplotlib.pyplot as plt


# Okabe-Ito colour-blind-safe palette (https://jfly.uni-koeln.de/color/), grey as the 8th.
OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#56B4E9",
    "#CC79A7",
    "#F0E442",
    "#999999",
]

# Fixed sentiment colours, consistent across every panel: blue=dovish, vermillion=hawkish, grey=neutral.
LABEL_C = {"dovish": "#0072B2", "hawkish": "#D55E00", "neutral": "#999999"}


def set_plot_style():
    """Apply the project's shared matplotlib rcParams (font sizes, bold titles, dpi).

    Call once near the top of a plotting section; mutates the global rcParams.

    Returns:
        None.
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "figure.dpi": 110,
        }
    )


def color_map(order, palette=OKABE_ITO):
    """Map an ordered list of categories to palette colours, cycling if needed.

    Args:
        order: Category values, in the order they should take palette colours.
        palette: Hex colours to draw from (defaults to OKABE_ITO).

    Returns:
        A dict {category: hex colour}.
    """
    return {c: palette[i % len(palette)] for i, c in enumerate(order)}


def despine(ax, keep=("left", "bottom")):
    """Hide the Axes border lines (spines) not named in `keep`.

    Args:
        ax: The matplotlib Axes to clean up.
        keep: Which spines stay visible (subset of top/right/left/bottom).

    Returns:
        None; mutates `ax` in place.
    """
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(sp in keep)


def fmt(n):
    """Format a number compactly: 1.2M, 34k, or the integer itself.

    Args:
        n: A non-negative number.

    Returns:
        A short human-readable string (millions / thousands / units).
    """
    return (
        f"{n / 1e6:.1f}M"
        if n >= 1e6
        else (f"{n / 1e3:.0f}k" if n >= 1e3 else f"{int(n)}")
    )


def stacked_bar_over_time(
    ax,
    df,
    group_col,
    *,
    year_col="year",
    value_col=None,
    color=None,
    order=None,
    scale=1.0,
    normalize=False,
    width=0.9,
    label=True,
):
    """Draw a year-axis stacked bar chart, one stacked segment per `group_col` value.

    Rows with a missing `year_col` are dropped (they have no place on the axis), and
    gaps between years are filled with zeros so the bars stay evenly spaced.

    Args:
        ax: Matplotlib Axes to draw on.
        df: Long-form data, one row per observation.
        group_col: Column whose values become the stacked segments (e.g. "source").
        year_col: Integer-or-coercible year column for the x-axis.
        value_col: Column summed per (year, group); None counts rows instead.
        color: Mapping group value -> colour; defaults to color_map(columns).
        order: Segment order and inclusion filter; defaults to the groups present,
            largest total first.
        scale: Divide bar heights by this (e.g. 1e6 to plot millions). Ignored
            when `normalize` is True.
        normalize: If True, plot each year's segments as a percent of that year's total.
        width: Bar width.
        label: True -> label each segment "<group>  (n=<total>)"; a callable
            label(group) -> str for a custom label; falsy -> no labels.

    Returns:
        The pivoted (year x group) DataFrame of unscaled totals.
    """
    d = df.dropna(subset=[year_col]).copy()
    d[year_col] = d[year_col].astype(int)
    g = d.groupby([year_col, group_col])
    piv = (g.size() if value_col is None else g[value_col].sum()).unstack(fill_value=0)

    if order is None:
        order = piv.sum().sort_values(ascending=False).index.tolist()
    piv = piv.reindex(columns=[c for c in order if c in piv.columns], fill_value=0)
    piv = piv.reindex(
        np.arange(piv.index.min(), piv.index.max() + 1), fill_value=0
    )  # fill year gaps

    if color is None:
        color = color_map(list(piv.columns))
    heights = (
        (piv.div(piv.sum(axis=1).replace(0, np.nan), axis=0) * 100)
        if normalize
        else piv / scale
    )

    bottom = np.zeros(len(piv))
    for c in piv.columns:
        h = heights[c].fillna(0).values
        if label is True:
            lab = f"{c}  (n={int(df[group_col].eq(c).sum()):,})"
        elif callable(label):
            lab = label(c)
        else:
            lab = None
        ax.bar(piv.index, h, bottom=bottom, color=color[c], width=width, label=lab)
        bottom += h
    return piv
