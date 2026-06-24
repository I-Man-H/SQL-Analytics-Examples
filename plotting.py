"""
Reusable plotting utilities.

All functions return matplotlib Figure objects so they can be
displayed in notebooks or saved to file with fig.savefig().

Style is set once at module import — call set_style() to apply
it manually if needed.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

def set_style() -> None:
    """Apply a clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.framealpha": 0.8,
        "figure.dpi": 120,
    })


set_style()

PALETTE = sns.color_palette("muted")


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def plot_distribution(
    series: pd.Series,
    title: str = "",
    xlabel: str = "",
    bins: int = 40,
    show_stats: bool = True,
) -> plt.Figure:
    """Histogram with KDE overlay and optional summary stats annotation."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(series.dropna(), bins=bins, kde=True, ax=ax, color=PALETTE[0])

    if show_stats:
        mean, median = series.mean(), series.median()
        ax.axvline(mean, color="crimson", linestyle="--", linewidth=1.2, label=f"Mean: {mean:.2f}")
        ax.axvline(median, color="darkorange", linestyle=":", linewidth=1.2, label=f"Median: {median:.2f}")
        ax.legend()

    ax.set_title(title or f"Distribution of {series.name}")
    ax.set_xlabel(xlabel or series.name)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_boxplot_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    title: str = "",
    palette: str = "muted",
) -> plt.Figure:
    """Box plot comparing a numeric variable across groups."""
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby(group_col)[value_col].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=group_col, y=value_col, order=order, palette=palette, ax=ax)
    ax.set_title(title or f"{value_col} by {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "",
    rolling_window: Optional[int] = None,
    color: str = None,
) -> plt.Figure:
    """Line chart with optional rolling average overlay."""
    fig, ax = plt.subplots(figsize=(12, 5))
    color = color or PALETTE[0]

    ax.plot(df[date_col], df[value_col], alpha=0.5, linewidth=1, color=color, label=value_col)

    if rolling_window:
        rolling = df[value_col].rolling(rolling_window, center=True).mean()
        ax.plot(df[date_col], rolling, linewidth=2, color="crimson",
                label=f"{rolling_window}-period rolling avg")
        ax.legend()

    ax.set_title(title or f"{value_col} over time")
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Categorical / ranking
# ---------------------------------------------------------------------------

def plot_bar_ranked(
    series: pd.Series,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Count",
    top_n: int = 15,
    horizontal: bool = True,
) -> plt.Figure:
    """Ranked bar chart for categorical frequency or aggregated values."""
    data = series.nlargest(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))

    if horizontal:
        bars = ax.barh(data.index[::-1], data.values[::-1], color=PALETTE[0])
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        for bar, val in zip(bars, data.values[::-1]):
            ax.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,.0f}", va="center", fontsize=9)
    else:
        bars = ax.bar(data.index, data.values, color=PALETTE[0])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        plt.xticks(rotation=35, ha="right")

    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation matrix",
    annot: bool = True,
    mask_upper: bool = True,
) -> plt.Figure:
    """Heatmap of a correlation matrix with optional upper-triangle mask."""
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) if mask_upper else None
    fig, ax = plt.subplots(figsize=(max(6, len(corr_matrix) * 0.9),
                                    max(5, len(corr_matrix) * 0.8)))
    sns.heatmap(
        corr_matrix, mask=mask, annot=annot, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cohort / retention
# ---------------------------------------------------------------------------

def plot_cohort_heatmap(
    cohort_matrix: pd.DataFrame,
    title: str = "Cohort retention (%)",
) -> plt.Figure:
    """Percentage retention heatmap for cohort analysis."""
    fig, ax = plt.subplots(figsize=(max(10, len(cohort_matrix.columns)),
                                    max(5, len(cohort_matrix) * 0.5)))
    sns.heatmap(
        cohort_matrix, annot=True, fmt=".0f",
        cmap="YlGnBu", vmin=0, vmax=100,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Cohort (first purchase month)")
    fig.tight_layout()
    return fig
