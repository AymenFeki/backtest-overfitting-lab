"""Figures. One colour and one marker shape per filter, in a fixed order.

A variant of a filter shares its parent's colour and is drawn dashed with a
hollow marker, so the palette never needs more than eight hues.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from . import config as cfg

# Fixed slot per filter, so a filter keeps its colour in every figure.
# name: (colour, marker, is a variant of the filter above it)
STYLE = {
    "naive": ("#2a78d6", "o", False),
    "holdout": ("#eb6834", "s", False),
    "walk_forward": ("#1baf7a", "^", False),
    "dsr": ("#eda100", "D", False),
    "dsr_null": ("#eda100", "D", True),
    "cpcv": ("#e87ba4", "v", False),
    "bh": ("#008300", "P", False),
    "by": ("#008300", "P", True),
    "romano_wolf": ("#6250d6", "X", False),
}
LABEL = {
    "naive": "Naive Sharpe",
    "holdout": "Holdout",
    "walk_forward": "Walk-forward",
    "dsr": "Deflated Sharpe (paper)",
    "dsr_null": "Deflated Sharpe (variant)",
    "cpcv": "CPCV",
    "bh": "Benjamini-Hochberg",
    "by": "Benjamini-Yekutieli",
    "romano_wolf": "Romano-Wolf",
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def fdr_power_figure(curves: pd.DataFrame, points: pd.DataFrame, path: Path) -> None:
    """FDR against power, one panel per signal strength, one curve per filter.

    The curve is the filter's whole range of cuts; the marker is its
    conventional cut with 95% intervals. Bottom right is where a filter wants
    to be: most real strategies found, few false ones let through.
    """
    strengths = list(cfg.STRENGTH_TARGETS)
    fig, axes = plt.subplots(
        1, len(strengths), figsize=(13, 3.9), sharex=True, sharey=True
    )
    fig.set_facecolor(SURFACE)
    for ax, strength in zip(axes, strengths, strict=True):
        group = f"all_{strength}"
        for name, (colour, marker, variant) in STYLE.items():
            line = "--" if variant else "-"
            curve = curves[(curves.worlds == group) & (curves["filter"] == name)]
            curve = curve.sort_values("cut")
            ax.plot(curve.power, curve.fdr, color=colour, linewidth=2, linestyle=line)
            p = points[(points.worlds == group) & (points["filter"] == name)].iloc[0]
            ax.errorbar(
                p.power,
                p.fdr,
                xerr=[[p.power - p.power_low], [p.power_high - p.power]],
                yerr=[[p.fdr - p.fdr_low], [p.fdr_high - p.fdr]],
                color=colour,
                linestyle=line,
                marker=marker,
                markersize=8,
                markerfacecolor=SURFACE if variant else colour,
                markeredgecolor=colour if variant else SURFACE,
                markeredgewidth=1.5,
                elinewidth=1.2,
                capsize=2,
                label=LABEL[name],
            )
        target = cfg.STRENGTH_TARGETS[strength]
        ax.set_title(
            f"Signal: {strength} (oracle Sharpe {target})", fontsize=10, color=INK
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("FDR: share of accepted that are false", color=MUTED)
    fig.supxlabel(
        "Power: share of truly profitable strategies accepted",
        fontsize=10,
        color=MUTED,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.96),
    )
    fig.suptitle(
        "Grid search: false discoveries against power, by filter "
        "(marker = conventional cut, bars = 95% interval)",
        y=1.2,
        fontsize=11,
        color=INK,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
