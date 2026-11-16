"""Figures. One colour and one marker shape per filter, in a fixed order.

A variant of a filter shares its parent's colour and is drawn dashed with a
hollow marker, so the palette never needs more than eight hues.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
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
    "pbo": ("#e34948", "*", False),
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
    "pbo": "PBO (best strategy only)",
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


_SERIES = (("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "D"))
# Each panel: (title, [(group of worlds, legend label), ...]), first = baseline.
VARIANT_PANELS = [
    (
        "No signal",
        [("no_signal", "Plain world"), ("no_signal_regimes", "Regime switching")],
    ),
    (
        "Medium signal",
        [
            ("all_medium", "Plain world"),
            ("all_medium_regimes", "Regime switching"),
            ("all_medium_regimes_matched", "Same, signal rescaled"),
        ],
    ),
    (
        "Obvious signal",
        [("all_obvious", "Plain world"), ("all_obvious_decay", "Alpha decay")],
    ),
]
VARIANT_TITLE = (
    "Filters at their conventional cut: plain worlds against regime "
    "switching and alpha decay (bars = 95% interval)"
)
AGENT_PANELS = [
    (
        "No signal",
        [("grid_no_signal", "Grid search"), ("agent_no_signal", "LLM agent")],
    ),
    (
        "Medium signal",
        [("grid_all_medium", "Grid search"), ("agent_all_medium", "LLM agent")],
    ),
]
AGENT_TITLE = (
    "Filters at their conventional cut: grid search (76 strategies, 200 worlds "
    "per type) against the LLM agent (20 rounds, 30 worlds per type)"
)


def comparison_figure(
    points: pd.DataFrame, path: Path, panels: list, title: str
) -> None:
    """Each filter at its conventional cut, one column of panels per setting.

    Top row FDR, bottom row power, bars are 95% intervals. Within a panel the
    first series is the baseline and the others are what it is compared with.
    """
    names = list(STYLE)
    height = np.arange(len(names))[::-1]
    width = 4.3 * len(panels) + 0.1
    fig, axes = plt.subplots(2, len(panels), figsize=(width, 7.6), sharey=True)
    fig.set_facecolor(SURFACE)
    for column, (panel_title, groups) in enumerate(panels):
        plain = groups[0][0]
        shifts = np.linspace(0.2, -0.2, len(groups))
        for row, measure in enumerate(("fdr", "power")):
            ax = axes[row, column]
            for (group, tag), (colour, marker), shift in zip(
                groups, _SERIES, shifts, strict=False
            ):
                rows = points[points.worlds == group].set_index("filter").loc[names]
                low = rows[measure] - rows[f"{measure}_low"]
                high = rows[f"{measure}_high"] - rows[measure]
                ax.errorbar(
                    rows[measure],
                    height + shift,
                    xerr=[low, high],
                    fmt=marker,
                    color=colour,
                    markersize=7,
                    markeredgecolor=SURFACE,
                    elinewidth=1.2,
                    capsize=2,
                    label=tag,
                )
            if row == 0:
                ax.set_title(panel_title, fontsize=10, color=INK, pad=26)
                ax.legend(
                    frameon=False,
                    fontsize=9,
                    ncol=3,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.99),
                )
            undefined = points[points.worlds == plain][measure].isna().all()
            if undefined:
                ax.text(
                    0.5,
                    0.5,
                    "No real strategy exists,\nso power is undefined",
                    transform=ax.transAxes,
                    ha="center",
                    color=MUTED,
                )
            ax.set_xlabel("FDR" if measure == "fdr" else "Power", color=MUTED)
            ax.set_xlim(left=-0.01)
            if undefined:
                ax.set_xticks([])
                ax.set_xlabel("")
            ax.set_yticks(height, [LABEL[name] for name in names])
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, linewidth=0.8, axis="x")
            ax.tick_params(colors=MUTED, length=0)
            for side in ax.spines.values():
                side.set_visible(False)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
