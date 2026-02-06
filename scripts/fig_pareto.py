#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pareto trade-off figure: QED vs end-to-end seconds.

We plot five governance lines:

    Raw, QED, RuleKit, LiteGov, HeavyGov

Data is read from figs/paper_metrics_agg.csv, which is expected to contain:
    - a "mode" column identifying each line
    - e2e_seconds_mean (or compatible aliases)
    - qed_mean_mean (or compatible aliases)
    - optionally qed_mean_std / qed_std for y-error bars

This script is styled to match the docking / Tanimoto / QED–SA figures.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "figs" / "paper_metrics_agg.csv"
FIG_DIR = ROOT / "figs"

# Colors aligned with docking / Tanimoto figures
METHOD_COLORS: Dict[str, str] = {
    "raw": "#b0b0b0",      # light gray
    "qed": "#a6cee3",      # light blue
    "rulekit": "#fdbf6f",  # light orange
    "lite": "#b2df8a",     # light green
    "heavy": "#cab2d6",    # light purple
}


def set_figure_style() -> None:
    """Match the style used in other figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Return the first column name from `candidates` that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"None of {list(candidates)} found in DataFrame columns={df.columns.tolist()}"
    )


def get_row_for_modes(df: pd.DataFrame, mode_values: Iterable[str]) -> Optional[pd.Series]:
    """
    Return the first row whose df['mode'] is in `mode_values`, or None.
    """
    if "mode" not in df.columns:
        raise ValueError("Expected a 'mode' column in the CSV.")
    for mv in mode_values:
        sub = df[df["mode"] == mv]
        if not sub.empty:
            return sub.iloc[0]
    return None


# ---------------------------------------------------------------------------
# Main plotting logic
# ---------------------------------------------------------------------------

def main(csv_path: Optional[str] = None) -> None:
    set_figure_style()

    csv_file = Path(csv_path) if csv_path is not None else DEFAULT_CSV
    if not csv_file.exists():
        raise FileNotFoundError(f"Could not find CSV: {csv_file}")

    df = pd.read_csv(csv_file)

    # X-axis: end-to-end seconds (try several aliases)
    sec_col = pick_first_existing(
        df,
        ["e2e_seconds_mean", "seconds_mean", "seconds", "latency_mean"],
    )

    # Y-axis: QED mean (try several aliases)
    y_col = pick_first_existing(
        df,
        ["qed_mean_mean", "qed_mean"],
    )

    # Y error bar: QED std (if present)
    yerr_col: Optional[str] = None
    for cand in ["qed_mean_std", "qed_std"]:
        if cand in df.columns:
            yerr_col = cand
            break

    # Config for each Pareto point:
    # (aliases in CSV, legend label, method key for color, marker style)
    mode_configs: List[Tuple[List[str], str, str, str]] = [
        (["raw", "pass"], "Raw", "raw", "o"),
        (["qed"], "QED", "qed", "s"),
        (["rulekit"], "RuleKit", "rulekit", "D"),
        (["lite", "filter+rank"], "LiteGov", "lite", "^"),
        (["heavy", "govern"], "HeavyGov", "heavy", "v"),
    ]

    xs: List[float] = []
    ys: List[float] = []
    yerrs: List[float] = []
    labels: List[str] = []
    colors: List[str] = []
    markers: List[str] = []

    for aliases, label, key, marker in mode_configs:
        row = get_row_for_modes(df, aliases)
        if row is None:
            print(f"[WARN] no row found for aliases {aliases}; skipping {label}")
            continue

        x = float(row[sec_col])
        y = float(row[y_col])

        if yerr_col is not None and not pd.isna(row[yerr_col]):
            err = float(row[yerr_col])
        else:
            err = 0.0

        xs.append(x)
        ys.append(y)
        yerrs.append(err)
        labels.append(label)
        colors.append(METHOD_COLORS.get(key, "#000000"))
        markers.append(marker)

        print(
            f"[INFO] {label}: {sec_col}={x:.3f}, {y_col}={y:.3f}, "
            f"{yerr_col or 'no_std'}={err:.3f}"
        )

    if not xs:
        print("No data points collected; nothing to plot.")
        return

    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    yerrs_arr = np.asarray(yerrs)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    for i in range(len(xs_arr)):
        ax.errorbar(
            xs_arr[i],
            ys_arr[i],
            yerr=yerrs_arr[i] if yerrs_arr[i] > 0 else None,
            fmt=markers[i],
            markersize=6,
            capsize=3,
            linestyle="none",
            color=colors[i],
            ecolor=colors[i],
            label=labels[i],
        )

    # Optionally, connect points with a light line for visual guidance
    # (sorted by x; purely cosmetic)
    order = np.argsort(xs_arr)
    ax.plot(
        xs_arr[order],
        ys_arr[order],
        linestyle="--",
        linewidth=1.0,
        color="#666666",
        alpha=0.4,
        zorder=0,
    )

    ax.set_xlabel("End-to-end time (s)")
    ax.set_ylabel("Mean QED")

    # Expand axes slightly for aesthetics
    x_margin = (xs_arr.max() - xs_arr.min()) * 0.1 if xs_arr.size > 1 else 0.1
    y_margin = (ys_arr.max() - ys_arr.min()) * 0.1 if ys_arr.size > 1 else 0.02

    ax.set_xlim(xs_arr.min() - x_margin, xs_arr.max() + x_margin)
    ax.set_ylim(ys_arr.min() - y_margin, ys_arr.max() + y_margin)

    ax.tick_params(axis="both", which="both", direction="out")

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.legend(loc="lower right", frameon=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / "pareto_qed_vs_time.png"
    out_pdf = FIG_DIR / "pareto_qed_vs_time.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")


if __name__ == "__main__":
    main()
