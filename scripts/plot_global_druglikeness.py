#!/usr/bin/env python
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True, parents=True)

TABLE2_CSV = TABLE_DIR / "table2_global_druglikeness_metrics.csv"

GENERATORS = ["GenMol", "MoLeR"]
METHODS = ["raw", "qed", "rulekit", "lite", "heavy"]

METRICS_BAR: List[Tuple[str, str]] = [
    ("qed_mean", "QED"),
    ("sa_mean", "SA"),
    ("logp_mean", "logP"),
    ("mw_mean", "MW"),
    ("tpsa_mean", "TPSA"),
    ("frac_csp3_mean", "Frac Csp3"),
]

METRICS_RADAR: List[Tuple[str, str]] = [
    ("qed_mean", "QED"),
    ("sa_mean", "SA"),
    ("logp_mean", "logP"),
    ("mw_mean", "MW"),
    ("tpsa_mean", "TPSA"),
    ("frac_csp3_mean", "Frac Csp3"),
]

COLOR_GENMOL = "#1f77b4"
COLOR_MOLER = "#ff7f0e"
COLOR_METHOD = {
    "raw": "#1f77b4",      # blue
    "qed": "#2ca02c",      # green
    "rulekit": "#d62728",  # red
    "lite": "#9467bd",     # purple
    "heavy": "#8c564b",    # brown
}


def load_table2(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["generator"] = pd.Categorical(df["generator"], categories=GENERATORS, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHODS, ordered=True)
    df = df.sort_values(["generator", "method"]).reset_index(drop=True)
    return df


def plot_global_bars(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    axes = axes.ravel()

    x = np.arange(len(METHODS))
    width = 0.35

    legend_handles = None

    for ax, (metric_col, metric_label) in zip(axes, METRICS_BAR):
        pivot = (
            df.pivot(index="method", columns="generator", values=metric_col)
            .reindex(METHODS)
        )

        genmol_vals = pivot[GENERATORS[0]].values
        moler_vals = pivot[GENERATORS[1]].values

        bars1 = ax.bar(
            x - width / 2,
            genmol_vals,
            width,
            label="GenMol",
            color=COLOR_GENMOL,
        )
        bars2 = ax.bar(
            x + width / 2,
            moler_vals,
            width,
            label="MoLeR",
            color=COLOR_MOLER,
        )

        if legend_handles is None:
            legend_handles = (bars1, bars2)

        ax.set_title(metric_label, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.tick_params(axis="both", labelsize=11)

    # Remove per-axis x labels; ticks already show methods
    for ax in axes:
        ax.set_xlabel("")

    # Common y label on the left
    fig.text(0.01, 0.5, "Mean value", va="center", rotation="vertical", fontsize=13)

    # Global legend at the very top, above all subplots
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ["GenMol", "MoLeR"],
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
            fontsize=13,
        )

    fig.tight_layout(rect=(0.03, 0.03, 0.97, 0.94))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _normalize_for_radar(values: np.ndarray) -> np.ndarray:
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if np.isclose(vmax, vmin):
        return np.ones_like(values) * 0.5
    return (values - vmin) / (vmax - vmin)


def plot_global_radar(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    n_metrics = len(METRICS_RADAR)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), subplot_kw=dict(polar=True))

    legend_handles = []

    for ax, generator in zip(axes, GENERATORS):
        sub = df[df["generator"] == generator].set_index("method")
        mat = []

        for metric_col, _ in METRICS_RADAR:
            vals = sub.loc[METHODS, metric_col].values.astype(float)
            mat.append(_normalize_for_radar(vals))

        mat = np.array(mat)  # shape: (metrics, methods)

        for j, method in enumerate(METHODS):
            vals = mat[:, j].tolist()
            vals += vals[:1]
            handle = ax.plot(
                angles,
                vals,
                label=method,
                color=COLOR_METHOD.get(method, None),
                linewidth=2,
            )[0]
            ax.fill(
                angles,
                vals,
                color=COLOR_METHOD.get(method, None),
                alpha=0.08,
            )
            if generator == GENERATORS[0]:
                legend_handles.append(handle)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [label for _, label in METRICS_RADAR],
            fontsize=12,
        )
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(generator, fontsize=20, pad=20)

    # Global legend at top, above both radar panels
    fig.legend(
        legend_handles,
        METHODS,
        loc="upper center",
        ncol=len(METHODS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.998),
        fontsize=13,
        title="Governance method",
        title_fontsize=14,
    )

    fig.tight_layout(rect=(0.03, 0.03, 0.97, 0.90))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    df = load_table2(TABLE2_CSV)

    bar_path = FIG_DIR / "global_properties_bars.png"
    radar_path = FIG_DIR / "global_properties_radar.png"

    print(f"[INFO] Loaded Table 2 with shape: {df.shape}")
    print(f"[INFO] Writing bar plot → {bar_path}")
    plot_global_bars(df, bar_path)

    print(f"[INFO] Writing radar plot → {radar_path}")
    plot_global_radar(df, radar_path)


if __name__ == "__main__":
    main()
