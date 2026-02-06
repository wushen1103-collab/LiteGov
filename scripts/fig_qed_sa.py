#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QED vs SA distributions for different governance lines and backbones.

- Two figures:
    1) Hexbin grid (2x5) with shared LogNorm color scale.
    2) KDE grid (2x5) with shared linear color scale (for supplement).

- Backbones: GenMol, MoLeR
- Lines: Raw, QED, RuleKit, LiteGov, HeavyGov
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# Style (aligned with docking / Tanimoto figures)
# ---------------------------------------------------------------------------

def set_figure_style() -> None:
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
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGS_DIR = REPO_ROOT / "figs"

BACKBONES: List[Tuple[str, str]] = [
    ("GenMol", "genmol"),
    ("MoLeR", "moler"),
]

GOV_LINES: List[Tuple[str, str]] = [
    ("Raw", "raw"),
    ("QED", "qed"),
    ("RuleKit", "rulekit"),
    ("LiteGov", "lite"),
    ("HeavyGov", "heavy"),
]

CMAP_SEQ = "viridis"
MAX_ROWS_PER_FILE = 2000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of candidate columns {list(candidates)} found in {list(df.columns)}")


def load_mode(
    backbone_dir: str,
    mode_dir: str,
    max_rows_per_file: int = MAX_ROWS_PER_FILE,
) -> Tuple[np.ndarray, np.ndarray]:
    base = RESULTS_DIR / backbone_dir / mode_dir
    files = sorted(p for p in base.glob("*.csv") if p.name.lower() != "summary.csv")

    if not files:
        print(f"[load_mode] no files found in {base}")
        return np.array([]), np.array([])

    sa_values: List[np.ndarray] = []
    qed_values: List[np.ndarray] = []

    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[load_mode] failed to read {path}: {e}")
            continue

        try:
            sa_col = find_column(df, ["SA", "sa_ertl", "sa"])
            q_col = find_column(df, ["QED", "qed"])
        except KeyError as e:
            print(f"[load_mode] skipping {path.name}: {e}")
            continue

        sa_arr = pd.to_numeric(df[sa_col], errors="coerce").to_numpy()
        q_arr = pd.to_numeric(df[q_col], errors="coerce").to_numpy()
        mask = np.isfinite(sa_arr) & np.isfinite(q_arr)

        sa_arr = sa_arr[mask]
        q_arr = q_arr[mask]

        if sa_arr.size == 0:
            continue

        if sa_arr.size > max_rows_per_file:
            sa_arr = sa_arr[:max_rows_per_file]
            q_arr = q_arr[:max_rows_per_file]

        sa_values.append(sa_arr.astype(float))
        qed_values.append(q_arr.astype(float))

    if not sa_values:
        print(f"[load_mode] no usable data in {base}")
        return np.array([]), np.array([])

    sa_all = np.concatenate(sa_values)
    q_all = np.concatenate(qed_values)

    print(
        f"[load_mode] {backbone_dir}/{mode_dir}: "
        f"{len(files)} file(s), {sa_all.size} valid SA/QED pairs"
    )

    return sa_all, q_all


# ---------------------------------------------------------------------------
# Axis & hexbin utilities
# ---------------------------------------------------------------------------

def compute_axis_limits(
    sa_list: Iterable[np.ndarray],
    q_list: Iterable[np.ndarray],
    pad_fraction: float = 0.02,
) -> Tuple[float, float, float, float]:
    sa_all = np.concatenate([x for x in sa_list if x.size > 0])
    q_all = np.concatenate([x for x in q_list if x.size > 0])

    sa_min, sa_max = float(sa_all.min()), float(sa_all.max())
    q_min, q_max = float(q_all.min()), float(q_all.max())

    sa_pad = (sa_max - sa_min) * pad_fraction
    q_pad = (q_max - q_min) * pad_fraction

    return sa_min - sa_pad, sa_max + sa_pad, q_min - q_pad, q_max + q_pad


def compute_global_hexbin_max(
    data: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    sa_min: float,
    sa_max: float,
    q_min: float,
    q_max: float,
    gridsize: int,
) -> int:
    global_max = 0
    sa_edges = np.linspace(sa_min, sa_max, gridsize + 1)
    q_edges = np.linspace(q_min, q_max, gridsize + 1)

    for (_, _), (sa, q) in data.items():
        if sa.size == 0:
            continue
        H, _, _ = np.histogram2d(sa, q, bins=(sa_edges, q_edges))
        global_max = max(global_max, int(H.max()))

    return max(global_max, 1)


def plot_hexbin_grid(
    data: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    sa_min: float,
    sa_max: float,
    q_min: float,
    q_max: float,
    gridsize: int = 80,
    figsize_per_panel: Tuple[float, float] = (2.1, 2.1),
) -> None:
    n_backbones = len(BACKBONES)
    n_lines = len(GOV_LINES)

    fig_width = figsize_per_panel[0] * n_lines + 1.3
    fig_height = figsize_per_panel[1] * n_backbones

    fig, axes = plt.subplots(
        n_backbones,
        n_lines,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = np.array(axes)

    max_count = compute_global_hexbin_max(
        data, sa_min, sa_max, q_min, q_max, gridsize
    )
    norm = LogNorm(vmin=1, vmax=max_count)

    reference_hex = None

    for i_b, (backbone_label, backbone_dir) in enumerate(BACKBONES):
        for j_l, (line_label, mode_dir) in enumerate(GOV_LINES):
            ax = axes[i_b, j_l]
            sa, q = data[(backbone_dir, mode_dir)]

            if sa.size == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    alpha=0.7,
                    fontsize=8,
                    transform=ax.transAxes,
                )
            else:
                hb = ax.hexbin(
                    sa,
                    q,
                    gridsize=gridsize,
                    extent=[sa_min, sa_max, q_min, q_max],
                    mincnt=1,
                    linewidths=0.0,
                    cmap=CMAP_SEQ,
                    edgecolors="none",
                    norm=norm,
                    rasterized=True,
                )
                if reference_hex is None:
                    reference_hex = hb

            if i_b == 0:
                ax.set_title(line_label, fontsize=10)
            if j_l == 0:
                ax.set_ylabel(f"{backbone_label}\nQED", fontsize=10)
            if i_b == n_backbones - 1:
                ax.set_xlabel("SA", fontsize=10)

            ax.set_xlim(sa_min, sa_max)
            ax.set_ylim(q_min, q_max)
            ax.tick_params(axis="both", which="both", direction="out")

    if reference_hex is not None:
        cbar = fig.colorbar(reference_hex, ax=axes, pad=0.015, fraction=0.035)
        cbar.set_label("Density (log scale)", fontsize=10)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIGS_DIR / "qed_sa_hexbin_genmol_moler.png"
    out_pdf = FIGS_DIR / "qed_sa_hexbin_genmol_moler.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# KDE utilities
# ---------------------------------------------------------------------------

def compute_kde_for_panel(sa, q, sa_grid, q_grid):
    if sa.size < 50:
        return np.zeros((q_grid.size, sa_grid.size))
    kde = gaussian_kde(np.vstack([sa, q]))
    SA, Q = np.meshgrid(sa_grid, q_grid)
    Z = kde(np.vstack([SA.ravel(), Q.ravel()])).reshape(SA.shape)
    return Z


def compute_kde_grid(
    data,
    sa_min,
    sa_max,
    q_min,
    q_max,
    n_grid=120,
):
    sa_grid = np.linspace(sa_min, sa_max, n_grid)
    q_grid = np.linspace(q_min, q_max, n_grid)

    results = {}
    all_Z = []

    for key, (sa, q) in data.items():
        if sa.size == 0:
            Z = np.zeros((q_grid.size, sa_grid.size))
        else:
            Z = compute_kde_for_panel(sa, q, sa_grid, q_grid)
        results[key] = (sa_grid, q_grid, Z)
        all_Z.append(Z.ravel())

    all_Z = np.concatenate(all_Z)
    vmax = float(np.percentile(all_Z, 99.5))
    if vmax <= 0:
        vmax = float(all_Z.max())
    return results, vmax


def plot_kde_grid(
    data,
    sa_min,
    sa_max,
    q_min,
    q_max,
    n_grid=120,
    figsize_per_panel=(2.1, 2.1),
) -> None:
    kde_results, global_vmax = compute_kde_grid(
        data, sa_min, sa_max, q_min, q_max, n_grid
    )

    n_backbones = len(BACKBONES)
    n_lines = len(GOV_LINES)

    fig_width = figsize_per_panel[0] * n_lines + 1.3
    fig_height = figsize_per_panel[1] * n_backbones

    fig, axes = plt.subplots(
        n_backbones,
        n_lines,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = np.array(axes)
    reference_im = None

    for i_b, (backbone_label, backbone_dir) in enumerate(BACKBONES):
        for j_l, (line_label, mode_dir) in enumerate(GOV_LINES):
            ax = axes[i_b, j_l]
            sa_grid, q_grid, Z = kde_results[(backbone_dir, mode_dir)]
            SA, Q = np.meshgrid(sa_grid, q_grid)

            if np.allclose(Z, 0):
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    alpha=0.7,
                    fontsize=8,
                    transform=ax.transAxes,
                )
            else:
                im = ax.pcolormesh(
                    SA,
                    Q,
                    Z,
                    shading="auto",
                    cmap=CMAP_SEQ,
                    vmin=0.0,
                    vmax=global_vmax,
                    rasterized=True,
                )
                if reference_im is None:
                    reference_im = im

            if i_b == 0:
                ax.set_title(line_label, fontsize=10)
            if j_l == 0:
                ax.set_ylabel(f"{backbone_label}\nQED", fontsize=10)
            if i_b == n_backbones - 1:
                ax.set_xlabel("SA", fontsize=10)

            ax.set_xlim(sa_min, sa_max)
            ax.set_ylim(q_min, q_max)
            ax.tick_params(axis="both", which="both", direction="out")

    if reference_im is not None:
        cbar = fig.colorbar(reference_im, ax=axes, pad=0.015, fraction=0.035)
        cbar.set_label("KDE density", fontsize=10)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIGS_DIR / "qed_sa_kde_genmol_moler.png"
    out_pdf = FIGS_DIR / "qed_sa_kde_genmol_moler.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_figure_style()

    data: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for _, backbone_dir in BACKBONES:
        for _, mode_dir in GOV_LINES:
            sa, q = load_mode(backbone_dir, mode_dir)
            data[(backbone_dir, mode_dir)] = (sa, q)

    sa_list = [sa for (sa, _) in data.values() if sa.size > 0]
    q_list = [q for (_, q) in data.values() if q.size > 0]
    if not sa_list or not q_list:
        print("No data found.")
        return

    sa_min, sa_max, q_min, q_max = compute_axis_limits(sa_list, q_list)

    plot_hexbin_grid(data, sa_min, sa_max, q_min, q_max)
    plot_kde_grid(data, sa_min, sa_max, q_min, q_max)

    print("Done.")


if __name__ == "__main__":
    main()
