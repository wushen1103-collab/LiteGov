#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCK_DIR = ROOT / "docking"
STATS_DIR = DOCK_DIR / "stats"
FIG_DIR = ROOT / "figs"

METHOD_ORDER = ["raw", "qed", "rulekit", "lite", "heavy"]

METHOD_COLORS = {
    "raw": "#b0b0b0",
    "qed": "#a6cee3",
    "rulekit": "#fdbf6f",
    "lite": "#b2df8a",
    "heavy": "#cab2d6",
}


# ----------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------


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


def load_scores(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Docking score file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ("target", "method", "score"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path}")
    df = df[(df["score"] > -30) & (df["score"] < 0)].copy()
    return df


def summarize_scores(df: pd.DataFrame, thresholds=(-6.0, -6.5)) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["target", "method"], as_index=False)

    for (target, method), g in grouped:
        scores = g["score"].values
        n = len(scores)
        row = {
            "target": target,
            "method": method,
            "n": n,
            "score_mean": float(np.mean(scores)) if n > 0 else np.nan,
            "score_std": float(np.std(scores, ddof=1)) if n > 1 else np.nan,
        }
        for thr in thresholds:
            col_name = f"hit_rate_le_{thr}"
            hit_rate = float((scores <= thr).mean()) if n > 0 else np.nan
            row[col_name] = hit_rate
        rows.append(row)

    return pd.DataFrame(rows)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0:
        raise ValueError("Both samples must be non-empty for Cliff's delta")
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float(gt - lt) / (nx * ny)


def compute_effect_sizes(df: pd.DataFrame, ref_method: str = "raw") -> pd.DataFrame:
    rows = []
    for target, g in df.groupby("target", as_index=False):
        methods_here = set(g["method"])
        if ref_method not in methods_here:
            continue
        ref_scores = -g.loc[g["method"] == ref_method, "score"].to_numpy()
        for method in sorted(methods_here):
            if method == ref_method:
                continue
            m_scores = -g.loc[g["method"] == method, "score"].to_numpy()
            if len(m_scores) == 0:
                continue
            delta = cliffs_delta(m_scores, ref_scores)
            rows.append(
                {
                    "target": target,
                    "method": method,
                    "ref_method": ref_method,
                    "cliffs_delta": delta,
                    "n_ref": len(ref_scores),
                    "n_method": len(m_scores),
                }
            )
    return pd.DataFrame(rows)


def find_hit_rate_column(columns, threshold: float) -> str | None:
    thr_str = str(threshold)
    thr_str_simple = thr_str.replace(".0", "")
    t1 = thr_str
    t2 = thr_str_simple
    t3 = thr_str.replace(".", "_")
    t4 = thr_str_simple.replace(".", "_")
    candidates = [
        f"hit_rate_le_{t1}",
        f"hit_rate_le_{t2}",
        f"hit_rate_le_{t3}",
        f"hit_rate_le_{t4}",
    ]
    for name in candidates:
        if name in columns:
            return name
    numeric_pattern = re.escape(t3.lstrip("-"))
    regex = re.compile(rf"^hit_rate.*{numeric_pattern}$")
    for col in columns:
        if regex.match(col):
            return col
    return None


def get_target_order(
    summary: pd.DataFrame,
    thresholds,
    sort_method: str = "lite",
) -> list[str]:
    """Decide a consistent target ordering for all plots of one generator."""
    all_targets = sorted(summary["target"].unique())
    methods_here = set(summary["method"])
    if sort_method not in methods_here:
        return all_targets

    primary_thr = thresholds[0]
    col = find_hit_rate_column(summary.columns, primary_thr)
    if col is None:
        return all_targets

    sub = summary[summary["method"] == sort_method].copy()
    if col not in sub.columns:
        return all_targets

    sub["_sort_value"] = sub[col].fillna(0.0)
    sub = sub.sort_values("_sort_value", ascending=False)
    ordered = list(sub["target"].values)

    missing = [t for t in all_targets if t not in ordered]
    ordered.extend(missing)
    return ordered


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def plot_hit_rates(
    summary: pd.DataFrame,
    thresholds,
    fig_dir: Path,
    tag: str,
    targets_order: list[str],
) -> None:
    set_figure_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    targets = targets_order or sorted(summary["target"].unique())
    x = np.arange(len(targets))
    n_methods = len(METHOD_ORDER)
    offsets = np.linspace(-0.5, 0.5, n_methods) * 0.12 * 1.6

    # compute a shared y-limit across thresholds for this generator
    hr_max_values = []
    for thr in thresholds:
        col_name = find_hit_rate_column(summary.columns, thr)
        if col_name is None:
            continue
        hr_max_values.append(summary[col_name].max())
    if hr_max_values:
        max_val = float(np.nanmax(hr_max_values))
        y_upper = 0.1 if max_val <= 0 else min(1.0, max_val * 1.2)
    else:
        y_upper = 0.1

    for thr in thresholds:
        col_name = find_hit_rate_column(summary.columns, thr)
        if col_name is None:
            print(f"[WARN] no hit-rate column found for threshold {thr}")
            continue

        fig, ax = plt.subplots(figsize=(3.5 + 0.5 * len(targets), 3.0))
        width = 0.12

        for idx_m, method in enumerate(METHOD_ORDER):
            sub = summary[summary["method"] == method].set_index("target")
            heights = []
            for t in targets:
                if t in sub.index:
                    heights.append(sub.loc[t, col_name])
                else:
                    heights.append(np.nan)
            heights_arr = np.asarray(heights, dtype=float)
            ax.bar(
                x + offsets[idx_m],
                heights_arr,
                width=width * 0.9,
                label=method,
                color=METHOD_COLORS.get(method, "#cccccc"),
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=30, ha="right")
        ax.set_ylabel(f"Hit rate (score \u2264 {thr})")
        ax.set_ylim(0.0, y_upper)

        ax.legend(
            frameon=False,
            ncol=len(METHOD_ORDER),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
        )

        fig.tight_layout()
        safe_thr = str(thr).replace(".", "_")
        out_path = fig_dir / f"{tag}_fig_hit_rates_le_{safe_thr}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] wrote {out_path}")


def plot_box_by_target(
    df: pd.DataFrame,
    fig_dir: Path,
    tag: str,
    targets_order: list[str],
) -> None:
    set_figure_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    targets = targets_order or sorted(df["target"].unique())
    n_targets = len(targets)

    n_cols = min(3, n_targets)
    n_rows = int(np.ceil(n_targets / n_cols))

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(n_cols * 3.0, n_rows * 2.6),
        squeeze=False,
        sharey=True,
    )

    score_min = df["score"].min()
    score_max = df["score"].max()

    for idx, target in enumerate(targets):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        sub = df[df["target"] == target]

        positions = np.arange(len(methods))
        box_data = []
        for method in methods:
            scores = sub.loc[sub["method"] == method, "score"].values
            box_data.append(scores if scores.size > 0 else [])

        box_props = dict(linewidth=0.8)
        whisker_props = dict(linewidth=0.8)

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            boxprops=box_props,
            whiskerprops=whisker_props,
            medianprops=dict(linewidth=0.9, color="#333333"),
        )

        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS.get(method, "#cccccc"))
            patch.set_edgecolor("white")

        for i, method in enumerate(methods):
            scores = sub.loc[sub["method"] == method, "score"].values
            if scores.size == 0:
                continue
            xs = np.random.normal(loc=positions[i], scale=0.03, size=scores.size)
            ax.scatter(
                xs,
                scores,
                s=4,
                alpha=0.4,
                color=METHOD_COLORS.get(method, "#666666"),
                linewidths=0,
            )

        ax.set_title(target)
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylim(score_min - 0.5, score_max + 0.5)

    total_axes = n_rows * n_cols
    for idx in range(n_targets, total_axes):
        r = idx // n_cols
        c = idx % n_cols
        fig.delaxes(axes[r, c])

    fig.tight_layout()
    out_path = fig_dir / f"{tag}_fig_scores_box_by_target.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_effect_forest(
    effects_df: pd.DataFrame,
    fig_dir: Path,
    tag: str,
    targets_order: list[str],
) -> None:
    set_figure_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    methods = ["qed", "rulekit", "lite", "heavy"]
    methods = [m for m in methods if m in set(effects_df["method"])]

    targets = targets_order or sorted(effects_df["target"].unique())
    y = np.arange(len(targets))

    fig_height = 0.6 * len(targets) + 1.5
    fig, ax = plt.subplots(figsize=(4.0, fig_height))

    ax.axvline(0.0, color="#999999", linewidth=0.8, linestyle="--", zorder=0)

    jitter_width = 0.06

    for method in methods:
        sub = effects_df[effects_df["method"] == method].set_index("target")
        xs = []
        ys = []
        for i, t in enumerate(targets):
            if t not in sub.index:
                continue
            xs.append(sub.loc[t, "cliffs_delta"])
            ys.append(y[i] + np.random.uniform(-jitter_width, jitter_width))
        ax.scatter(
            xs,
            ys,
            label=method,
            s=18,
            alpha=0.9,
            color=METHOD_COLORS.get(method, "#666666"),
        )

    ax.set_yticks(y)
    ax.set_yticklabels(targets)
    ax.set_xlabel("Cliff's delta vs raw (dock score)")
    ax.set_xlim(-1.0, 1.0)

    # move legend further above the axes so it no longer overlaps the plot
    ax.legend(
        frameon=False,
        ncol=len(methods),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )

    # leave some space at the top for the legend
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))

    out_path = fig_dir / f"{tag}_fig_effect_sizes_forest.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def run_for_generator(tag: str, scores_csv: Path) -> None:
    if not scores_csv.exists():
        print(f"[WARN] scores CSV not found for {tag}: {scores_csv}")
        return

    print(f"[INFO] === {tag.upper()} ===")
    df = load_scores(scores_csv)
    print(f"[INFO] Loaded {len(df)} rows from {scores_csv}")
    print(f"[INFO] Targets: {sorted(df['target'].unique())}")
    print(f"[INFO] Methods: {sorted(df['method'].unique())}")

    thresholds = (-6.0, -6.5)

    summary_df = summarize_scores(df, thresholds=thresholds)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    summary_out = STATS_DIR / f"dock_summary_by_target_method_{tag}.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"[OK] wrote {summary_out}")

    effects_df = compute_effect_sizes(df, ref_method="raw")
    effects_out = STATS_DIR / f"dock_effect_sizes_vs_raw_{tag}.csv"
    effects_df.to_csv(effects_out, index=False)
    print(f"[OK] wrote {effects_out}")

    targets_order = get_target_order(summary_df, thresholds=thresholds, sort_method="lite")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_hit_rates(summary_df, thresholds=thresholds, fig_dir=FIG_DIR, tag=tag, targets_order=targets_order)
    plot_box_by_target(df, fig_dir=FIG_DIR, tag=tag, targets_order=targets_order)
    plot_effect_forest(effects_df, fig_dir=FIG_DIR, tag=tag, targets_order=targets_order)


def main() -> None:
    genmol_csv = STATS_DIR / "dock_scores_all.csv"
    moler_csv = STATS_DIR / "dock_scores_all_moler.csv"

    run_for_generator("genmol", genmol_csv)
    run_for_generator("moler", moler_csv)


if __name__ == "__main__":
    main()
