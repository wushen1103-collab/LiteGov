#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LiteGov filter+rank ablation (Top-K on existing descriptors).

This version does NOT recompute any RDKit properties.
It assumes the input CSVs in results/denovo/filter/*.csv already contain at least:

- 'smiles'      : raw SMILES
- 'smiles_can'  : canonical SMILES (preferred, optional but recommended)
- 'QED'         : float
- one SA-like column: 'SA' (preferred), or 'sa_ertl', or 'sa'
- optionally 'scaffold' for scaffold diversity

We simply:
  1) read each filter_*.csv
  2) compute a LiteGov score in-place from existing QED / SA
  3) select Top-K by that score
  4) write filter_rank_*.csv + a summary.csv
"""

import glob
import os
import time
from typing import Optional

import numpy as np
import pandas as pd


K = 100
IN_GLOB = "results/denovo/filter/*.csv"
OUT_DIR = "results/denovo/filter_rank"

os.makedirs(OUT_DIR, exist_ok=True)


def get_smiles_col(df: pd.DataFrame) -> Optional[str]:
    """Return a sensible smiles column, preferring canonical SMILES."""
    for c in [
        "smiles_can",        # preferred canonical SMILES
        "smiles",
        "SMILES",
        "Smiles",
        "SMILE",
        "SMILEs",
        "canonical_smiles",
        "smile",
    ]:
        if c in df.columns:
            return c
    return None  # we can still proceed; uniqueness will be NaN


def min_max_norm(series: pd.Series, eps: float = 1e-6) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    vmin = s.min()
    vmax = s.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < eps:
        return pd.Series(0.0, index=s.index)
    return (s - vmin) / (vmax - vmin + eps)


def _pick_sa_column(df: pd.DataFrame) -> pd.Series:
    """Pick a SA column with a unified precedence."""
    if "SA" in df.columns:
        return df["SA"]
    if "sa_ertl" in df.columns:
        return df["sa_ertl"]
    if "sa" in df.columns:
        return df["sa"]
    raise ValueError("No SA-like column found (expected one of: SA, sa_ertl, sa).")


def compute_lite_score(df: pd.DataFrame) -> pd.Series:
    if "QED" not in df.columns:
        raise ValueError("Dataframe must contain 'QED' column.")

    qed_norm = min_max_norm(df["QED"])
    sa_series = _pick_sa_column(df)
    sa_norm = min_max_norm(sa_series)

    # LiteGov: prefer high QED and low SA
    return 0.8 * qed_norm + 0.2 * (1.0 - sa_norm)


def main() -> None:
    files = sorted(glob.glob(IN_GLOB))
    if not files:
        print(f"No input files matched {IN_GLOB}")
        return

    rows = []

    for f in files:
        t0 = time.time()
        df_raw = pd.read_csv(f)
        if df_raw.empty:
            print(f"WARNING {f}: empty dataframe, skipping.")
            continue

        if "QED" not in df_raw.columns:
            print(f"WARNING {f}: missing QED column, skipping.")
            continue
        try:
            _ = _pick_sa_column(df_raw)
        except ValueError as e:
            print(f"WARNING {f}: {e} Skipping.")
            continue

        # We assume RDKit descriptors are already computed; we do not touch them.
        df = df_raw.copy()
        n_raw = len(df)

        # Compute LiteGov score and select Top-K
        df["litegov_score"] = compute_lite_score(df)
        df_out = df.sort_values("litegov_score", ascending=False).head(K).copy()

        # Save Top-K file
        out_name = os.path.basename(f).replace("filter_", "filter_rank_")
        out_path = os.path.join(OUT_DIR, out_name)
        df_out.to_csv(out_path, index=False)

        # Metrics
        n = len(df_out)
        validity = 1.0 if n_raw > 0 else 0.0

        s_col = get_smiles_col(df_out)
        if s_col is not None and s_col in df_out.columns and n > 0:
            uniq = df_out[s_col].nunique() / n
        else:
            uniq = np.nan

        if "scaffold" in df_out.columns and n > 0:
            scaf_div = df_out["scaffold"].nunique() / n
        else:
            scaf_div = np.nan

        if n > 0:
            qed_mean = pd.to_numeric(df_out["QED"], errors="coerce").mean()
            sa_series_out = _pick_sa_column(df_out)
            sa_mean = pd.to_numeric(sa_series_out, errors="coerce").mean()
        else:
            qed_mean = np.nan
            sa_mean = np.nan

        sec = time.time() - t0

        if np.isfinite(uniq):
            uniq_disp = f"{uniq:.3f}"
        else:
            uniq_disp = "nan"

        if np.isfinite(scaf_div):
            scaf_disp = f"{scaf_div:.3f}"
        else:
            scaf_disp = "nan"

        if np.isfinite(qed_mean):
            qed_disp = f"{qed_mean:.3f}"
        else:
            qed_disp = "nan"

        if np.isfinite(sa_mean):
            sa_disp = f"{sa_mean:.3f}"
        else:
            sa_disp = "nan"

        rows.append(
            dict(
                file=os.path.basename(f),
                mode="filter+rank",
                K=K,
                N=n,
                validity=validity,
                uniqueness=uniq,
                scaf_div=scaf_div,
                qed_mean=qed_mean,
                sa_mean=sa_mean,
                seconds=sec,
            )
        )

        print(
            f"OK {os.path.basename(f)} -> {out_name} | "
            f"N={n}, validity={validity:.2f}, "
            f"uniqueness={uniq_disp}, scaf_div={scaf_disp}, "
            f"QED={qed_disp}, SA={sa_disp}, "
            f"time={sec:.2f}s"
        )

    if rows:
        summary_path = os.path.join(OUT_DIR, "summary.csv")
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")
    else:
        print("No summary rows written.")


if __name__ == "__main__":
    main()
