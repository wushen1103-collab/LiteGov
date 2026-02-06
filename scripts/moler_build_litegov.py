#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LiteGov baseline for MoLeR, aligned with GenMol LiteGov.

Input:
  results/moler/filter/*.csv  (PAINS-only filter set, multiple files)

Output:
  results/moler/lite/lite_<basename>.csv  (per-input-file LiteGov baseline)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


K = 100  # Top-K for LiteGov per file


def get_smiles_col(df: pd.DataFrame) -> Optional[str]:
    for c in [
        "smiles_can",
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
    return None


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
    if "SA" in df.columns:
        return df["SA"]
    if "sa_ertl" in df.columns:
        return df["sa_ertl"]
    if "sa" in df.columns:
        return df["sa"]
    raise ValueError("No SA-like column found (expected one of: SA, sa_ertl, sa).")


def compute_lite_score(df: pd.DataFrame) -> pd.Series:
    if "QED" not in df.columns:
        raise ValueError("DataFrame must contain 'QED' column.")
    qed_norm = min_max_norm(df["QED"])
    sa_series = _pick_sa_column(df)
    sa_norm = min_max_norm(sa_series)
    return 0.8 * qed_norm + 0.2 * (1.0 - sa_norm)


def main():
    root = Path(__file__).resolve().parents[1]
    moler_root = root / "results" / "moler"
    filter_dir = moler_root / "filter"
    out_dir = moler_root / "lite"
    out_dir.mkdir(parents=True, exist_ok=True)

    filter_files = sorted(filter_dir.glob("*.csv"))
    if not filter_files:
        raise SystemExit(f"[ERROR] No filter CSV files found in {filter_dir}")

    print(f"[INFO] Found {len(filter_files)} filter CSV files in {filter_dir}")

    for filter_path in filter_files:
        print(f"[INFO] Reading MoLeR filter-only set from {filter_path}")
        df_raw = pd.read_csv(filter_path)
        if df_raw.empty:
            print(f"[WARN] Skipping empty filter dataframe: {filter_path.name}")
            continue

        df = df_raw.copy()
        df["lite_score"] = compute_lite_score(df)
        df_sorted = df.sort_values("lite_score", ascending=False).reset_index(drop=True)

        if K is not None and K > 0:
            df_top = df_sorted.head(K).copy()
        else:
            df_top = df_sorted.copy()

        df_top["method"] = "lite"

        out_path = out_dir / f"lite_{filter_path.name}"
        df_top.to_csv(out_path, index=False)
        print(f"[OK] wrote LiteGov baseline to {out_path} (K={len(df_top)})")


if __name__ == "__main__":
    main()
