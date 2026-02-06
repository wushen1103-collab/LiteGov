#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoLeR PAINS-only filter, aligned with GenMol ablate_filter.py.

Input:
  results/moler/raw/*.csv   (excluding summary.csv)

Output:
  results/moler/filter/<input_name>.filtered.csv
  results/moler/filter/summary.csv
"""

import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd


def get_pains_col(df: pd.DataFrame) -> str:
    """Return the PAINS column name, prefer 'n_pains' then 'pains_n' then 'PAINS'."""
    if "n_pains" in df.columns:
        return "n_pains"
    if "pains_n" in df.columns:
        return "pains_n"
    if "PAINS" in df.columns:
        return "PAINS"
    raise ValueError("No PAINS-like column found (expected one of: n_pains, pains_n, PAINS).")


def get_smiles_col(df: pd.DataFrame) -> Optional[str]:
    """Return a sensible smiles column, preferring canonical SMILES."""
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


def process_single_file(in_path: Path, out_dir: Path) -> Dict[str, Any]:
    t0 = time.time()
    df = pd.read_csv(in_path)
    if df.empty:
        print(f"[WARN] {in_path} is empty, skip.")
        return {
            "file": in_path.name,
            "mode": "filter",
            "N": 0,
            "validity": 1.0,
            "uniqueness": 0.0,
            "scaf_div": np.nan,
            "qed_mean": np.nan,
            "pains_rate": 0.0,
            "seconds": time.time() - t0,
        }

    try:
        pains_col = get_pains_col(df)
    except ValueError as e:
        print(f"[ERROR] {in_path}: {e}")
        return {
            "file": in_path.name,
            "mode": "filter",
            "N": 0,
            "validity": 1.0,
            "uniqueness": 0.0,
            "scaf_div": np.nan,
            "qed_mean": np.nan,
            "pains_rate": np.nan,
            "seconds": time.time() - t0,
        }

    df[pains_col] = pd.to_numeric(df[pains_col], errors="coerce")
    s_col = get_smiles_col(df)

    filtered = df[df[pains_col] == 0].copy()
    n = len(filtered)

    out_name = f"{in_path.stem}.filtered.csv"
    out_path = out_dir / out_name
    filtered.to_csv(out_path, index=False)

    if n:
        if s_col is not None and s_col in filtered.columns:
            uniq = filtered[s_col].nunique() / n
        else:
            uniq = np.nan

        scaf_div = (
            filtered["scaffold"].nunique() / n
            if "scaffold" in filtered.columns
            else np.nan
        )
        qed_mean = (
            pd.to_numeric(filtered["QED"], errors="coerce").mean()
            if "QED" in filtered.columns
            else np.nan
        )
    else:
        uniq = 0.0
        scaf_div = np.nan
        qed_mean = np.nan

    seconds = time.time() - t0

    print(
        f"-> {out_path} | file={in_path.name} N={n}, "
        f"uniqueness={uniq if np.isfinite(uniq) else 'nan'} "
        f"time={seconds:.2f}s"
    )

    return {
        "file": in_path.name,
        "mode": "filter",
        "N": n,
        "validity": 1.0,
        "uniqueness": uniq,
        "scaf_div": scaf_div,
        "qed_mean": qed_mean,
        "pains_rate": 0.0,  # all PAINS hits removed
        "seconds": seconds,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    in_dir = root / "results" / "moler" / "raw"
    out_dir = root / "results" / "moler" / "filter"
    os.makedirs(out_dir, exist_ok=True)

    if not in_dir.exists():
        print(f"[ERROR] input directory not found: {in_dir}")
        return

    csv_files: List[Path] = sorted(in_dir.glob("*.csv"))
    csv_files = [p for p in csv_files if p.name.lower() != "summary.csv"]

    if not csv_files:
        print(f"[ERROR] no input csv files found in {in_dir}")
        return

    rows: List[Dict[str, Any]] = []

    print(f"[INFO] found {len(csv_files)} input csv files in {in_dir}")
    for fp in csv_files:
        print(f"[INFO] processing {fp}")
        row = process_single_file(fp, out_dir)
        rows.append(row)

    summary_path = out_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[OK] summary -> {summary_path}")


if __name__ == "__main__":
    main()
