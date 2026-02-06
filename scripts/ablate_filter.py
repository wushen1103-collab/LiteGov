#!/usr/bin/env python3
"""PAINS-only filter ablation.

Assumptions (input CSVs, typically from build_pass_from_genmol.py):
- Columns:
    - 'smiles'      : raw SMILES
    - 'smiles_can'  : canonical SMILES (preferred for uniqueness, optional)
    - 'PAINS'       : 0.0/1.0 flag (preferred) or 'pains'
    - optional: 'scaffold', 'QED', etc.

This script:
  1) keeps only rows with PAINS == 0
  2) preserves all existing columns
  3) writes filtered CSVs to results/denovo/filter/
  4) writes a summary.csv with metrics:
       - file, mode="filter"
       - validity   : 1.0 (we do not re-run RDKit here)
       - uniqueness : based on smiles_can if present, else smiles
       - scaf_div   : scaffold diversity
       - qed_mean   : mean QED if available
       - pains_rate : 0.0 (after PAINS filtering)
       - seconds    : processing time
"""

import glob
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

OUT_DIR = "results/denovo/filter"
os.makedirs(OUT_DIR, exist_ok=True)


def get_pains_col(df: pd.DataFrame) -> str:
    """Return the PAINS column name, preferring 'PAINS' then 'pains'."""
    if "PAINS" in df.columns:
        return "PAINS"
    if "pains" in df.columns:
        return "pains"
    raise ValueError("No PAINS / pains column found in dataframe.")


def get_smiles_col(df: pd.DataFrame) -> Optional[str]:
    """Return a sensible smiles column, preferring canonical SMILES."""
    for c in [
        "smiles_can",        # preferred canonical SMILES
        "smiles",
        "SMILES",
        "Smiles",
        "SMILE",
        "SMILEs",
        "canonical_smiles",  # legacy naming
        "smile",
    ]:
        if c in df.columns:
            return c
    # not fatal, only affects uniqueness metric
    return None


def main() -> None:
    rows = []

    files = sorted(glob.glob("results/denovo/pass/*.csv"))
    if not files:
        print("No input files matched results/denovo/pass/*.csv")
    else:
        print(f"Found {len(files)} pass file(s).")

    for f in files:
        t0 = time.time()
        df = pd.read_csv(f)
        if df.empty:
            print(f"Skip {f} (empty dataframe)")
            continue

        try:
            pains_col = get_pains_col(df)
        except ValueError as e:
            print(f"Skip {f}: {e}")
            continue

        # Make sure PAINS is numeric for comparison
        df[pains_col] = pd.to_numeric(df[pains_col], errors="coerce")

        s_col = get_smiles_col(df)

        # Filter: PAINS == 0
        filtered = df[df[pains_col] == 0].copy()
        n = len(filtered)

        out_basename = os.path.basename(f).replace("pass_", "filter_")
        out_path = os.path.join(OUT_DIR, out_basename)
        filtered.to_csv(out_path, index=False)

        if n:
            if s_col is not None and s_col in filtered.columns:
                uniq = filtered[s_col].nunique() / n
            elif "smiles" in filtered.columns:
                uniq = filtered["smiles"].nunique() / n
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

        if np.isfinite(uniq):
            uniq_disp = f"{uniq:.3f}"
        else:
            uniq_disp = "nan"

        rows.append(
            {
                "file": os.path.basename(f),
                "mode": "filter",
                "validity": 1.0,      # RDKit is not re-run here
                "uniqueness": uniq,
                "scaf_div": scaf_div,
                "qed_mean": qed_mean,
                "pains_rate": 0.0,    # all PAINS hits removed
                "seconds": seconds,
            }
        )

        print(
            f"-> {out_path} | "
            f"N={n}, uniqueness={uniq_disp}, "
            f"time={seconds:.2f}s"
        )

    if rows:
        summary_path = os.path.join(OUT_DIR, "summary.csv")
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"All done! summary -> {summary_path}")
    else:
        print("No summary rows written (no files processed).")


if __name__ == "__main__":
    main()
