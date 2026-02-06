#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HeavyGov baseline for MoLeR, aligned with GenMol gov_baseline.py.

- Input:
    results/moler/filter/*.csv

- Output:
    results/moler/heavy/heavy_<basename>.csv  # sorted by heavy_score (desc), all rows
    results/moler/heavy/summary.csv           # per-file metrics

The scoring and thresholds are identical to GenMol side gov_baseline.py:
    gov_score = w_qed * QED - w_sa * SA - w_logp * |logP - target_logp|
with:
    sa_threshold = 4.0
    qed_floor    = 0.4
    w_qed        = 1.0
    w_sa         = 1.0
    w_logp       = 0.5
    target_logp  = 2.0
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def safe_float(x) -> float:
    """Convert to float, returning NaN on failure."""
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def gov_score(
    qed: float,
    sa: float,
    logp: float,
    w_qed: float,
    w_sa: float,
    w_logp: float,
    target_logp: float,
) -> float:
    """
    Governance score (higher is better).

    - QED is rewarded
    - SA is penalized
    - deviation from target logP is penalized
    """
    qed = safe_float(qed)
    sa = safe_float(sa)
    logp = safe_float(logp)

    qed_term = w_qed * qed if np.isfinite(qed) else 0.0
    sa_pen = w_sa * sa if np.isfinite(sa) else 0.0

    if np.isfinite(logp):
        logp_pen = w_logp * abs(logp - target_logp)
    else:
        logp_pen = 0.0

    return qed_term - sa_pen - logp_pen


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute validity, uniqueness, scaffold diversity, QED mean,
    SA mean, PAINS rate from a dataframe that already contains
    descriptors and canonical smiles.

    Expected columns:
      - 'smiles_can' (preferred) or 'smiles'
      - 'scaffold' (optional)
      - 'QED'
      - 'SA'
      - one of: 'n_pains', 'pains_n', 'PAINS' (optional)
    """
    n_total = len(df)
    if n_total == 0:
        return {
            "K": 0,
            "N": 0,
            "validity": 0.0,
            "uniqueness": 0.0,
            "scaf_div": 0.0,
            "qed_mean": float("nan"),
            "sa_mean": float("nan"),
            "pains_rate": 0.0,
        }

    if "smiles_can" in df.columns:
        valid_mask = df["smiles_can"].notna()
    elif "smiles" in df.columns:
        valid_mask = df["smiles"].notna()
    else:
        valid_mask = pd.Series(False, index=df.index)

    n_valid = int(valid_mask.sum())
    validity = n_valid / float(n_total) if n_total > 0 else 0.0

    if n_valid == 0:
        return {
            "K": 0,
            "N": 0,
            "validity": validity,
            "uniqueness": 0.0,
            "scaf_div": 0.0,
            "qed_mean": float("nan"),
            "sa_mean": float("nan"),
            "pains_rate": 0.0,
        }

    df_valid = df[valid_mask].copy()
    n = len(df_valid)

    if "smiles_can" in df_valid.columns:
        n_unique = df_valid["smiles_can"].nunique(dropna=True)
    elif "smiles" in df_valid.columns:
        n_unique = df_valid["smiles"].nunique(dropna=True)
    else:
        n_unique = 0
    uniqueness = n_unique / float(n) if n > 0 else 0.0

    if "scaffold" in df_valid.columns:
        n_scaf = df_valid["scaffold"].nunique(dropna=True)
        scaf_div = n_scaf / float(n) if n > 0 else 0.0
    else:
        scaf_div = 0.0

    qed_mean = float(pd.to_numeric(df_valid["QED"], errors="coerce").mean())
    sa_mean = float(pd.to_numeric(df_valid["SA"], errors="coerce").mean())

    if "n_pains" in df_valid.columns:
        pains_n = pd.to_numeric(df_valid["n_pains"], errors="coerce")
        pains_hit = pains_n > 0
        pains_rate = float(pains_hit.mean())
    elif "pains_n" in df_valid.columns:
        pains_n = pd.to_numeric(df_valid["pains_n"], errors="coerce")
        pains_hit = pains_n > 0
        pains_rate = float(pains_hit.mean())
    elif "PAINS" in df_valid.columns:
        pains_rate = float(pd.to_numeric(df_valid["PAINS"], errors="coerce").mean())
    else:
        pains_rate = 0.0

    return {
        "K": n,
        "N": n,
        "validity": validity,
        "uniqueness": uniqueness,
        "scaf_div": scaf_div,
        "qed_mean": qed_mean,
        "sa_mean": sa_mean,
        "pains_rate": pains_rate,
    }


def process_file(
    path: Path,
    outdir: Path,
    sa_threshold: float,
    qed_floor: float,
    w_qed: float,
    w_sa: float,
    w_logp: float,
    target_logp: float,
) -> Dict[str, object]:
    """
    Apply HeavyGov baseline on a filter-layer MoLeR table.

    Input:
      - path: CSV from results/moler/filter

    Output:
      - heavy_<basename>.csv in outdir, with all rows sorted by heavy_score
    """
    df = pd.read_csv(path)
    if "smiles" not in df.columns and "smiles_can" not in df.columns:
        raise ValueError(
            f"Input file {path} must contain 'smiles' or 'smiles_can' column."
        )

    for col in ["QED", "SA", "logP", "PAINS", "pains_n", "n_pains"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "SA" in df.columns:
        df = df[df["SA"].isna() | (df["SA"] <= sa_threshold)]
    if "QED" in df.columns:
        df = df[df["QED"].isna() | (df["QED"] >= qed_floor)]

    scores: List[float] = []
    for _, row in df.iterrows():
        sc = gov_score(
            row.get("QED", np.nan),
            row.get("SA", np.nan),
            row.get("logP", np.nan),
            w_qed,
            w_sa,
            w_logp,
            target_logp,
        )
        scores.append(sc)
    df["heavy_score"] = scores
    df["method"] = "heavy"

    df_sorted = df.sort_values("heavy_score", ascending=False).reset_index(drop=True)

    metrics = compute_metrics(df_sorted)

    base = path.name
    out_name = f"heavy_{base}"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / out_name
    df_sorted.to_csv(out_path, index=False)

    summary_row: Dict[str, object] = {
        "file": out_name,
        "mode": "heavy",
        "K": metrics["K"],
        "N": metrics["N"],
        "validity": metrics["validity"],
        "uniqueness": metrics["uniqueness"],
        "scaf_div": metrics["scaf_div"],
        "qed_mean": metrics["qed_mean"],
        "sa_mean": metrics["sa_mean"],
        "pains_rate": metrics["pains_rate"],
    }

    return summary_row


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    moler_root = root / "results" / "moler"
    filter_dir = moler_root / "filter"
    outdir = moler_root / "heavy"

    sa_threshold = 4.0
    qed_floor = 0.4
    w_qed = 1.0
    w_sa = 1.0
    w_logp = 0.5
    target_logp = 2.0

    paths = sorted(filter_dir.glob("*.csv"))
    if not paths:
        raise SystemExit(f"No filter CSV files found in: {filter_dir}")

    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    print(f"Found {len(paths)} input file(s) in {filter_dir}.")
    for p in paths:
        print(f"Processing: {p.name}")
        row = process_file(
            path=p,
            outdir=outdir,
            sa_threshold=sa_threshold,
            qed_floor=qed_floor,
            w_qed=w_qed,
            w_sa=w_sa,
            w_logp=w_logp,
            target_logp=target_logp,
        )
        summary_rows.append(row)

        qed_mean = row["qed_mean"]
        sa_mean = row["sa_mean"]
        pains_rate = row["pains_rate"]

        if np.isfinite(qed_mean):
            qed_disp = f"{qed_mean:.3f}"
        else:
            qed_disp = "nan"

        if np.isfinite(sa_mean):
            sa_disp = f"{sa_mean:.3f}"
        else:
            sa_disp = "nan"

        if np.isfinite(pains_rate):
            pains_disp = f"{pains_rate:.3f}"
        else:
            pains_disp = "nan"

        print(
            f"  -> K={row['K']} N={row['N']} "
            f"qed_mean={qed_disp} sa_mean={sa_disp} "
            f"pains_rate={pains_disp}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
