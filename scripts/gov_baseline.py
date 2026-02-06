#!/usr/bin/env python

import argparse
import glob
import os
import time
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
      - 'pains_n' or 'PAINS' (optional)
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

    # validity: fraction of rows with a non-null smiles_can (or smiles fallback)
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

    # uniqueness by canonical smiles if available, else smiles
    if "smiles_can" in df_valid.columns:
        n_unique = df_valid["smiles_can"].nunique(dropna=True)
    elif "smiles" in df_valid.columns:
        n_unique = df_valid["smiles"].nunique(dropna=True)
    else:
        n_unique = 0
    uniqueness = n_unique / float(n) if n > 0 else 0.0

    # scaffold diversity
    if "scaffold" in df_valid.columns:
        n_scaf = df_valid["scaffold"].nunique(dropna=True)
        scaf_div = n_scaf / float(n) if n > 0 else 0.0
    else:
        scaf_div = 0.0

    # QED / SA means on valid subset
    qed_mean = float(pd.to_numeric(df_valid["QED"], errors="coerce").mean())
    sa_mean = float(pd.to_numeric(df_valid["SA"], errors="coerce").mean())

    # PAINS rate: fraction with any PAINS hit
    if "pains_n" in df_valid.columns:
        pains_n = pd.to_numeric(df_valid["pains_n"], errors="coerce")
        pains_hit = pains_n > 0
        pains_rate = float(pains_hit.mean())
    elif "PAINS" in df_valid.columns:
        pains_rate = float(pd.to_numeric(df_valid["PAINS"], errors="coerce").mean())
    else:
        pains_rate = 0.0

    return {
        "K": 0,  # filled later with actual K used
        "N": n,
        "validity": validity,
        "uniqueness": uniqueness,
        "scaf_div": scaf_div,
        "qed_mean": qed_mean,
        "sa_mean": sa_mean,
        "pains_rate": pains_rate,
    }


def process_file(
    path: str,
    outdir: str,
    K: int,
    sa_threshold: float,
    qed_floor: float,
    w_qed: float,
    w_sa: float,
    w_logp: float,
    target_logp: float,
) -> Dict[str, object]:
    """Apply governance scoring and compute metrics on an existing descriptor table."""
    t0 = time.time()

    df = pd.read_csv(path)
    if "smiles" not in df.columns and "smiles_can" not in df.columns:
        raise ValueError(
            f"Input file {path} must contain 'smiles' or 'smiles_can' column."
        )

    # cast numeric columns
    for col in ["QED", "SA", "logP", "PAINS", "pains_n"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # apply filters
    if "SA" in df.columns:
        df = df[df["SA"].isna() | (df["SA"] <= sa_threshold)]
    if "QED" in df.columns:
        df = df[df["QED"].isna() | (df["QED"] >= qed_floor)]

    # compute gov_score
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
    df["gov_score"] = scores

    # sort by score descending
    df_sorted = df.sort_values("gov_score", ascending=False).reset_index(drop=True)

    # top-K subset for metrics (if K <= 0, use all)
    if K is not None and K > 0:
        df_top = df_sorted.head(K).copy()
    else:
        df_top = df_sorted.copy()

    metrics = compute_metrics(df_top)
    metrics["K"] = len(df_top)

    seconds = float(time.time() - t0)
    metrics["seconds"] = seconds

    # write per-file govern CSV
    base = os.path.basename(path)
    out_name = "govern_" + base

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, out_name)
    df_sorted.to_csv(out_path, index=False)

    # summary row
    summary_row: Dict[str, object] = {
        "file": out_name,
        "mode": "govern",
        "K": metrics["K"],
        "N": metrics["N"],
        "validity": metrics["validity"],
        "uniqueness": metrics["uniqueness"],
        "scaf_div": metrics["scaf_div"],
        "qed_mean": metrics["qed_mean"],
        "sa_mean": metrics["sa_mean"],
        "pains_rate": metrics["pains_rate"],
        "seconds": metrics["seconds"],
    }

    return summary_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply governance baseline (QED/SA/logP-based) "
            "on filtered molecules using existing descriptors."
        )
    )
    parser.add_argument(
        "--in-glob",
        type=str,
        default="results/denovo/filter/filter_genmol_tau*_r1.0_seed*.csv",
        help="Glob pattern for input CSV files (typically filter outputs).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/denovo/govern",
        help="Output directory for govern CSV files and summary.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1000,
        help="Number of top-scoring molecules to keep per file for metrics (<=0 means all).",
    )
    parser.add_argument(
        "--sa-threshold",
        type=float,
        default=4.0,
        help="Maximum allowed SA score; rows with SA > threshold are filtered out.",
    )
    parser.add_argument(
        "--qed-floor",
        type=float,
        default=0.4,
        help="Minimum allowed QED; rows with QED < floor are filtered out.",
    )
    parser.add_argument(
        "--w-qed",
        type=float,
        default=1.0,
        help="Weight for QED in gov_score.",
    )
    parser.add_argument(
        "--w-sa",
        type=float,
        default=1.0,
        help="Weight for SA penalty in gov_score.",
    )
    parser.add_argument(
        "--w-logp",
        type=float,
        default=0.5,
        help="Weight for logP deviation penalty in gov_score.",
    )
    parser.add_argument(
        "--target-logp",
        type=float,
        default=2.0,
        help="Target logP value used in gov_score.",
    )

    args = parser.parse_args()

    paths = sorted(glob.glob(args.in_glob))
    if not paths:
        raise SystemExit(f"No files matched pattern: {args.in_glob}")

    os.makedirs(args.outdir, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    print(f"Found {len(paths)} input file(s).")
    for p in paths:
        print(f"Processing: {p}")
        row = process_file(
            path=p,
            outdir=args.outdir,
            K=args.K,
            sa_threshold=args.sa_threshold,
            qed_floor=args.qed_floor,
            w_qed=args.w_qed,
            w_sa=args.w_sa,
            w_logp=args.w_logp,
            target_logp=args.target_logp,
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
            f"pains_rate={pains_disp} seconds={row['seconds']:.2f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.outdir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
