#!/usr/bin/env python

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "tables"

GENMOL_DIR = ROOT / "results" / "genmol"
MOLER_DIR = ROOT / "results" / "moler"

METHODS = ["raw", "qed", "rulekit", "lite", "heavy"]


PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "GenMol": {
        "raw": ["*pass_genmol_*.csv"],
        "qed": ["*qed_genmol_*.csv"],
        "rulekit": ["*rulekit_genmol_*.csv"],
        "lite": ["*filter_rank_genmol_*.csv"],
        "heavy": ["*govern_filter_genmol_*.csv"],
    },
    "MoLeR": {
        "raw": ["*pass_moler_*.csv", "*raw_moler_*.csv"],
        "qed": ["*qed_moler_*.csv"],
        "rulekit": ["*rulekit_moler_*.csv"],
        "lite": ["*filter_rank_moler_*.csv", "*lite_moler_*.csv"],
        "heavy": ["*govern_filter_moler_*.csv", "*heavy_moler_*.csv"],
    },
}


def find_csvs(root: Path, generator: str, method: str) -> List[Path]:
    patterns = PATTERNS.get(generator, {}).get(method, [])
    matches: List[Path] = []
    for pat in patterns:
        matches.extend(root.rglob(pat))
    if not matches:
        matches = list(root.rglob(f"*{method}*.csv"))
    return matches


def load_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = str(path)
    return df


def aggregate_metrics(df: pd.DataFrame, generator: str, method: str) -> Dict[str, object]:
    row: Dict[str, object] = {"generator": generator, "method": method}
    n = len(df)
    row["n_molecules"] = int(n)

    metric_map = {
        "qed": ["qed", "QED"],
        "sa": ["SA", "sa", "sa_score", "sa_ertl"],
        "logp": ["logP", "logp"],
        "mw": ["mw", "MolecularWeight"],
        "tpsa": ["tpsa"],
        "hba": ["hba"],
        "hbd": ["hbd"],
        "rtb": ["rtb"],
        "frac_csp3": ["frac_csp3"],
        "rings_aromatic": ["rings_aromatic"],
        "pains_n": ["pains_n"],
    }

    def find_column(sub_df: pd.DataFrame, candidates):
        cols = sub_df.columns
        for c in candidates:
            if c in cols:
                return c
        return None

    for out_name, candidates in metric_map.items():
        col = find_column(df, candidates)
        if col is None:
            row[f"{out_name}_mean"] = np.nan
            row[f"{out_name}_median"] = np.nan
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            row[f"{out_name}_mean"] = np.nan
            row[f"{out_name}_median"] = np.nan
        else:
            row[f"{out_name}_mean"] = float(series.mean())
            row[f"{out_name}_median"] = float(series.median())

    return row


def process_generator(root: Path, generator_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method in METHODS:
        csvs = find_csvs(root, generator_name, method)
        if not csvs:
            print(f"[WARN] No CSVs found for {generator_name} / {method} in {root}")
            continue

        dfs = [load_one_csv(p) for p in csvs]
        df_all = pd.concat(dfs, ignore_index=True)

        row = aggregate_metrics(df_all, generator_name, method)
        rows.append(row)

        print(f"[INFO] {generator_name}/{method}: loaded {len(df_all)} rows from {len(csvs)} files")

    return rows


def main():
    TABLE_DIR.mkdir(exist_ok=True)

    all_rows: List[Dict[str, object]] = []

    if GENMOL_DIR.exists():
        print(f"[INFO] Processing GenMol from {GENMOL_DIR}")
        all_rows.extend(process_generator(GENMOL_DIR, "GenMol"))
    else:
        print("[WARN] GenMol directory not found:", GENMOL_DIR)

    if MOLER_DIR.exists():
        print(f"[INFO] Processing MoLeR from {MOLER_DIR}")
        all_rows.extend(process_generator(MOLER_DIR, "MoLeR"))
    else:
        print("[WARN] MoLeR directory not found:", MOLER_DIR)

    if not all_rows:
        print("[ERROR] No rows aggregated; check paths and patterns.")
        return

    df = pd.DataFrame(all_rows)

    method_order = {"raw": 0, "qed": 1, "rulekit": 2, "lite": 3, "heavy": 4}
    df["method_order"] = df["method"].map(method_order).fillna(99)
    df = df.sort_values(["generator", "method_order", "method"])
    df = df.drop(columns=["method_order"])

    keep_cols = [
        "generator",
        "method",
        "n_molecules",
        "qed_mean",
        "sa_mean",
        "logp_mean",
        "mw_mean",
        "tpsa_mean",
        "frac_csp3_mean",
        "rings_aromatic_mean",
        "pains_n_mean",
    ]
    available = [c for c in keep_cols if c in df.columns]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        print("[WARN] The following columns are missing and will be skipped in output:")
        for c in missing:
            print("  -", c)

    df_compact = df[available]

    out_csv = TABLE_DIR / "table2_global_druglikeness_metrics.csv"
    df_compact.to_csv(out_csv, index=False)

    print(f"[OK] Wrote compact Table 2 CSV → {out_csv}")
    print("[INFO] Preview:")
    print(df_compact.to_string(index=False))


if __name__ == "__main__":
    main()
