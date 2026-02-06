#!/usr/bin/env python

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = ROOT / "docking" / "stats"
LIG_FILE = ROOT / "docking" / "ligands" / "ligands_full_metadata.csv"
TABLE_DIR = ROOT / "tables"

SCORE_FILES = {
    "GenMol": STATS_DIR / "dock_scores_all.csv",
    "MoLeR": STATS_DIR / "dock_scores_all_moler.csv",
}

METHOD_ORDER = {"raw": 0, "qed": 1, "rulekit": 2, "lite": 3, "heavy": 4}


def load_scores(path: Path, gen: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["generator"] = gen
    return df


def load_ligands() -> pd.DataFrame:
    df = pd.read_csv(LIG_FILE)
    return df


def summarize_top100(scores: pd.DataFrame, ligs: pd.DataFrame, gen: str, k: int = 100) -> pd.DataFrame:
    scores = scores.copy()
    ligs = ligs.copy()

    merged = pd.merge(
        scores,
        ligs,
        on=["generator", "ligand_id"],
        how="left",
        suffixes=("", "_meta"),
    )

    rows = []
    for method, sub in merged.groupby("method"):
        sub = sub.sort_values("score")
        top = sub.head(k)
        n_top = len(top)

        scores_top = pd.to_numeric(top["score"], errors="coerce").dropna()

        row = {
            "generator": gen,
            "method": method,
            "topk": k,
            "n_top": int(n_top),
            "score_mean": float(scores_top.mean()) if len(scores_top) else np.nan,
            "score_median": float(scores_top.median()) if len(scores_top) else np.nan,
        }

        def mean_of(col: str):
            if col not in top.columns:
                return np.nan
            s = pd.to_numeric(top[col], errors="coerce").dropna()
            return float(s.mean()) if len(s) else np.nan

        row["qed_mean"] = mean_of("qed")
        if "SA" in top.columns:
            row["sa_mean"] = mean_of("SA")
        elif "sa" in top.columns:
            row["sa_mean"] = mean_of("sa")
        else:
            row["sa_mean"] = mean_of("sa_ertl")

        row["mw_mean"] = mean_of("mw")
        row["tpsa_mean"] = mean_of("tpsa")
        row["frac_csp3_mean"] = mean_of("frac_csp3")
        row["rings_aromatic_mean"] = mean_of("rings_aromatic")

        rows.append(row)

    out = pd.DataFrame(rows)
    out["method_order"] = out["method"].map(METHOD_ORDER).fillna(99)
    out = out.sort_values(["generator", "method_order"]).drop(columns=["method_order"])

    return out


def main():
    TABLE_DIR.mkdir(exist_ok=True)

    ligs = load_ligands()

    all_results = []

    for gen, path in SCORE_FILES.items():
        if not path.exists():
            print(f"[WARN] {path} not found, skip {gen}")
            continue

        print(f"[INFO] Loading scores for {gen} from {path}")
        scores = load_scores(path, gen)
        tbl = summarize_top100(scores, ligs, gen, k=100)
        all_results.append(tbl)

    if not all_results:
        print("[ERROR] No results generated.")
        return

    final = pd.concat(all_results, ignore_index=True)

    out_path = TABLE_DIR / "table3_top100_properties.csv"
    final.to_csv(out_path, index=False)

    print(f"[OK] Wrote Table 3 to {out_path}")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
