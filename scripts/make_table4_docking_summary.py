#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCORES_GENMOL = ROOT / "docking" / "stats" / "dock_scores_all.csv"
SCORES_MOLER = ROOT / "docking" / "stats" / "dock_scores_all_moler.csv"

OUT_CSV = ROOT / "tables" / "table4_docking_summary.csv"


def load_scores() -> pd.DataFrame:
    """Load and combine docking scores from GenMol and MoLeR."""
    df_g = pd.read_csv(SCORES_GENMOL)
    df_m = pd.read_csv(SCORES_MOLER)

    df_g["generator"] = "GenMol"
    df_m["generator"] = "MoLeR"

    needed_cols = ["ligand_id", "target", "method", "score", "generator"]
    df_g = df_g[needed_cols].copy()
    df_m = df_m[needed_cols].copy()

    df = pd.concat([df_g, df_m], ignore_index=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    print("[INFO] Combined scores shape:", df.shape)
    print("[INFO] Targets:", sorted(df["target"].unique()))
    print("[INFO] Methods:", sorted(df["method"].unique()))
    print("[INFO] Generators:", sorted(df["generator"].unique()))
    return df


def summarize_group(df: pd.DataFrame) -> pd.Series:
    """
    Summarize one (generator, method, target) group.

    We treat score == 0 as "invalid docking" (e.g., failed job, no pose).
    n_total: all rows
    n_valid: rows with non-zero, non-NaN scores
    If n_valid == 0, all metrics are set to NA.
    """
    scores = df["score"]
    valid_mask = scores.notna() & (scores != 0.0)
    n_total = int(len(df))
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        # No usable docking scores for this target/method
        return pd.Series(
            {
                "n_total": n_total,
                "n_valid": 0,
                "score_mean": pd.NA,
                "score_std": pd.NA,
                "hit_rate_le_-6.0": pd.NA,
                "hit_rate_le_-6.5": pd.NA,
                "top1_score": pd.NA,
                "top1_ligand": pd.NA,
            }
        )

    df_valid = df.loc[valid_mask].copy()
    scores_valid = df_valid["score"]

    hit6 = (scores_valid <= -6.0).mean()
    hit65 = (scores_valid <= -6.5).mean()

    idx_min = scores_valid.idxmin()
    top1_row = df_valid.loc[idx_min]
    top1_score = float(top1_row["score"])
    top1_ligand = str(top1_row["ligand_id"])

    return pd.Series(
        {
            "n_total": n_total,
            "n_valid": n_valid,
            "score_mean": scores_valid.mean(),
            "score_std": scores_valid.std(),
            "hit_rate_le_-6.0": hit6,
            "hit_rate_le_-6.5": hit65,
            "top1_score": top1_score,
            "top1_ligand": top1_ligand,
        }
    )


def build_table4():
    df = load_scores()

    grouped = (
        df.groupby(["generator", "method", "target"])
        .apply(summarize_group)
        .reset_index()
    )

    # Sort for readability
    grouped = grouped.sort_values(
        by=["generator", "target", "method"],
        ascending=[True, True, True],
    )

    # Round numeric columns
    for col in [
        "score_mean",
        "score_std",
        "hit_rate_le_-6.0",
        "hit_rate_le_-6.5",
        "top1_score",
    ]:
        if col in grouped.columns:
            grouped[col] = grouped[col].astype("Float64").round(3)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(OUT_CSV, index=False)

    print(f"[OK] Wrote Table 4 → {OUT_CSV}")
    print("[INFO] Preview:")
    print(grouped.head(30))


if __name__ == "__main__":
    build_table4()
