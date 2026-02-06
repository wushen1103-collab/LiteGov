#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List
import pandas as pd


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Return the first column name that exists in df.columns from the candidates list.
    Raise ValueError if none of them is present.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of candidate columns {candidates} found in {list(df.columns)}")


def apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the shared \"common filters\" used for QED and RuleKit baselines.
    This is intentionally aligned with the GenMol pipeline:

    - QED >= 0.2
    - SA <= 7.0
    - 150 <= MW <= 650
    - -1.0 <= logP <= 6.0
    - TPSA <= 180.0
    - RTB <= 12
    - Lipinski violations <= 2
    - No PAINS hit (if PAINS columns are available)
    """
    df = df.copy()

    # Core physchem columns
    q_col = find_column(df, ["QED", "qed"])
    sa_col = find_column(df, ["SA", "sa_ertl", "sa"])
    mw_col = find_column(df, ["MW", "mw", "MolWt", "molwt"])
    logp_col = find_column(df, ["logP", "LogP", "logp", "LOGP"])
    tpsa_col = find_column(df, ["TPSA", "tpsa"])
    hba_col = find_column(df, ["HBA", "hba"])
    hbd_col = find_column(df, ["HBD", "hbd"])
    rtb_col = find_column(df, ["RTB", "rtb", "rotatable_bonds", "RotBonds"])

    # Cast to numeric
    for col in [q_col, sa_col, mw_col, logp_col, tpsa_col, hba_col, hbd_col, rtb_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute Lipinski violations (recomputed for consistency)
    lip_col = "lipinski_violations_calc"
    df[lip_col] = 0
    df[lip_col] += (df[mw_col] > 500).astype(int)
    df[lip_col] += (df[logp_col] > 5).astype(int)
    df[lip_col] += (df[hbd_col] > 5).astype(int)
    df[lip_col] += (df[hba_col] > 10).astype(int)

    # PAINS handling (if any PAINS column exists)
    pains_cols = [c for c in ["n_pains", "pains_n", "PAINS", "pains"] if c in df.columns]
    pains_ok = None
    if pains_cols:
        p_col = pains_cols[0]
        df[p_col] = pd.to_numeric(df[p_col], errors="coerce").fillna(0)
        pains_ok = (df[p_col] == 0)
    elif "pains_ok" in df.columns:
        pains_ok = df["pains_ok"].astype(bool)

    # Build common filter mask
    mask = (
        (df[q_col] >= 0.2)
        & (df[sa_col] <= 7.0)
        & (df[mw_col] >= 150.0)
        & (df[mw_col] <= 650.0)
        & (df[logp_col] >= -1.0)
        & (df[logp_col] <= 6.0)
        & (df[tpsa_col] <= 180.0)
        & (df[rtb_col] <= 12)
        & (df[lip_col] <= 2)
    )
    if pains_ok is not None:
        mask &= pains_ok

    return df.loc[mask].copy()


def build_qed_baseline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    QED baseline:
    - Apply the shared common filters.
    - Sort by QED (descending).
    """
    df = apply_common_filters(df_raw)
    if df.empty:
        return df

    q_col = find_column(df, ["QED", "qed"])
    df = df.sort_values(q_col, ascending=False).reset_index(drop=True)
    df["method"] = "qed"
    df["rank_qed"] = df.index + 1
    return df


def build_rulekit_baseline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    RuleKit baseline (handcrafted rule set, aligned with the GenMol implementation).

    Design:
    - Start from the same common filters as QED baseline.
    - Apply a stricter rule set:
        QED >= 0.4
        SA <= 5.0
        0.0 <= logP <= 5.0
        TPSA <= 140.0
        RTB <= 10
    - Sort by QED (descending).

    This does NOT use any external \"rulekit_score\" field; instead, it encodes
    the rule set directly via hard thresholds so that both GenMol and MoLeR
    baselines are exactly comparable.
    """
    df = apply_common_filters(df_raw)
    if df.empty:
        return df

    q_col = find_column(df, ["QED", "qed"])
    sa_col = find_column(df, ["SA", "sa_ertl", "sa"])
    logp_col = find_column(df, ["logP", "LogP", "logp", "LOGP"])
    tpsa_col = find_column(df, ["TPSA", "tpsa"])
    rtb_col = find_column(df, ["RTB", "rtb", "rotatable_bonds", "RotBonds"])

    # Ensure numeric types for stricter thresholds
    for col in [q_col, sa_col, logp_col, tpsa_col, rtb_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rule_mask = (
        (df[q_col] >= 0.4)
        & (df[sa_col] <= 5.0)
        & (df[logp_col] >= 0.0)
        & (df[logp_col] <= 5.0)
        & (df[tpsa_col] <= 140.0)
        & (df[rtb_col] <= 10)
    )
    df = df.loc[rule_mask].copy()
    if df.empty:
        return df

    df = df.sort_values(q_col, ascending=False).reset_index(drop=True)
    df["method"] = "rulekit"
    df["rank_rulekit"] = df.index + 1
    return df


def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "results" / "moler" / "raw"
    qed_dir = root / "results" / "moler" / "qed"
    rule_dir = root / "results" / "moler" / "rulekit"

    qed_dir.mkdir(parents=True, exist_ok=True)
    rule_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(p for p in raw_dir.glob("*.csv") if p.name != "summary.csv")
    if not raw_files:
        print(f"[WARN] No raw CSV files found in {raw_dir}")
        return

    for raw_path in raw_files:
        stem = raw_path.stem
        print(f"[INFO] Processing {raw_path.name}")

        df = pd.read_csv(raw_path)

        # QED baseline
        try:
            df_qed = build_qed_baseline(df)
            out_qed = qed_dir / f"{stem}_qed.csv"
            df_qed.to_csv(out_qed, index=False)
            print(f"[OK] wrote {out_qed.name} (n={len(df_qed)})")
        except Exception as e:
            print(f"[WARN] Failed QED baseline for {raw_path.name}: {e}")

        # RuleKit baseline (handcrafted rules, aligned with GenMol)
        try:
            df_rule = build_rulekit_baseline(df)
            out_rule = rule_dir / f"{stem}_rulekit.csv"
            df_rule.to_csv(out_rule, index=False)
            print(f"[OK] wrote {out_rule.name} (n={len(df_rule)})")
        except Exception as e:
            print(f"[WARN] Failed RuleKit baseline for {raw_path.name}: {e}")


if __name__ == "__main__":
    main()
