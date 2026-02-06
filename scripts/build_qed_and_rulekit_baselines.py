#!/usr/bin/env python
from pathlib import Path
import re
import pandas as pd


def find_column(df, candidates):
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
    Common pre-filter used by all baselines (RuleKit, LiteGov, HeavyGov):

    - basic QED floor (very loose)
    - SA not too bad
    - MW in a broad range
    - logP in a broad range
    - remove PAINS
    - enforce Veber-style TPSA/RB soft limits
    - compute Lipinski violations and require <= 2
    """
    q_col = find_column(df, ["QED", "qed"])
    sa_col = find_column(df, ["SA", "sa", "sa_ertl"])
    logp_col = find_column(df, ["logP", "LogP", "logp"])
    mw_col = find_column(df, ["mw", "MW"])
    tpsa_col = find_column(df, ["tpsa", "TPSA"])
    hba_col = find_column(df, ["hba", "HBA"])
    hbd_col = find_column(df, ["hbd", "HBD"])
    rtb_col = find_column(df, ["rtb", "rotatable_bonds"])
    pains_col = find_column(df, ["n_pains", "pains_n", "PAINS", "pains_ok"])

    df = df.copy()

    # compute Lipinski violations
    lip_viol = (
        (df[mw_col] > 500).astype(int)
        + (df[logp_col] > 5).astype(int)
        + (df[hbd_col] > 5).astype(int)
        + (df[hba_col] > 10).astype(int)
    )
    df["lipinski_violations_calc"] = lip_viol

    # base filter: fairly broad, meant to match LiteGov/HeavyGov front-end
    mask = (
        (df[q_col] >= 0.2)  # very loose QED floor
        & (df[sa_col] <= 7.0)  # SA not terrible
        & (df[mw_col].between(150.0, 650.0))
        & (df[logp_col].between(-1.0, 6.0))
        & (df[tpsa_col] <= 180.0)
        & (df[rtb_col] <= 12)
        & (df["lipinski_violations_calc"] <= 2)
        & (df[pains_col] == 0)
    )

    return df.loc[mask].copy()


def build_qed_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    QED-only baseline:
    - apply common filters
    - sort by QED descending
    """
    df_f = apply_common_filters(df)
    q_col = find_column(df_f, ["QED", "qed"])
    out = df_f.sort_values(q_col, ascending=False).reset_index(drop=True)
    out["baseline"] = "qed"
    out["rank_qed"] = range(1, len(out) + 1)
    return out


def build_rulekit_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    RuleKit baseline:
    - apply common filters (Lipinski+Veber+PAINS+SA/logP/MW)
    - optionally tighten thresholds slightly
    - rank by QED
    """
    df_f = apply_common_filters(df)

    q_col = find_column(df_f, ["QED", "qed"])
    sa_col = find_column(df_f, ["SA", "sa", "sa_ertl"])
    logp_col = find_column(df_f, ["logP", "LogP", "logp"])
    tpsa_col = find_column(df_f, ["tpsa", "TPSA"])
    rtb_col = find_column(df_f, ["rtb", "rotatable_bonds"])

    # slightly tighter rule kit (you can tune these later if needed)
    mask = (
        (df_f[q_col] >= 0.4)
        & (df_f[sa_col] <= 5.0)
        & (df_f[logp_col].between(0.0, 5.0))
        & (df_f[tpsa_col] <= 140.0)
        & (df_f[rtb_col] <= 10)
    )

    filtered = df_f.loc[mask].copy()
    filtered = filtered.sort_values(q_col, ascending=False).reset_index(drop=True)
    filtered["baseline"] = "rulekit"
    filtered["rank_rulekit"] = range(1, len(filtered) + 1)
    return filtered


def parse_tau_seed(path: Path):
    """
    Extract tau and seed from filename like:
        pass_genmol_tau0.8_r1.0_seed0.csv
    Returns (tau_str, seed_int)
    """
    m = re.search(r"tau([0-9.]+)_r[0-9.]+_seed(\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot parse tau/seed from filename: {path}")
    tau_str = m.group(1)
    seed = int(m.group(2))
    return tau_str, seed


def main():
    root = Path(__file__).resolve().parents[1]
    denovo_root = root / "results" / "denovo"
    pass_dir = denovo_root / "pass"
    qed_dir = denovo_root / "qed"
    rule_dir = denovo_root / "rulekit"

    qed_dir.mkdir(parents=True, exist_ok=True)
    rule_dir.mkdir(parents=True, exist_ok=True)

    pass_files = sorted(pass_dir.glob("pass_genmol_tau*_r1.0_seed*.csv"))
    if not pass_files:
        print(f"[ERROR] No pass files found in {pass_dir}")
        return

    print(f"[INFO] Found {len(pass_files)} pass files")

    for path in pass_files:
        print(f"[INFO] Processing {path.name}")
        try:
            tau_str, seed = parse_tau_seed(path)
        except ValueError as e:
            print(f"[WARN] {e}, skipping")
            continue

        df = pd.read_csv(path)

        # ensure we have SMILES; adjust column names if needed
        if not any(c in df.columns for c in ["smiles", "SMILES"]):
            print(f"[WARN] No SMILES column in {path.name}, skipping")
            continue

        # QED baseline
        try:
            df_qed = build_qed_baseline(df)
            out_qed = qed_dir / f"qed_genmol_tau{tau_str}_r1.0_seed{seed}.csv"
            df_qed.to_csv(out_qed, index=False)
            print(f"[OK] wrote {out_qed} (n={len(df_qed)})")
        except Exception as e:
            print(f"[WARN] Failed to build QED baseline for {path.name}: {e}")

        # RuleKit baseline
        try:
            df_rule = build_rulekit_baseline(df)
            out_rule = rule_dir / f"rulekit_genmol_tau{tau_str}_r1.0_seed{seed}.csv"
            df_rule.to_csv(out_rule, index=False)
            print(f"[OK] wrote {out_rule} (n={len(df_rule)})")
        except Exception as e:
            print(f"[WARN] Failed to build RuleKit baseline for {path.name}: {e}")


if __name__ == "__main__":
    main()
