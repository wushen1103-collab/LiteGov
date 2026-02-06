#!/usr/bin/env python
from pathlib import Path
import re
import pandas as pd


N_TOP = 100  # number of ligands per (method, tau, seed) to send to docking


def find_column(df, candidates):
    """
    Return the first column name that exists in df.columns from the candidates list.
    Raise ValueError if none of them is present.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of candidate columns {candidates} found in {list(df.columns)}")


def parse_tau_seed(path: Path):
    """
    Extract tau and seed from filename like:
        pass_genmol_tau0.8_r1.0_seed0.csv
        qed_genmol_tau1.0_r1.0_seed2.csv
        rulekit_genmol_tau1.2_r1.0_seed1.csv
        filter_rank_genmol_tau0.8_r1.0_seed0.csv
        govern_filter_genmol_tau1.0_r1.0_seed1.csv
    Returns (tau_str, seed_int)
    """
    m = re.search(r"tau([0-9.]+)_r[0-9.]+_seed(\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot parse tau/seed from filename: {path}")
    tau_str = m.group(1)
    seed = int(m.group(2))
    return tau_str, seed


def load_topk(df: pd.DataFrame, method_label: str, n_top: int) -> pd.DataFrame:
    """
    Take the first n_top rows from df, keeping only SMILES and adding method/rank.
    Assumes df is already pre-sorted according to the correct score for this method.
    """
    smi_col = find_column(df, ["smiles", "SMILES"])
    sub = df.head(n_top).copy()
    sub = sub.reset_index(drop=True)

    out = pd.DataFrame()
    out["smiles"] = sub[smi_col]
    out["method"] = method_label
    out["rank"] = range(1, len(out) + 1)
    return out


def collect_for_method(
    denovo_root: Path,
    method_name: str,
    pattern: str,
    method_label: str,
    n_top: int,
):
    """
    Collect top-k ligands for a given method across all tau/seed files.

    - method_name: for logging only
    - pattern: glob pattern relative to denovo_root
    - method_label: short label used in the output (raw/qed/rulekit/lite/heavy)
    """
    files = sorted(denovo_root.glob(pattern))
    all_rows = []

    print(f"[INFO] Method {method_name}: found {len(files)} files with pattern {pattern}")
    if not files:
        return pd.DataFrame()

    for path in files:
        print(f"[INFO]  reading {path.name}")
        try:
            tau_str, seed = parse_tau_seed(path)
        except ValueError as e:
            print(f"[WARN] {e}, skipping this file")
            continue

        df = pd.read_csv(path)
        try:
            topk = load_topk(df, method_label=method_label, n_top=n_top)
        except Exception as e:
            print(f"[WARN] failed to load top-k from {path.name}: {e}")
            continue

        tau = float(tau_str)
        seed_int = int(seed)
        n = len(topk)

        # build ligand_id: method_tau{tau}_seed{seed}_{rank:04d}
        ligand_ids = [
            f"{method_label}_tau{tau_str}_seed{seed_int}_{i+1:04d}" for i in range(n)
        ]

        topk.insert(0, "ligand_id", ligand_ids)
        topk["tau"] = tau
        topk["seed"] = seed_int

        all_rows.append(topk)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def main():
    root = Path(__file__).resolve().parents[1]
    denovo_root = root / "results" / "denovo"
    out_dir = root / "docking" / "ligands"
    out_dir.mkdir(parents=True, exist_ok=True)

    # raw: pass
    df_raw = collect_for_method(
        denovo_root=denovo_root,
        method_name="raw",
        pattern="pass/pass_genmol_tau*_r1.0_seed*.csv",
        method_label="raw",
        n_top=N_TOP,
    )

    # qed baseline
    df_qed = collect_for_method(
        denovo_root=denovo_root,
        method_name="qed",
        pattern="qed/qed_genmol_tau*_r1.0_seed*.csv",
        method_label="qed",
        n_top=N_TOP,
    )

    # rulekit baseline
    df_rule = collect_for_method(
        denovo_root=denovo_root,
        method_name="rulekit",
        pattern="rulekit/rulekit_genmol_tau*_r1.0_seed*.csv",
        method_label="rulekit",
        n_top=N_TOP,
    )

    # lite: filter_rank
    df_lite = collect_for_method(
        denovo_root=denovo_root,
        method_name="lite",
        pattern="filter_rank/filter_rank_genmol_tau*_r1.0_seed*.csv",
        method_label="lite",
        n_top=N_TOP,
    )

    # heavy: govern
    df_heavy = collect_for_method(
        denovo_root=denovo_root,
        method_name="heavy",
        pattern="govern/govern_filter_genmol_tau*_r1.0_seed*.csv",
        method_label="heavy",
        n_top=N_TOP,
    )

    frames = [df_raw, df_qed, df_rule, df_lite, df_heavy]
    frames = [f for f in frames if not f.empty]

    if not frames:
        print("[ERROR] No ligands collected for any method.")
        return

    all_ligands = pd.concat(frames, ignore_index=True)

    out_csv = out_dir / "ligands_for_docking.csv"
    all_ligands.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} with {len(all_ligands)} rows")


if __name__ == "__main__":
    main()
