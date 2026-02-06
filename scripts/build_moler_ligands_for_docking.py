#!/usr/bin/env python
"""
Build docking ligand list for MoLeR-based experiments.

Aligns the output schema and selection logic with build_ligands_for_docking.py
used for GenMol:

- For each governance line (raw / qed / rulekit / lite / heavy)
- For each seed-specific CSV file under results/moler/<method>/
- Take the first N_TOP molecules (assuming files are pre-sorted)
- Output columns: ligand_id, smiles, method, rank, tau, seed

ligand_id format:
    moler_<method>_seed<seed>_<rank:04d>

Notes:
- MoLeR results usually do not have a tau dimension, so tau is set to NaN.
"""

from pathlib import Path
from typing import Dict, List, Optional
import re
import math

import pandas as pd
from rdkit import Chem

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIRS: Dict[str, Path] = {
    "raw": ROOT / "results" / "moler" / "raw",
    "qed": ROOT / "results" / "moler" / "qed",
    "rulekit": ROOT / "results" / "moler" / "rulekit",
    "lite": ROOT / "results" / "moler" / "lite",
    "heavy": ROOT / "results" / "moler" / "heavy",
}

OUTPUT_DIR = ROOT / "docking" / "ligands"
OUTPUT_CSV = OUTPUT_DIR / "ligands_for_docking_moler.csv"

# Number of ligands per (method, seed) file
N_TOP = 100


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_smiles_like_column(df: pd.DataFrame) -> Optional[str]:
    """Return a column name that can be interpreted as SMILES or RDKit Mol."""
    candidates = [
        "smiles",
        "SMILES",
        "Smiles",
        "canonical_smiles",
        "can_smiles",
        "Mol",
        "mol",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_to_smiles(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Ensure df has a 'smiles' column.

    If `col` is text-like, copy it into 'smiles'.
    If `col` contains RDKit Mol objects, convert them to SMILES.
    """
    series = df[col]
    df = df.copy()

    nonnull = series.dropna()
    is_mol = False
    if not nonnull.empty:
        sample = nonnull.iloc[0]
        is_mol = hasattr(sample, "GetNumAtoms")

    if is_mol:
        smiles_list: List[str] = []
        for x in series:
            if x is None:
                smiles_list.append("")
                continue
            try:
                s = Chem.MolToSmiles(x)
            except Exception:
                s = ""
            smiles_list.append(s)
        df["smiles"] = smiles_list
    else:
        df["smiles"] = series.astype(str)

    return df


def parse_seed_from_filename(path: Path) -> int:
    """
    Extract seed number from filename if it contains 'seed<N>'.

    Examples:
        heavy_processed_raw_seed6.filtered.csv -> 6
        something_seed0.csv                    -> 0

    If no seed pattern is found, return -1.
    """
    m = re.search(r"seed(\d+)", path.name)
    if m:
        return int(m.group(1))
    return -1


def load_topk_from_file(path: Path, method_label: str, n_top: int) -> pd.DataFrame:
    """
    Read a single CSV file, normalize SMILES, and take the top-n_top rows.

    Returns a DataFrame with columns:
        ligand_id, smiles, method, rank, tau, seed
    """
    try:
        df_raw = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read {path}: {e}")
        return pd.DataFrame()

    col = find_smiles_like_column(df_raw)
    if col is None:
        print(
            f"[WARN] skipping {path}, no SMILES-like or Mol-like column found "
            f"(expected one of smiles/SMILES/canonical_smiles/Mol/mol)."
        )
        return pd.DataFrame()

    try:
        df = normalize_to_smiles(df_raw, col)
    except Exception as e:
        print(f"[WARN] failed to normalize column '{col}' in {path}: {e}")
        return pd.DataFrame()

    if "smiles" not in df.columns:
        print(f"[WARN] skipping {path}, could not obtain a 'smiles' column.")
        return pd.DataFrame()

    # Clean SMILES and truncate to top-n
    df = df.dropna(subset=["smiles"])
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"] != ""]
    if df.empty:
        return pd.DataFrame()

    df = df.head(n_top).reset_index(drop=True)

    seed = parse_seed_from_filename(path)
    seed_int = int(seed)

    n = len(df)
    ligand_ids = [
        f"moler_{method_label}_seed{seed_int}_{i+1:04d}" for i in range(n)
    ]

    out = pd.DataFrame()
    out["ligand_id"] = ligand_ids
    out["smiles"] = df["smiles"]
    out["method"] = method_label
    out["rank"] = range(1, n + 1)
    # MoLeR has no tau dimension; keep column for alignment with GenMol
    out["tau"] = math.nan
    out["seed"] = seed_int

    return out


def collect_for_method(method_label: str, directory: Path, n_top: int) -> pd.DataFrame:
    """
    Collect top-k ligands for a given MoLeR method across all seed files.

    Each CSV under `directory` contributes up to n_top ligands.
    """
    files = sorted(
        p for p in directory.glob("*.csv") if p.name.lower() != "summary.csv"
    )

    print(
        f"[INFO] Method {method_label}: found {len(files)} files in {directory}"
    )

    if not files:
        return pd.DataFrame()

    all_rows: List[pd.DataFrame] = []

    for path in files:
        print(f"[INFO]  reading {path.name}")
        topk = load_topk_from_file(path, method_label=method_label, n_top=n_top)
        if topk.empty:
            continue
        all_rows.append(topk)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []

    for method_label, directory in INPUT_DIRS.items():
        df = collect_for_method(method_label=method_label, directory=directory, n_top=N_TOP)
        if df.empty:
            print(f"[WARN] no ligands collected for method {method_label}")
            continue
        frames.append(df)

    if not frames:
        print("[ERROR] No ligands collected for any MoLeR method.")
        return

    all_ligands = pd.concat(frames, ignore_index=True)
    all_ligands.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] wrote {OUTPUT_CSV} with {len(all_ligands)} rows")


if __name__ == "__main__":
    main()
