#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import QED, Descriptors


ROOT = Path(__file__).resolve().parents[1]
DOCK_DIR = ROOT / "docking"
STATS_DIR = DOCK_DIR / "stats"

GENMOL_SCORES = STATS_DIR / "dock_scores_all.csv"
GENMOL_LIGANDS = DOCK_DIR / "ligands" / "ligands_for_docking_genmol.csv"

OUT_DIR = ROOT / "figs"
TARGET = "6LU7"
TOP_K = 20  # number of top hits per method to export


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Docking score file not found: {path}")
    df = pd.read_csv(path)
    for col in ("ligand_id", "target", "method", "score"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {path}")
    df = df[(df["score"] > -30) & (df["score"] < 0)].copy()
    return df


def load_ligands(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Ligand metadata CSV not found, skipping: {path}")
        return None

    df = pd.read_csv(path)
    if "ligand_id" not in df.columns:
        print(f"[WARN] 'ligand_id' column missing in {path}, skipping metadata.")
        return None

    if "smiles" not in df.columns and "SMILES" not in df.columns:
        print(f"[WARN] no SMILES column in {path}; properties will not be computed.")
    return df


def compute_properties(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    if smiles_col not in df.columns:
        return df

    qed_vals = []
    mw_vals = []
    logp_vals = []
    tpsa_vals = []
    hbd_vals = []
    hba_vals = []
    rotb_vals = []

    for s in df[smiles_col]:
        mol = Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
        if mol is None:
            qed_vals.append(np.nan)
            mw_vals.append(np.nan)
            logp_vals.append(np.nan)
            tpsa_vals.append(np.nan)
            hbd_vals.append(np.nan)
            hba_vals.append(np.nan)
            rotb_vals.append(np.nan)
        else:
            qed_vals.append(float(QED.qed(mol)))
            mw_vals.append(float(Descriptors.MolWt(mol)))
            logp_vals.append(float(Descriptors.MolLogP(mol)))
            tpsa_vals.append(float(Descriptors.TPSA(mol)))
            hbd_vals.append(float(Descriptors.NumHDonors(mol)))
            hba_vals.append(float(Descriptors.NumHAcceptors(mol)))
            rotb_vals.append(float(Descriptors.NumRotatableBonds(mol)))

    df = df.copy()
    df["qed"] = qed_vals
    df["mw"] = mw_vals
    df["logp"] = logp_vals
    df["tpsa"] = tpsa_vals
    df["hbd"] = hbd_vals
    df["hba"] = hba_vals
    df["rotb"] = rotb_vals
    return df


def main() -> None:
    scores = load_scores(GENMOL_SCORES)
    scores_t = scores[scores["target"] == TARGET].copy()

    print(f"[INFO] Loaded {len(scores)} rows from {GENMOL_SCORES}")
    print(f"[INFO] Target = {TARGET}, rows = {len(scores_t)}")
    print(f"[INFO] Methods present: {sorted(scores_t['method'].unique())}")

    lig = load_ligands(GENMOL_LIGANDS)
    if lig is not None:
        if "smiles" in lig.columns:
            smiles_col = "smiles"
        elif "SMILES" in lig.columns:
            lig = lig.rename(columns={"SMILES": "smiles"})
            smiles_col = "smiles"
        else:
            smiles_col = None

        if smiles_col is not None:
            lig = compute_properties(lig, smiles_col=smiles_col)
    else:
        smiles_col = None

    methods_of_interest = ["raw", "lite", "heavy"]
    rows = []
    for m in methods_of_interest:
        sub = scores_t[scores_t["method"] == m].copy()
        if sub.empty:
            continue
        top = sub.nsmallest(TOP_K, "score")
        top["case_method"] = m
        rows.append(top)

    if not rows:
        print("[ERROR] no rows found for the selected target/methods.")
        return

    top_all = pd.concat(rows, ignore_index=True)

    if lig is not None:
        lig = lig.set_index("ligand_id")
        merged_rows = []
        for _, row in top_all.iterrows():
            lid = row["ligand_id"]
            row_dict = row.to_dict()
            if lid in lig.index:
                meta = lig.loc[lid]
                for col in lig.columns:
                    if col not in row_dict:
                        row_dict[col] = meta[col]
            merged_rows.append(row_dict)
        out_df = pd.DataFrame(merged_rows)
    else:
        out_df = top_all

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"case_candidates_genmol_{TARGET}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} with {len(out_df)} rows")


if __name__ == "__main__":
    main()
