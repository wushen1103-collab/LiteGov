#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors,
    QED,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import FilterCatalog


# --------- config ---------
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "results" / "moler" / "raw"
OUT_DIR = REPO_ROOT / "results" / "moler" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMILES_COL = "smiles"

# --------- PAINS catalog ---------
params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
PAINS_CATALOG = FilterCatalog.FilterCatalog(params)

# --------- SA scorer (Ertl) ---------
try:
    import sascorer

    def calc_sa_score(mol):
        return float(sascorer.calculateScore(mol))

except ImportError:

    def calc_sa_score(mol):
        return None


# --------- per-molecule descriptors ---------
def calc_descriptors(smiles: str) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "Mol": smiles,
            "QED": None,
            "logP": None,
            "SA": None,
            "scaffold": None,
            "sa_ertl": None,
            "smiles_can": None,
            "qed": None,
            "mw": None,
            "logp": None,
            "tpsa": None,
            "hba": None,
            "hbd": None,
            "rtb": None,
            "heavy_atoms": None,
            "rings_aromatic": None,
            "frac_csp3": None,
            "murcko_scaffold": None,
            "pains_n": None,
            "pains_ok": None,
            "sa": None,
        }

    smiles_can = Chem.MolToSmiles(mol)

    qed_val = QED.qed(mol)
    logp_val = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    rtb = Lipinski.NumRotatableBonds(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    rings_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    scaff_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaff_smiles = Chem.MolToSmiles(scaff_mol) if scaff_mol is not None else None

    pains_matches = PAINS_CATALOG.GetMatches(mol)
    pains_n = len(pains_matches)
    pains_ok = pains_n == 0

    sa_score = calc_sa_score(mol)

    return {
        "Mol": smiles,              # original SMILES
        "QED": qed_val,
        "logP": logp_val,
        "SA": sa_score,
        "scaffold": scaff_smiles,
        "sa_ertl": sa_score,
        "smiles_can": smiles_can,
        "qed": qed_val,
        "mw": mw,
        "logp": logp_val,
        "tpsa": tpsa,
        "hba": hba,
        "hbd": hbd,
        "rtb": rtb,
        "heavy_atoms": heavy_atoms,
        "rings_aromatic": rings_aromatic,
        "frac_csp3": frac_csp3,
        "murcko_scaffold": scaff_smiles,
        "pains_n": pains_n,
        "pains_ok": pains_ok,
        "sa": sa_score,
    }


TARGET_COLS: List[str] = [
    "Mol",
    "QED",
    "logP",
    "SA",
    "scaffold",
    "sa_ertl",
    "smiles_can",
    "qed",
    "mw",
    "logp",
    "tpsa",
    "hba",
    "hbd",
    "rtb",
    "heavy_atoms",
    "rings_aromatic",
    "frac_csp3",
    "murcko_scaffold",
    "pains_n",
    "pains_ok",
    "sa",
]


def process_file(path: Path) -> None:
    print(f"[INFO] Reading file: {path}")
    df_raw = pd.read_csv(path)

    if SMILES_COL not in df_raw.columns:
        raise ValueError(f"Column '{SMILES_COL}' not found in {path}, got {list(df_raw.columns)}")

    print(f"[INFO] Columns found: {list(df_raw.columns)}")
    smiles_list = df_raw[SMILES_COL].astype(str).tolist()

    print(f"[INFO] Parsing SMILES for file: {path}")
    records: List[Dict[str, Any]] = []
    n_valid = 0
    for s in smiles_list:
        rec = calc_descriptors(s)
        records.append(rec)
        if rec["smiles_can"] is not None:
            n_valid += 1

    print(f"[INFO] Valid SMILES: {n_valid} / {len(smiles_list)}")

    df_des = pd.DataFrame(records)
    df_des = df_des[TARGET_COLS]

    out_name = f"processed_{path.name}"
    out_path = OUT_DIR / out_name
    df_des.to_csv(out_path, index=False)

    print(f"[INFO] Output columns: {list(df_des.columns)}")
    print(f"[INFO] First 3 rows:\n{df_des.head(3)}")
    print(f"[INFO] Processed file saved: {out_path}\n")


def main() -> None:
    if not RAW_DIR.exists():
        print(f"[ERROR] RAW_DIR does not exist: {RAW_DIR}")
        sys.exit(1)

    files = sorted(RAW_DIR.glob("raw_*.csv"))
    if not files:
        print(f"[ERROR] No raw_*.csv found in {RAW_DIR}")
        sys.exit(1)

    for f in files:
        process_file(f)

    print("[INFO] All files processed.")


if __name__ == "__main__":
    main()
