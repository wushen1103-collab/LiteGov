#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figs"

CASE_CSV = FIG_DIR / "case_candidates_genmol_6LU7.csv"
OUT_PNG = FIG_DIR / "case_6LU7_genmol_2d.png"
OUT_SVG = FIG_DIR / "case_6LU7_genmol_2d.svg"

TARGET = "6LU7"

# final choices for the paper
CASE_LIGANDS = [
    ("raw", "raw_tau1.2_seed1_0031"),
    ("lite", "lite_tau0.8_seed2_0098"),
    ("heavy", "heavy_tau1.0_seed0_0086"),
]


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Case candidates CSV not found: {path}")
    df = pd.read_csv(path)
    if "smiles" not in df.columns and "SMILES" in df.columns:
        df = df.rename(columns={"SMILES": "smiles"})
    return df


def build_mols_and_legends(df: pd.DataFrame):
    df = df[df["target"] == TARGET].copy()

    mols = []
    legends = []

    # map for quick lookup
    df_map = {lid: row for lid, row in df.set_index("ligand_id").iterrows()}

    for method, lid in CASE_LIGANDS:
        if lid not in df_map:
            print(f"[WARN] ligand_id {lid} not found in case CSV, skipping.")
            continue
        row = df_map[lid]
        smi = str(row["smiles"])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[WARN] Failed to parse SMILES for {lid}, skipping.")
            continue

        score = float(row["score"])
        qed = float(row["qed"]) if "qed" in row.index and not np.isnan(row["qed"]) else None

        # short, paper-style legend
        if qed is not None:
            legend = f"{method} | score={score:.2f}, QED={qed:.2f}"
        else:
            legend = f"{method} | score={score:.2f}"

        mols.append(mol)
        legends.append(legend)

    return mols, legends


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_candidates(CASE_CSV)
    print(f"[INFO] Loaded {len(df)} rows from {CASE_CSV}")

    mols, legends = build_mols_and_legends(df)
    if not mols:
        print("[ERROR] No molecules to draw.")
        return

    print(f"[INFO] Will draw {len(mols)} molecules:")
    for lg in legends:
        print(f"       {lg}")

    subimg = (500, 500)

    # PNG
    png = Draw.MolsToGridImage(
        mols,
        molsPerRow=len(mols),
        subImgSize=subimg,
        legends=legends,
        useSVG=False,
        returnPNG=True,
    )
    with open(OUT_PNG, "wb") as f:
        f.write(png)
    print(f"[OK] PNG written: {OUT_PNG}")

    # SVG (publication quality)
    svg = Draw.MolsToGridImage(
        mols,
        molsPerRow=len(mols),
        subImgSize=subimg,
        legends=legends,
        useSVG=True,
    )
    with open(OUT_SVG, "w") as f:
        f.write(svg)
    print(f"[OK] SVG written: {OUT_SVG}")


if __name__ == "__main__":
    main()