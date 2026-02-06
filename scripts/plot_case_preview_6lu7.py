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
OUT_PNG = FIG_DIR / "case_preview_genmol_6LU7.png"
OUT_SVG = FIG_DIR / "case_preview_genmol_6LU7.svg"

N_PER_METHOD = 10
TARGET = "6LU7"


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Case candidates CSV not found: {path}")
    df = pd.read_csv(path)
    if "smiles" not in df.columns and "SMILES" in df.columns:
        df = df.rename(columns={"SMILES": "smiles"})
    return df


def build_preview(df: pd.DataFrame):
    df = df[df["target"] == TARGET].copy()

    methods_order = ["raw", "lite", "heavy"]
    mols = []
    legends = []

    for m in methods_order:
        sub = df[df["method"] == m].copy()
        sub = sub.sort_values("score").head(N_PER_METHOD)

        for _, row in sub.iterrows():
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None:
                continue
            lid = row["ligand_id"]
            score = row["score"]
            qed = row["qed"]
            legend = f"{m}\n{lid}\nscore={score:.2f}, QED={qed:.2f}"
            mols.append(mol)
            legends.append(legend)

    return mols, legends


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_candidates(CASE_CSV)
    mols, legends = build_preview(df)

    subimg = (350, 350)  # increase clarity

    # PNG version
    png = Draw.MolsToGridImage(
        mols,
        molsPerRow=N_PER_METHOD,
        subImgSize=subimg,
        legends=legends,
        useSVG=False,
        returnPNG=True,
    )
    with open(OUT_PNG, "wb") as f:
        f.write(png)
    print(f"[OK] PNG written: {OUT_PNG}")

    # SVG version (super clean)
    svg = Draw.MolsToGridImage(
        mols,
        molsPerRow=N_PER_METHOD,
        subImgSize=subimg,
        legends=legends,
        useSVG=True,
    )
    with open(OUT_SVG, "w") as f:
        f.write(svg)
    print(f"[OK] SVG written: {OUT_SVG}")


if __name__ == "__main__":
    main()