#!/usr/bin/env python

import argparse
import glob
import os
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import FilterCatalog

import SA_Score


def build_pains_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)


PAINS_CATALOG = build_pains_catalog()


def compute_pains(mol: Optional[Chem.Mol]) -> Tuple[int, bool]:
    if mol is None:
        return 0, False
    hits = PAINS_CATALOG.GetMatches(mol)
    n = len(hits)
    return n, n == 0


def compute_sa(mol: Optional[Chem.Mol]) -> float:
    if mol is None:
        return float("nan")
    try:
        return float(SA_Score.sascorer.calculateScore(mol))
    except Exception:
        return float("nan")


def compute_descriptors(smiles: str) -> Dict[str, object]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "Mol": np.nan,
            "QED": float("nan"),
            "logP": float("nan"),
            "SA": float("nan"),
            "PAINS": float("nan"),
            "scaffold": None,
            "sa_ertl": float("nan"),
            "smiles_can": None,
            "qed": float("nan"),
            "mw": float("nan"),
            "logp": float("nan"),
            "tpsa": float("nan"),
            "hba": float("nan"),
            "hbd": float("nan"),
            "rtb": float("nan"),
            "heavy_atoms": float("nan"),
            "rings_aromatic": float("nan"),
            "frac_csp3": float("nan"),
            "murcko_scaffold": None,
            "pains_n": 0,
            "pains_ok": False,
            "sa": float("nan"),
        }

    # canonical smiles
    try:
        smiles_can = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        smiles_can = None

    # QED
    try:
        qed_val = float(QED.qed(mol))
    except Exception:
        qed_val = float("nan")

    # logP
    try:
        logp_val = float(Descriptors.MolLogP(mol))
    except Exception:
        logp_val = float("nan")

    # MW
    try:
        mw_val = float(Descriptors.MolWt(mol))
    except Exception:
        mw_val = float("nan")

    # TPSA
    try:
        tpsa_val = float(rdMolDescriptors.CalcTPSA(mol))
    except Exception:
        tpsa_val = float("nan")

    # HBA / HBD
    try:
        hba_val = float(Lipinski.NumHAcceptors(mol))
    except Exception:
        hba_val = float("nan")

    try:
        hbd_val = float(Lipinski.NumHDonors(mol))
    except Exception:
        hbd_val = float("nan")

    # rotatable bonds
    try:
        rtb_val = float(Lipinski.NumRotatableBonds(mol))
    except Exception:
        rtb_val = float("nan")

    # heavy atoms
    try:
        heavy_atoms_val = float(mol.GetNumHeavyAtoms())
    except Exception:
        heavy_atoms_val = float("nan")

    # aromatic rings
    try:
        rings_aromatic_val = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    except Exception:
        rings_aromatic_val = float("nan")

    # fraction Csp3
    try:
        frac_csp3_val = float(rdMolDescriptors.CalcFractionCSP3(mol))
    except Exception:
        frac_csp3_val = float("nan")

    # scaffolds
    try:
        scaf_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaf_smiles = Chem.MolToSmiles(scaf_mol, isomericSmiles=True, canonical=True)
    except Exception:
        scaf_smiles = None

    try:
        murcko_scaf = scaf_smiles
    except Exception:
        murcko_scaf = None

    # SA
    sa_val = compute_sa(mol)

    # PAINS
    pains_n, pains_ok = compute_pains(mol)
    pains_flag = 0.0 if pains_n == 0 else 1.0

    return {
        "Mol": np.nan,
        "QED": qed_val,
        "logP": logp_val,
        "SA": sa_val,
        "PAINS": pains_flag,
        "scaffold": scaf_smiles,
        "sa_ertl": sa_val,
        "smiles_can": smiles_can,
        "qed": qed_val,
        "mw": mw_val,
        "logp": logp_val,
        "tpsa": tpsa_val,
        "hba": hba_val,
        "hbd": hbd_val,
        "rtb": rtb_val,
        "heavy_atoms": heavy_atoms_val,
        "rings_aromatic": rings_aromatic_val,
        "frac_csp3": frac_csp3_val,
        "murcko_scaffold": murcko_scaf,
        "pains_n": int(pains_n),
        "pains_ok": bool(pains_ok),
        "sa": sa_val,
    }


def process_file(path: str, outdir: str) -> str:
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        raise ValueError(f"Input file {path} must contain a 'smiles' column.")

    records: List[Dict[str, object]] = []
    for smi in df["smiles"].astype(str).tolist():
        rec = {"smiles": smi}
        rec.update(compute_descriptors(smi))
        records.append(rec)

    out_df = pd.DataFrame.from_records(records)

    cols = [
        "smiles",
        "Mol",
        "QED",
        "logP",
        "SA",
        "PAINS",
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
    for c in out_df.columns:
        if c not in cols:
            cols.append(c)

    out_df = out_df[cols]

    base = os.path.basename(path)
    if base.startswith("genmol_"):
        out_name = "pass_" + base
    else:
        out_name = "pass_" + base

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, out_name)
    out_df.to_csv(out_path, index=False)
    return out_path


def pick_sa_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["sa", "sa_ertl", "SA"]:
        if col in df.columns:
            return col
    return None


def recompute_sa_mean_for_file(csv_dir: str, filename: str) -> float:
    csv_path = filename
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(csv_dir, filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find pass CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    sa_col = pick_sa_column(df)
    if sa_col is None:
        raise ValueError(
            f"No SA column found in {csv_path}. Expected one of: sa, sa_ertl, SA"
        )

    sa_series = pd.to_numeric(df[sa_col], errors="coerce")
    sa_mean = float(sa_series.mean())
    if not np.isfinite(sa_mean):
        return float("nan")
    return sa_mean


def update_pass_summary(summary_path: str, pass_dir: str) -> None:
    if not os.path.exists(summary_path):
        print(f"Summary CSV not found: {summary_path}. Skipping summary update.")
        return

    df_sum = pd.read_csv(summary_path)

    mode_col = "mode"
    file_col = "file"

    if mode_col not in df_sum.columns:
        raise ValueError(
            f"Summary CSV {summary_path} is missing mode column '{mode_col}'."
        )
    if file_col not in df_sum.columns:
        raise ValueError(
            f"Summary CSV {summary_path} is missing file column '{file_col}'."
        )
    if "sa_mean" not in df_sum.columns:
        raise ValueError(
            f"Summary CSV {summary_path} is missing 'sa_mean' column."
        )

    mask_pass = df_sum[mode_col] == "pass"
    if not mask_pass.any():
        print("No rows with mode == 'pass' found in summary. Nothing to update.")
        return

    df_pass = df_sum[mask_pass].copy()
    print(f"Found {len(df_pass)} pass rows in summary. Recomputing sa_mean...")

    new_sa_means = []
    for idx, row in df_pass.iterrows():
        fname = str(row[file_col])
        sa_mean = recompute_sa_mean_for_file(pass_dir, fname)
        new_sa_means.append((idx, sa_mean))
        print(f"  {fname}: sa_mean -> {sa_mean:.6f}")

    for idx, sa_mean in new_sa_means:
        df_sum.at[idx, "sa_mean"] = sa_mean

    backup_path = summary_path + ".bak"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    os.replace(summary_path, backup_path)
    df_sum.to_csv(summary_path, index=False)

    print(f"Updated sa_mean for pass rows in {summary_path}")
    print(f"Backup of old summary written to {backup_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute pass-level descriptors from raw genmol outputs and update pass summary."
    )
    parser.add_argument(
        "--in-glob",
        type=str,
        default="/root/govmol-lite/genmol-raw/genmol_tau*_r1.0_seed*.csv",
        help="Glob pattern for raw genmol CSV files.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/denovo/pass",
        help="Output directory for pass CSV files.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="results/denovo/pass/summary.csv",
        help="Path to pass summary CSV to update.",
    )

    args = parser.parse_args()

    paths = sorted(glob.glob(args.in_glob))
    if not paths:
        raise SystemExit(f"No files matched pattern: {args.in_glob}")

    print(f"Found {len(paths)} input files.")
    os.makedirs(args.outdir, exist_ok=True)

    for p in paths:
        out_path = process_file(p, args.outdir)
        print(f"Wrote: {out_path}")

    # Update summary sa_mean for pass mode, if summary exists
    update_pass_summary(args.summary, args.outdir)


if __name__ == "__main__":
    main()
