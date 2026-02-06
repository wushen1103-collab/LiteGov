#!/usr/bin/env python

from __future__ import annotations
import pandas as pd
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIRS = {
    "GenMol": ROOT / "results" / "genmol",
    "MoLeR": ROOT / "results" / "moler",
}

OUT_PATH = ROOT / "docking" / "ligands" / "ligands_full_metadata.csv"


def normalize_property_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # unify qed
    if "qed" not in df.columns and "QED" in df.columns:
        df["qed"] = df["QED"]

    # unify sa
    if "sa" not in df.columns:
        if "SA" in df.columns:
            df["sa"] = df["SA"]
        elif "sa_ertl" in df.columns:
            df["sa"] = df["sa_ertl"]

    # unify logp
    if "logp" not in df.columns and "logP" in df.columns:
        df["logp"] = df["logP"]

    return df


def parse_genmol_tau_seed(stem: str):
    m = re.search(r"tau([0-9.]+)_r1\.0_seed([0-9]+)", stem)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def infer_genmol_method(path: Path) -> str | None:
    p = str(path).lower()
    if "/raw/" in p:
        return "raw"
    if "/qed/" in p:
        return "qed"
    if "/rulekit/" in p:
        return "rulekit"
    if "/lite/" in p or "filter_rank" in p:
        return "lite"
    if "/heavy/" in p or "govern_filter" in p:
        return "heavy"
    return None


def parse_moler_method_seed(path: Path):
    name = path.name.lower()

    # raw
    if name.startswith("processed_raw_seed"):
        m = re.match(r"processed_raw_seed([0-9]+)\.csv", name)
        if m:
            return "raw", m.group(1)

    # qed
    if name.startswith("processed_raw_seed") and "_qed" in name:
        m = re.match(r"processed_raw_seed([0-9]+)_qed\.csv", name)
        if m:
            return "qed", m.group(1)

    # rulekit
    if name.startswith("processed_raw_seed") and "_rulekit" in name:
        m = re.match(r"processed_raw_seed([0-9]+)_rulekit\.csv", name)
        if m:
            return "rulekit", m.group(1)

    # lite
    if name.startswith("lite_processed_raw_seed") and ".filtered" in name:
        m = re.match(r"lite_processed_raw_seed([0-9]+)\.filtered\.csv", name)
        if m:
            return "lite", m.group(1)

    # heavy
    if name.startswith("heavy_processed_raw_seed") and ".filtered" in name:
        m = re.match(r"heavy_processed_raw_seed([0-9]+)\.filtered\.csv", name)
        if m:
            return "heavy", m.group(1)

    return None, None


def gather_metadata_genmol(root: Path) -> pd.DataFrame:
    rows = []

    for path in root.rglob("*.csv"):
        name = path.name.lower()
        if "summary" in name:
            continue

        method = infer_genmol_method(path)
        if method is None:
            continue

        tau, seed = parse_genmol_tau_seed(path.stem)
        if tau is None or seed is None:
            continue

        try:
            df = pd.read_csv(path)
        except:
            continue

        if df.empty or "smiles" not in df.columns:
            continue

        df = normalize_property_columns(df)
        df = df.reset_index(drop=True)
        df["row_index"] = df.index + 1

        df["ligand_id"] = (
            f"{method}_tau{tau}_seed{seed}_"
            + df["row_index"].astype(str).str.zfill(4)
        )

        df["generator"] = "GenMol"
        df["method"] = method

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def gather_metadata_moler(root: Path) -> pd.DataFrame:
    rows = []

    for path in root.rglob("*.csv"):
        name = path.name.lower()
        if "summary" in name:
            continue

        method, seed = parse_moler_method_seed(path)
        if method is None or seed is None:
            continue

        try:
            df = pd.read_csv(path)
        except:
            continue

        # MoLeR always uses smiles_can
        smiles_col = "smiles_can"
        if smiles_col not in df.columns:
            continue

        df = df.rename(columns={smiles_col: "smiles"})

        df = normalize_property_columns(df)
        df = df.reset_index(drop=True)
        df["row_index"] = df.index + 1

        df["ligand_id"] = (
            f"moler_{method}_seed{seed}_"
            + df["row_index"].astype(str).str.zfill(4)
        )

        df["generator"] = "MoLeR"
        df["method"] = method

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def main():
    genmol_meta = gather_metadata_genmol(RESULTS_DIRS["GenMol"])
    moler_meta = gather_metadata_moler(RESULTS_DIRS["MoLeR"])

    all_meta = pd.concat([genmol_meta, moler_meta], ignore_index=True)

    # keep only relevant columns
    keep = [
        "generator", "method", "ligand_id", "smiles",
        "qed", "sa", "mw", "tpsa", "logp",
        "frac_csp3", "rings_aromatic"
    ]
    available = [c for c in keep if c in all_meta.columns]
    out = all_meta[available].drop_duplicates(subset=["generator", "ligand_id"])

    OUT_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("[OK] wrote:", OUT_PATH)
    print(out.head())


if __name__ == "__main__":
    main()
