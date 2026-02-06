#!/usr/bin/env python
import argparse
import csv
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]

# Input CSV: must contain at least columns "ligand_id" and "smiles"
LIG_CSV = ROOT / "docking" / "ligands" / "ligands_for_docking_moler.csv"

# Output directories
SDF_DIR = ROOT / "docking" / "ligands" / "sdf"
PDBQT_DIR = ROOT / "docking" / "ligands" / "pdbqt"


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------


def ensure_dirs() -> None:
    SDF_DIR.mkdir(parents=True, exist_ok=True)
    PDBQT_DIR.mkdir(parents=True, exist_ok=True)


def generate_3d_sdf(ligand_id: str, smiles: str) -> Optional[Path]:
    """
    Generate a 3D SDF file for a given SMILES using RDKit.
    Returns the path to the SDF file, or None on failure.
    """
    sdf_path = SDF_DIR / f"{ligand_id}.sdf"
    if sdf_path.exists():
        # Already generated
        return sdf_path

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[WARN] RDKit failed to parse SMILES for {ligand_id}")
        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0

    try:
        res = AllChem.EmbedMolecule(mol, params)
        if res != 0:
            print(f"[WARN] Embedding failed for {ligand_id}")
            return None

        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception as e:
        print(f"[WARN] 3D generation failed for {ligand_id}: {e}")
        return None

    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()

    # You can comment this out if output is too verbose
    print(f"[OK] wrote SDF: {sdf_path}")
    return sdf_path


def obabel_available() -> bool:
    exe = shutil.which("obabel")
    return exe is not None


def sdf_to_pdbqt_with_obabel(sdf_path: Path, pdbqt_path: Path) -> bool:
    """
    Convert SDF to PDBQT using OpenBabel if available.
    Returns True on success, False otherwise.
    """
    exe = shutil.which("obabel")
    if exe is None:
        return False

    cmd = [
        exe,
        "-isdf",
        str(sdf_path),
        "-opdbqt",
        "-O",
        str(pdbqt_path),
        "-xh",
    ]
    print(f"[CMD] {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(f"[WARN] obabel failed for {sdf_path.name}")
        print(result.stdout)
        print(result.stderr)
        return False

    print(f"[OK] wrote PDBQT: {pdbqt_path}")
    return True


def process_ligand(row: Dict[str, str], use_obabel: bool) -> None:
    """
    Process a single ligand:
      - generate 3D SDF with RDKit
      - optionally convert to PDBQT using obabel
    """
    ligand_id = row["ligand_id"]
    smiles = row["smiles"]

    pdbqt_path = PDBQT_DIR / f"{ligand_id}.pdbqt"
    if pdbqt_path.exists():
        # Already processed
        return

    sdf_path = generate_3d_sdf(ligand_id, smiles)
    if sdf_path is None:
        return

    if use_obabel:
        ok = sdf_to_pdbqt_with_obabel(sdf_path, pdbqt_path)
        if not ok:
            return
    else:
        # If obabel is not available, we stop at SDF.
        # The user can convert SDF to PDBQT with their own pipeline.
        return


def worker(args: Tuple[Dict[str, str], bool]) -> Optional[str]:
    """
    Small wrapper for multiprocessing.
    Returns the ligand_id for progress reporting, or None on failure.
    """
    row, use_obabel = args
    ligand_id = row.get("ligand_id", "")
    try:
        process_ligand(row, use_obabel=use_obabel)
        return ligand_id
    except Exception as e:
        print(f"[ERROR] worker failed for ligand {ligand_id}: {e}")
        return None


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 3D SDF/PDBQT for ligands in parallel (MoLeR)."
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=None,
        help="Number of worker processes (default: min(32, cpu_count)).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(LIG_CSV),
        help="Path to ligand CSV (default: ligands_for_docking_moler.csv).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)

    ensure_dirs()

    if not csv_path.exists():
        print(f"[ERROR] ligand CSV not found: {csv_path}")
        return

    use_obabel = obabel_available()
    print(f"[INFO] obabel available: {use_obabel}")

    # Decide number of processes
    cpu_count = os.cpu_count() or 1
    if args.n_procs is None:
        n_procs = min(32, cpu_count)
    else:
        n_procs = max(1, min(args.n_procs, cpu_count))

    print(f"[INFO] using {n_procs} worker processes (cpu_count={cpu_count})")

    # Load all rows first so they can be distributed to workers
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = [row for row in reader]

    total = len(rows)
    print(f"[INFO] total ligands in CSV: {total}")

    if total == 0:
        return

    tasks: List[Tuple[Dict[str, str], bool]] = [(row, use_obabel) for row in rows]

    # Parallel execution with progress reporting
    processed = 0
    with ProcessPoolExecutor(max_workers=n_procs) as pool:
        futures = [pool.submit(worker, t) for t in tasks]

        for fut in as_completed(futures):
            try:
                _ = fut.result()
            except Exception as e:
                print(f"[ERROR] worker raised: {e}")

            processed += 1
            if processed % 50 == 0 or processed == total:
                print(f"[INFO] processed {processed}/{total} ligands")

    print(f"[INFO] finished processing {processed} ligands")


if __name__ == "__main__":
    main()
