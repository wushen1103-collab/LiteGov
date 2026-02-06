#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCK_DIR = ROOT / "docking"
STATS_DIR = DOCK_DIR / "stats"

# GenMol paths
GENMOL_OUT_ROOT = DOCK_DIR / "genmol-outs"
GENMOL_LIG_CSV = DOCK_DIR / "ligands" / "ligands_for_docking.csv"
GENMOL_OUT_CSV = STATS_DIR / "dock_scores_all.csv"  # kept for backward compatibility

# MOLeR paths (try both capitalised and lowercase)
MOLER_OUT_ROOT_CANDIDATES = [
    DOCK_DIR / "Moler-outs",
    DOCK_DIR / "moler-outs",
]
MOLER_LIG_CSV = DOCK_DIR / "ligands" / "ligands_for_docking_moler.csv"
MOLER_OUT_CSV = STATS_DIR / "dock_scores_all_moler.csv"


def parse_vina_score(pdbqt_path: Path) -> float | None:
    """
    Parse the first Vina score from a *_out.pdbqt file.

    Looks for a line like:
        REMARK VINA RESULT:    -7.6      0.0      0.0

    Returns the score as float, or None if not found.
    """
    pattern = re.compile(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)")
    with pdbqt_path.open("r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return None
    return None


def load_ligand_meta(csv_path: Path) -> pd.DataFrame | None:
    """
    Load ligand metadata and index by ligand_id if available.

    If the file does not exist or does not contain a 'ligand_id' column,
    returns None and metadata will be skipped.
    """
    if not csv_path.exists():
        print(f"[INFO] Ligand metadata CSV not found, skipping: {csv_path}")
        return None

    meta = pd.read_csv(csv_path)
    if "ligand_id" not in meta.columns:
        print(
            f"[INFO] Ligand metadata CSV has no 'ligand_id' column, "
            f"skipping metadata: {csv_path}"
        )
        return None

    meta = meta.set_index("ligand_id")
    print(f"[INFO] Loaded ligand metadata from {csv_path} with {len(meta)} rows")
    return meta


def collect_for_generator(
    outs_root: Path,
    lig_csv: Path,
    out_csv: Path,
    generator_name: str,
) -> None:
    """
    Walk a docking outs directory and collect all Vina scores into a CSV.

    Directory layout is assumed to be:
        outs_root / <target> / <method> / *_out.pdbqt

    The output CSV has columns:
        ligand_id, target, method, score, (optional) tau, seed, ...
    """
    if not outs_root.exists():
        print(f"[WARN] Docking outputs directory not found for {generator_name}: {outs_root}")
        return

    STATS_DIR.mkdir(parents=True, exist_ok=True)

    lig_meta = load_ligand_meta(lig_csv)

    rows: list[dict] = []

    print(f"[INFO] Collecting docking scores for {generator_name} from {outs_root}")

    for target_dir in sorted(outs_root.iterdir()):
        if not target_dir.is_dir():
            continue
        target = target_dir.name

        for method_dir in sorted(target_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name

            for pdbqt_path in sorted(method_dir.glob("*_out.pdbqt")):
                ligand_id = pdbqt_path.stem.replace("_out", "")
                score = parse_vina_score(pdbqt_path)
                if score is None:
                    print(f"[WARN] No Vina score found in {pdbqt_path}")
                    continue

                row: dict[str, object] = {
                    "ligand_id": ligand_id,
                    "target": target,
                    "method": method,
                    "score": score,
                }

                if lig_meta is not None and ligand_id in lig_meta.index:
                    meta = lig_meta.loc[ligand_id]
                    # Attach tau/seed/generator if available in metadata
                    for col in ("tau", "seed", "generator"):
                        if col in meta:
                            row[col] = meta[col]

                rows.append(row)

    if not rows:
        print(f"[ERROR] No docking scores collected for {generator_name}.")
        return

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} with {len(df)} rows")


def main() -> None:
    # GenMol
    collect_for_generator(
        outs_root=GENMOL_OUT_ROOT,
        lig_csv=GENMOL_LIG_CSV,
        out_csv=GENMOL_OUT_CSV,
        generator_name="genmol",
    )

    # MOLeR (try both candidate directories)
    moler_out_root = None
    for cand in MOLER_OUT_ROOT_CANDIDATES:
        if cand.exists():
            moler_out_root = cand
            break

    if moler_out_root is None:
        print(
            "[WARN] No MOLeR docking outputs directory found. "
            f"Tried: {', '.join(str(p) for p in MOLER_OUT_ROOT_CANDIDATES)}"
        )
    else:
        collect_for_generator(
            outs_root=moler_out_root,
            lig_csv=MOLER_LIG_CSV,
            out_csv=MOLER_OUT_CSV,
            generator_name="moler",
        )


if __name__ == "__main__":
    main()
