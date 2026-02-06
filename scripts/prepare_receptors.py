#!/usr/bin/env python
import subprocess
from pathlib import Path
import shutil


def clean_pdb(raw_path: Path, clean_path: Path) -> None:
    """
    Create a cleaned PDB file that keeps only protein ATOM/TER lines.
    HETATM, water, ligands and ions are removed.
    """
    lines_out = []
    with raw_path.open("r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("TER") or line.startswith("END"):
                lines_out.append(line)
    with clean_path.open("w") as f:
        f.writelines(lines_out)


def run_cmd(cmd, cwd=None):
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("[WARN] Command failed:")
        print(result.stdout)
        print(result.stderr)
    return result.returncode == 0


def prepare_with_mgl(clean_path: Path, pdbqt_path: Path) -> bool:
    exe = shutil.which("prepare_receptor4.py")
    if exe is None:
        return False
    cmd = [exe, "-r", str(clean_path), "-o", str(pdbqt_path), "-A", "checkhydrogens"]
    return run_cmd(cmd)


def prepare_with_obabel(clean_path: Path, pdbqt_path: Path) -> bool:
    exe = shutil.which("obabel")
    if exe is None:
        return False
    cmd = ["obabel", "-ipdb", str(clean_path), "-opdbqt", "-O", str(pdbqt_path), "-xh"]
    return run_cmd(cmd)


def main():
    root = Path(__file__).resolve().parents[1]
    rec_root = root / "docking" / "receptors"

    if not rec_root.exists():
        print(f"[ERROR] Receptors directory not found: {rec_root}")
        return

    use_mgl = shutil.which("prepare_receptor4.py") is not None
    use_obabel = shutil.which("obabel") is not None

    print(f"[INFO] prepare_receptor4.py found: {use_mgl}")
    print(f"[INFO] obabel found: {use_obabel}")

    if not use_mgl and not use_obabel:
        print("[ERROR] Neither prepare_receptor4.py nor obabel is available in PATH.")
        print("        Please install MGLTools or OpenBabel, or modify this script to use your own pipeline.")
        return

    for target_dir in sorted(rec_root.iterdir()):
        if not target_dir.is_dir():
            continue

        raw_pdb = target_dir / "raw.pdb"
        if not raw_pdb.exists():
            print(f"[WARN] raw.pdb not found for target {target_dir.name}, skipping.")
            continue

        clean_pdb_path = target_dir / "receptor_clean.pdb"
        pdbqt_path = target_dir / "receptor.pdbqt"

        print(f"[INFO] Processing target: {target_dir.name}")
        print(f"       raw:   {raw_pdb}")
        print(f"       clean: {clean_pdb_path}")
        print(f"       pdbqt: {pdbqt_path}")

        # Step 1: clean PDB
        clean_pdb(raw_pdb, clean_pdb_path)

        # Step 2: generate PDBQT
        ok = False
        if use_mgl:
            ok = prepare_with_mgl(clean_pdb_path, pdbqt_path)
        if not ok and use_obabel:
            print("[INFO] Falling back to obabel for pdbqt generation.")
            ok = prepare_with_obabel(clean_pdb_path, pdbqt_path)

        if ok and pdbqt_path.exists():
            print(f"[OK] Generated {pdbqt_path}")
        else:
            print(f"[FAIL] Failed to generate receptor.pdbqt for {target_dir.name}")


if __name__ == "__main__":
    main()
