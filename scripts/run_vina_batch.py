#!/usr/bin/env python
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]

VINA_BIN = "/root/miniconda3/envs/govmol/bin/vina"

LIGAND_PDBQT_DIR = ROOT / "docking" / "ligands" / "pdbqt"
RECEPTOR_ROOT = ROOT / "docking" / "receptors"
OUT_ROOT = ROOT / "docking" / "outs"

# Targets and methods to run docking for
TARGETS = [ "6LU7", "ERa"]
METHODS = ["raw", "qed", "rulekit", "lite", "heavy"]

# Support both GenMol-style ("raw_*") and MoLeR-style ("moler_raw_*") prefixes
LIGAND_PREFIXES = ["", "moler_"]

CPU_PER_TASK = 1          # number of threads per vina process
TOTAL_CORES = 32          # total CPU cores you want to use
MAX_WORKERS = max(1, TOTAL_CORES // CPU_PER_TASK)


@dataclass
class DockingJob:
    target: str
    method: str
    ligand_id: str
    receptor_pdbqt: Path
    config_txt: Path
    ligand_pdbqt: Path
    out_pdbqt: Path
    log_path: Path


def build_jobs() -> List[DockingJob]:
    jobs: List[DockingJob] = []

    for target in TARGETS:
        receptor_dir = RECEPTOR_ROOT / target
        receptor_pdbqt = receptor_dir / "receptor.pdbqt"
        config_txt = receptor_dir / "config.txt"

        if not receptor_pdbqt.exists():
            print(f"[WARN] receptor not found: {receptor_pdbqt}")
            continue
        if not config_txt.exists():
            print(f"[WARN] config not found: {config_txt}")
            continue

        for method in METHODS:
            ligand_paths: List[str] = []
            # collect ligands for both naming styles
            for prefix in LIGAND_PREFIXES:
                pattern = str(LIGAND_PDBQT_DIR / f"{prefix}{method}_*.pdbqt")
                ligand_paths.extend(glob.glob(pattern))

            ligand_paths = sorted(set(ligand_paths))
            if not ligand_paths:
                pat_str = ", ".join(
                    str(LIGAND_PDBQT_DIR / f"{prefix}{method}_*.pdbqt")
                    for prefix in LIGAND_PREFIXES
                )
                print(f"[WARN] no ligands for {method}, patterns={pat_str}")
                continue

            out_dir = OUT_ROOT / target / method
            out_dir.mkdir(parents=True, exist_ok=True)

            for lig_path_str in ligand_paths:
                lig_path = Path(lig_path_str)
                ligand_id = lig_path.stem

                out_pdbqt = out_dir / f"{ligand_id}_out.pdbqt"
                log_path = out_dir / f"{ligand_id}.log"

                jobs.append(
                    DockingJob(
                        target=target,
                        method=method,
                        ligand_id=ligand_id,
                        receptor_pdbqt=receptor_pdbqt,
                        config_txt=config_txt,
                        ligand_pdbqt=lig_path,
                        out_pdbqt=out_pdbqt,
                        log_path=log_path,
                    )
                )

    return jobs


def run_one_job(job: DockingJob) -> bool:
    cmd = [
        VINA_BIN,
        "--receptor",
        str(job.receptor_pdbqt),
        "--ligand",
        str(job.ligand_pdbqt),
        "--config",
        str(job.config_txt),
        "--out",
        str(job.out_pdbqt),
        "--cpu",
        str(CPU_PER_TASK),
    ]

    log_dir = job.log_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as e:
        msg = f"vina exception: {e}"
        try:
            with open(job.log_path, "w") as f:
                f.write(msg + "\n")
        except Exception:
            pass
        return False

    try:
        with open(job.log_path, "w") as f:
            f.write(result.stdout)
    except Exception as e:
        print(f"[WARN] could not write log for {job.ligand_id}: {e}")

    if result.returncode != 0:
        return False

    return True


def main() -> None:
    print(f"[INFO] Root dir     : {ROOT}")
    print(f"[INFO] Vina binary  : {VINA_BIN}")
    print(f"[INFO] Targets      : {TARGETS}")
    print(f"[INFO] Methods      : {METHODS}")
    print(f"[INFO] CPU per task : {CPU_PER_TASK}")
    print(f"[INFO] Max workers  : {MAX_WORKERS}")

    jobs = build_jobs()
    if not jobs:
        print("[ERROR] no docking jobs found")
        return

    total = len(jobs)
    print(f"[INFO] Total jobs   : {total}")

    max_workers = min(MAX_WORKERS, total)

    ok_count = 0
    fail_count = 0
    done = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(run_one_job, job): job for job in jobs}

        for fut in as_completed(future_to_job):
            job = future_to_job[fut]
            try:
                ok = fut.result()
            except Exception as e:
                print(
                    f"[FAIL] {job.target}/{job.method}/{job.ligand_id}: "
                    f"exception in worker {e}"
                )
                ok = False

            done += 1
            if ok:
                ok_count += 1
                status = "OK  "
            else:
                fail_count += 1
                status = "FAIL"

            # counter here: [done/total] ...
            print(
                f"[{done}/{total}] {status} "
                f"{job.target}/{job.method}/{job.ligand_id}"
            )

    print("=========== SUMMARY ===========")
    print(f"Total = {total}")
    print(f"OK    = {ok_count}")
    print(f"FAIL  = {fail_count}")
    print("================================")


if __name__ == "__main__":
    main()
