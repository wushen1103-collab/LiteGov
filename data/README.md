# Data notes

This GitHub-ready package keeps only small seed files and processed statistics needed to reproduce the paper tables/figures.

## What is included
- `moler_samples/`: small seed sets used in the experiments
- `moler_raw.csv`, `moler_samples_raw.smi`: compact raw seed listings
- Processed paper tables in `../tables/` and figure CSVs in `../figs/`

## What is NOT included
To keep the repository lightweight, we do **not** include large raw docking outputs (e.g., Vina logs / PDBQT outputs) or any large model checkpoints.

If you need to regenerate docking outputs, use the scripts under `../scripts/` and provide your own docking backend (e.g., AutoDock Vina) and receptor preparation.
