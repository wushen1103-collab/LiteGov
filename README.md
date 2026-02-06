# LiteGov / GovMol-Lite

LiteGov (a lightweight governance layer) helps you **filter**, **rank**, and **select diverse** molecules from de novo generation outputs.  
This repository also provides scripts to compute common drug-likeness descriptors (e.g., QED / SA / LogP, Ro5/Veber-related properties) and to aggregate results into tables/figures.

## What’s included

- **Filtering**: validity checks, deduplication, optional PAINS filtering
- **Ranking**: a lightweight score combining QED and SA for Top-K selection
- **Diversity-aware selection**: MMR selection using Morgan/ECFP fingerprints
- **Batch analysis**: descriptor computation + summary statistics
- **Optional docking utilities**: helper scripts for preparing ligands/configs, running AutoDock Vina, and collecting docking scores

> Some scripts assume a `results/denovo/...` layout (from ablation runs), while example outputs may also be organized under `results/` by generator/method.

---

## Installation

From the repo root:

### Option A: Conda
```bash
conda env create -f env/govmol_env.yml
conda activate govmol
```

### Option B: pip
```bash
pip install -r env/requirements.txt
```

---

## Quickstart (run on included example files)

### 1) Compute batch drug-likeness summaries
```bash
python scripts/analyze_druglikeness_batch.py --root results
```

Typical outputs (depending on script options):
- `analysis_out/druglikeness_summary.csv`
- `analysis_out/druglikeness_detailed.csv`

### 2) Build summary tables from result CSVs
```bash
python scripts/make_table3_top100.py
python scripts/make_table4_docking_summary.py
```

> If your results are not under the default paths expected by a script, check its CLI arguments (some support `--root` / `--in-glob`) or edit the path variables at the top of the script.

---

## Running LiteGov on your own generator outputs

### Input format
Most scripts expect a CSV file with at least:
- `smiles` (one SMILES per row)

If you already have descriptor columns (e.g., `QED`, `SA`, `logP`, `pains_ok`), some steps can reuse them.

### Typical pipeline
1) **Prepare / normalize raw outputs** (optional, depends on your generator format)
2) **Filter** candidates (validity, PAINS, etc.)
3) **Rank** candidates with LiteGov score (QED + SA)
4) **Select** Top-K with optional **MMR** (diversity-aware)

Example commands (adjust paths to your setup):
```bash
python scripts/ablate_filter.py
python scripts/ablate_filter_rank.py
python scripts/mmr_select.py --k 100 --lambda 0.8
```

---

## Optional: Docking utilities

This repo includes scripts such as:
- `scripts/init_docking_configs.py`
- `scripts/run_vina_batch.py`
- `scripts/collect_dock_scores.py`

To use docking you typically need:
- AutoDock Vina installed and available in `PATH`
- receptor structures + docking box settings
- compute resources (batch docking can be slow)

Since docking setups differ across targets, read the header comments in each script and adapt paths/configs accordingly.

---

## Configuration

`configs/governance.yml` contains key parameters for filtering/ranking (e.g., thresholds, weights, Top-K).  
Keep this file in sync with your experiments to avoid mismatch between “paper settings” and “code defaults”.

---

## Repository layout

- `scripts/` — main pipeline scripts (filter/rank/MMR/docking/analysis)
- `configs/` — governance configuration (`governance.yml`)
- `env/` — environment specifications
- `results/` — example outputs / expected output layout
- `genmol-raw/` — example raw generator outputs (CSV)
- `analysis_out/` — aggregated analysis outputs
- `tables/` — exported tables (CSV/MD)
- `figs/` — small figure data (CSV etc.)
- `rdkit/` — RDKit helper code (if used by scripts)
- `molecule-generation/` — related code (vendor/submodule copy)
- `data/` — small reference inputs (or pointers/format examples)

---

## License

See `LICENSE`.
