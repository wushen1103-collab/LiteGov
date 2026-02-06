#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
python scripts/make_table1_overview.py
python scripts/make_table2_global_metrics.py
python scripts/make_table3_top100.py
python scripts/make_table4_docking_summary.py

# Optional: regenerate figures (CSV inputs are already provided)
python scripts/fig_firewall.py || true
python scripts/fig_qed_sa.py || true
python scripts/fig_tanimoto.py || true
python scripts/fig_pareto.py || true

echo "Done. Tables are in ./tables and figure CSVs are in ./figs."
