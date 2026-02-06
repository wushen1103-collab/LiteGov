#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tanimoto diversity figure for generated libraries (single backbone).

We compare five governance lines:

    Raw, QED, RuleKit, LiteGov, HeavyGov

For each line we:
    - aggregate SMILES from results/<backbone>/<mode>/*.csv
    - convert to Morgan fingerprints (radius 2, 2048 bits)
    - sample a fixed number of pairwise Tanimoto similarities
    - estimate a 1D density with Gaussian KDE

Output:
    figs/tanimoto_diversity.png
    figs/tanimoto_diversity.pdf
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

# silence RDKit warnings
RDLogger.DisableLog("rdApp.warning")

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGS_DIR = ROOT / "figs"

# If your GenMol results are under results/denovo, change this to "denovo".
BACKBONE_DIR = "genmol"

# (legend label, subdirectory name, method key for colors)
MODES: List[Tuple[str, str, str]] = [
    ("Raw", "raw", "raw"),
    ("QED", "qed", "qed"),
    ("RuleKit", "rulekit", "rulekit"),
    ("LiteGov", "lite", "lite"),
    ("HeavyGov", "heavy", "heavy"),
]

# colors aligned with docking figures
METHOD_COLORS: Dict[str, str] = {
    "raw": "#b0b0b0",      # light gray
    "qed": "#a6cee3",      # light blue
    "rulekit": "#fdbf6f",  # light orange
    "lite": "#b2df8a",     # light green
    "heavy": "#cab2d6",    # light purple
}

MAX_MOLS_PER_LINE = 2000
MAX_PAIRS_PER_LINE = 50000
RANDOM_SEED = 0

# Morgan fingerprint generator (shared)
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_NBITS,
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def set_figure_style() -> None:
    """Match the style used in docking figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_smiles_column(df: pd.DataFrame) -> str:
    candidates = [
        "smiles",
        "SMILES",
        "Smiles",
        "SMILE",
        "smiles_can",
        "canonical_smiles",
        "can_smiles",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError("No SMILES-like column found.")


def load_smiles_for_mode(backbone_dir: str, mode_dir: str) -> List[str]:
    root = RESULTS_DIR / backbone_dir / mode_dir
    if not root.exists():
        print(f"[WARN] directory not found: {root}")
        return []

    files = sorted(p for p in root.glob("*.csv") if p.name.lower() != "summary.csv")
    if not files:
        print(f"[WARN] no CSV files found in {root}")
        return []

    smiles: List[str] = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] failed to read {path}: {e}")
            continue

        try:
            col = find_smiles_column(df)
        except KeyError:
            print(f"[WARN] skipping {path.name}: no SMILES column found")
            continue

        vals = df[col].dropna().astype(str).tolist()
        smiles.extend(vals)

    smiles = list(dict.fromkeys(smiles))  # deduplicate

    if not smiles:
        print(f"[WARN] no valid SMILES for {backbone_dir}/{mode_dir}")
    return smiles


def smiles_to_morgan(smiles: List[str]):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = MORGAN_GENERATOR.GetFingerprint(mol)
        fps.append(fp)
    return fps


def sample_tanimoto_pairs(fps, max_pairs: int = MAX_PAIRS_PER_LINE) -> np.ndarray:
    n = len(fps)
    if n < 2:
        return np.array([], dtype=float)

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        sims: List[float] = []
        for i in range(n):
            sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :]))
        return np.asarray(sims, dtype=float)

    rng = np.random.default_rng(RANDOM_SEED)
    sims: List[float] = []
    while len(sims) < max_pairs:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        sims.append(sim)
    return np.asarray(sims, dtype=float)


def compute_density(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if values.size < 10:
        return np.zeros_like(x_grid)
    kde = gaussian_kde(values)
    return kde(x_grid)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(RANDOM_SEED)
    set_figure_style()

    tanimoto_data: Dict[str, np.ndarray] = {}

    for label, mode_dir, key in MODES:
        smiles = load_smiles_for_mode(BACKBONE_DIR, mode_dir)
        if not smiles:
            tanimoto_data[key] = np.array([], dtype=float)
            continue

        if len(smiles) > MAX_MOLS_PER_LINE:
            rng = np.random.default_rng(RANDOM_SEED)
            idx = rng.choice(len(smiles), size=MAX_MOLS_PER_LINE, replace=False)
            smiles = [smiles[i] for i in idx]

        fps = smiles_to_morgan(smiles)
        sims = sample_tanimoto_pairs(fps)
        tanimoto_data[key] = sims

        if sims.size:
            print(
                f"[INFO] {label}: {len(smiles)} molecules, "
                f"{sims.size} pairs, median={np.median(sims):.3f}"
            )
        else:
            print(f"[INFO] {label}: insufficient data")

    x_grid = np.linspace(0.0, 1.0, 400)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    for label, _, key in MODES:
        sims = tanimoto_data.get(key, np.array([]))
        if sims.size == 0:
            continue

        color = METHOD_COLORS[key]
        y = compute_density(sims, x_grid)
        ax.plot(
            x_grid,
            y,
            label=label,
            linewidth=2.0,
            color=color,
        )

        median_val = float(np.median(sims))
        ax.axvline(
            median_val,
            linestyle="--",
            linewidth=1.2,
            color=color,
            alpha=0.9,
        )

    ax.set_xlabel("Tanimoto similarity")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)

    ax.tick_params(axis="both", which="both", direction="out")
    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.legend(loc="upper right", frameon=False, ncol=1)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIGS_DIR / "tanimoto_diversity.png"
    out_pdf = FIGS_DIR / "tanimoto_diversity.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")


if __name__ == "__main__":
    main()
