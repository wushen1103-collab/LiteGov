#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Firewall ablation heatmap for governance variants.

We start from raw batches (pass) for a single backbone, run the default LiteGov
selector, and then compare a set of variants against this default in terms of:

    - mean QED of the selected Top-K
    - mean SA of the selected Top-K
    - scaffold diversity of the selected Top-K
    - selection-stage wall-clock time (seconds)

For each variant we compute paired deltas (variant - default) across (tau, seed)
combinations and aggregate them by averaging. The resulting 4×N matrix is then
visualised as a "firewall" style heatmap.

Example:
    python scripts/fig_firewall.py \\
        --pass-dir results/denovo/pass \\
        --taus 0.8 1.0 1.2 \\
        --seeds 0 \\
        --topk 100 \\
        --variants "mmr=0.5" "wqed=0.7" "wqed=0.9" "scaler=zscore" "scaler=quantile" \\
        --out-prefix figs/firewall_ablation
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Optional seaborn (for nicer heatmap, but not required)
# ---------------------------------------------------------------------------

def try_import_seaborn(use: bool = True):
    if not use:
        return None
    try:
        import seaborn as sns  # type: ignore
        return sns
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RDKit utilities (Morgan fingerprints)
# ---------------------------------------------------------------------------

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator, DataStructs

RDLogger.DisableLog("rdApp.warning")

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_NBITS,
)


def smiles_to_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _MORGAN_GEN.GetFingerprint(mol)


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn, mx = np.min(x), np.max(x)
    eps = 1e-6 if mx - mn == 0 else 0.0
    return (x - mn) / (mx - mn + eps)


def zscore_then_minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu, sd = np.mean(x), np.std(x, ddof=0)
    z = (x - mu) / (sd if sd != 0 else 1.0)
    return minmax(z)


def quantile_scaler(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x))
    return ranks / max(len(x) - 1, 1)


SCALERS: Dict[str, callable] = {
    "minmax": minmax,
    "zscore": zscore_then_minmax,
    "quantile": quantile_scaler,
}


# ---------------------------------------------------------------------------
# Selector (LiteGov + MMR)
# ---------------------------------------------------------------------------

def select_topk(
    df: pd.DataFrame,
    w_qed: float = 0.8,
    scaler: str = "minmax",
    mmr_alpha: float = None,
    topk: int = 100,
    nbits: int = 2048,        # kept for CLI compatibility, not used explicitly
    mmr_top_p: int = None,
):
    """
    PAINS-first -> batch normalization -> q = w_qed * QED' + (1-w_qed) * (1 - SA')

    - If mmr_alpha is None:
        select Top-K by q (stable sort).
    - Otherwise:
        run an MMR-style selection within the top-P items ranked by q.
    """
    t0 = time.time()

    work = df[df["PAINS"] == 0].copy()
    if work.empty:
        return work.head(0).copy(), {"select_ms": 0.0}

    # scale
    if scaler not in SCALERS:
        raise KeyError(f"Unknown scaler: {scaler}")
    scale_fn = SCALERS[scaler]
    xq = scale_fn(work["QED"].values)
    xs = scale_fn(work["SA"].values)

    w_sa = 1.0 - w_qed
    q = w_qed * xq + w_sa * (1.0 - xs)

    work = work.reset_index(drop=True)
    work["__q"] = q

    # Fast path: no MMR
    if mmr_alpha is None:
        idx = np.argsort(-work["__q"].values, kind="mergesort")[:topk]
        sel = work.iloc[idx].copy()
        return sel, {"select_ms": (time.time() - t0) * 1000.0}

    # MMR path
    alpha = float(mmr_alpha)
    order = np.argsort(-work["__q"].values, kind="mergesort")
    P = int(mmr_top_p) if (mmr_top_p is not None) else int(
        min(len(order), max(5 * topk, topk))
    )
    order = order[:P]
    cand = work.iloc[order].reset_index(drop=True)

    smiles = cand["smiles"].astype(str).tolist()
    fps = [smiles_to_fp(s) for s in smiles]
    keep = [i for i, fp in enumerate(fps) if fp is not None]
    if not keep:
        idx = np.argsort(-cand["__q"].values, kind="mergesort")[:topk]
        sel = cand.iloc[idx].copy()
        return sel, {"select_ms": (time.time() - t0) * 1000.0}

    cand = cand.iloc[keep].reset_index(drop=True)
    fps = [fps[i] for i in keep]

    K = min(topk, len(cand))
    selected_idx = [0]  # pick the largest-q item first

    max_sim = np.zeros(len(cand))
    sims0 = np.array(DataStructs.BulkTanimotoSimilarity(fps[0], fps))
    max_sim = np.maximum(max_sim, sims0)

    for _ in range(1, K):
        scores = alpha * cand["__q"].values - (1.0 - alpha) * max_sim
        scores[selected_idx] = -np.inf
        j = int(np.argmax(scores))
        selected_idx.append(j)
        sims_new = np.array(DataStructs.BulkTanimotoSimilarity(fps[j], fps))
        max_sim = np.maximum(max_sim, sims_new)

    sel = cand.iloc[selected_idx].copy()
    return sel, {"select_ms": (time.time() - t0) * 1000.0}


# ---------------------------------------------------------------------------
# Metrics and IO
# ---------------------------------------------------------------------------

def metrics_on(sel: pd.DataFrame) -> Dict[str, float]:
    if sel.empty:
        return {"QED": np.nan, "SA": np.nan, "Scaffold": np.nan}
    qed = float(np.mean(sel["QED"].values))
    sa = float(np.mean(sel["SA"].values))
    scaff = sel["scaffold"].astype(str).tolist()
    div = len(set(scaff)) / max(len(scaff), 1)
    return {"QED": qed, "SA": sa, "Scaffold": div}


from typing import Optional, List

def load_batch(pass_dir: Path, tau: float, seed: int) -> pd.DataFrame:
    """
    Locate a raw batch CSV for a given (tau, seed) under `pass_dir`.

    We do NOT assume a fixed file-name template. Instead we:
      - list all *.csv in pass_dir (excluding summary.csv)
      - keep those whose filename contains both "tau<tau_str>" and "seed<seed>"
      - if exactly one match is found, use it
      - otherwise raise a helpful error listing available files

    This allows the script to work with directories like:
      results/genmol/raw/
    where files may be named, for example:
      genmol_raw_tau0.8_r1.0_seed0.csv
      pass_tau0.8_seed0_genmol.csv
    as long as "tau0.8" and "seed0" both appear in the file name.
    """
    pass_dir = Path(pass_dir)
    if not pass_dir.exists():
        raise FileNotFoundError(f"pass-dir does not exist: {pass_dir}")

    tau_str = str(tau)          # e.g. "0.8", "1.0"
    seed_str = str(seed)        # e.g. "0"

    candidates: List[Path] = []
    for p in pass_dir.glob("*.csv"):
        if p.name.lower() == "summary.csv":
            continue
        name = p.name
        if f"tau{tau_str}" in name and f"seed{seed_str}" in name:
            candidates.append(p)

    if len(candidates) == 0:
        all_csv = [p.name for p in pass_dir.glob("*.csv")]
        raise FileNotFoundError(
            f"No CSV found in {pass_dir} matching tau={tau}, seed={seed}.\n"
            f"Looked for filenames containing 'tau{tau_str}' and 'seed{seed_str}'.\n"
            f"Available CSVs: {all_csv}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple CSVs match tau={tau}, seed={seed} in {pass_dir}: "
            f"{[p.name for p in candidates]}"
        )

    path = candidates[0]
    df = pd.read_csv(path)

    required = ["smiles", "QED", "SA", "PAINS", "scaffold"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing} in {path}")

    return df

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def set_figure_style() -> None:
    """Match the style used in other figures (QED–SA, Tanimoto, Pareto, docking)."""
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
            "axes.grid": False,
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-dir", type=str, default="results/denovo/pass")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--wqed", type=float, default=0.8)
    ap.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["mmr=0.5", "wqed=0.7", "wqed=0.9", "scaler=zscore", "scaler=quantile"],
        help="Variants: mmr=alpha | wqed=x | scaler={minmax,zscore,quantile}",
    )
    ap.add_argument("--mmr-top-p", type=int, default=None)
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument("--out-prefix", type=str, default="figs/firewall_ablation")
    ap.add_argument("--font", type=str, default="DejaVu Sans")
    ap.add_argument("--no-seaborn", action="store_true")
    ap.add_argument(
        "--cbar-limit",
        type=float,
        default=None,
        help="Limit colorbar range (symmetric), e.g. 0.02 -> vmin=-0.02,vmax=+0.02",
    )
    args = ap.parse_args()

    # Global style
    set_figure_style()
    plt.rcParams["font.family"] = args.font

    sns = try_import_seaborn(use=not args.no_seaborn)

    pass_dir = Path(args.pass_dir)

    # Parse variant configs
    variants: List[Tuple[str, Dict[str, object]]] = []
    for v in args.variants:
        if v.startswith("mmr="):
            alpha_str = v.split("=", 1)[1]
            variants.append(
                (f"MMR on (α={alpha_str})", dict(mmr_alpha=float(alpha_str)))
            )
        elif v.startswith("wqed="):
            w_str = v.split("=", 1)[1]
            variants.append((f"Weight wQED={w_str}", dict(w_qed=float(w_str))))
        elif v.startswith("scaler="):
            sc = v.split("=", 1)[1].lower()
            if sc not in SCALERS:
                raise ValueError(f"Unknown scaler: {sc}")
            pretty = "z-score" if sc == "zscore" else sc
            variants.append((f"Scaler {pretty}", dict(scaler=sc)))
        else:
            raise ValueError(f"Unrecognized variant: {v}")

    # Compute paired deltas per (tau, seed)
    paired_rows: List[Dict[str, float]] = []
    for tau in args.taus:
        for seed in args.seeds:
            df = load_batch(pass_dir, tau, seed)
            base_sel, base_info = select_topk(
                df,
                w_qed=args.wqed,
                scaler="minmax",
                mmr_alpha=None,
                topk=args.topk,
                nbits=args.nbits,
            )
            base = metrics_on(base_sel)
            base_rt = base_info["select_ms"]

            for name, cfg in variants:
                p = dict(
                    w_qed=args.wqed,
                    scaler="minmax",
                    mmr_alpha=None,
                    topk=args.topk,
                    nbits=args.nbits,
                    mmr_top_p=args.mmr_top_p,
                )
                p.update(cfg)
                sel, info = select_topk(df, **p)
                m = metrics_on(sel)
                paired_rows.append(
                    {
                        "tau": tau,
                        "seed": seed,
                        "variant": name,
                        "dQED": m["QED"] - base["QED"],
                        "dSA": m["SA"] - base["SA"],
                        "dScaffold": m["Scaffold"] - base["Scaffold"],
                        "dSeconds_rank": (info["select_ms"] - base_rt) / 1000.0,
                    }
                )

    robust = pd.DataFrame(paired_rows)

    # Aggregate means (across tau, seed) and keep the CLI variant order
    variant_order = [name for (name, _) in variants]
    agg = robust.groupby("variant", as_index=False).mean(numeric_only=True)
    agg["variant"] = pd.Categorical(agg["variant"], categories=variant_order, ordered=True)
    agg = agg.sort_values("variant")

    metric_rows = ["QED", "SA", "Scaffold", "Seconds"]
    cols = agg["variant"].tolist()
    mat = np.stack(
        [
            agg["dQED"].values,
            agg["dSA"].values,
            agg["dScaffold"].values,
            agg["dSeconds_rank"].values,
        ],
        axis=0,
    )
    heat_df = pd.DataFrame(mat, index=metric_rows, columns=cols)

    # Save CSVs
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    heat_df.to_csv(out_prefix.with_suffix(".csv"), index=True)
    robust.to_csv(
        out_prefix.with_name(out_prefix.name + "_paired_deltas.csv"),
        index=False,
    )

    # Draw firewall heatmap
    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    if args.cbar_limit is not None:
        vlim = float(abs(args.cbar_limit))
    else:
        vlim = float(np.max(np.abs(heat_df.values))) if heat_df.size else 0.0
    if vlim == 0.0:
        vmin, vmax = -0.01, 0.01
    else:
        vmin, vmax = -vlim, vlim

    if sns is not None:
        hm = sns.heatmap(
            heat_df,
            cmap="coolwarm",
            center=0.0,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            annot_kws={"size": 7},
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Δ vs default", "shrink": 0.8},
            ax=ax,
        )
    else:
        im = ax.imshow(
            heat_df.values,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
        )
        for i in range(heat_df.shape[0]):
            for j in range(heat_df.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{heat_df.values[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Δ vs default", fontsize=9)

    ax.set_yticklabels(heat_df.index.tolist(), rotation=0, fontsize=8)
    ax.set_xticklabels(
        heat_df.columns.tolist(),
        rotation=35,
        ha="center",
        va="top",
        fontsize=7,
    )

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout(pad=0.2)

    out_pdf = out_prefix.with_suffix(".pdf")
    out_png = out_prefix.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_pdf}")
    print(f"[OK] Saved: {out_png}")


if __name__ == "__main__":
    main()
