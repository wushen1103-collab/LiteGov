#!/usr/bin/env python3
import argparse, glob, os, time
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_smiles_col(df):
    for c in ["smiles","SMILES","smile","SMILE"]:
        if c in df.columns:
            return c
    raise ValueError("No SMILES column found. Expected one of: smiles, SMILES, smile, SMILE.")

def ensure_mol(df, smiles_col):
    # Always reconstruct from SMILES because CSV may stringify RDKit Mol objects
    mols = []
    for s in df[smiles_col]:
        m = Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
        mols.append(m)
    return mols

def mol_to_scaffold_smiles(m):
    if m is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(scaf, canonical=True) if scaf is not None else None

def fp_morgan(m, radius=2, nBits=2048):
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)

def mmr_select(df, score_col="score", K=100, lamb=0.8, fps=None):
    """
    Greedy MMR selection:
        argmax lambda * score(i) - (1-lambda) * max_j sim(i, j in S)
    """
    n = len(df)
    if n == 0:
        return []
    if fps is None:
        raise ValueError("fps required")

    scores = df[score_col].values if score_col in df.columns else None
    if scores is None:
        raise ValueError(f"score column '{score_col}' not found in dataframe.")

    selected = []
    remaining = list(range(n))

    def sim(i, j):
        if fps[i] is None or fps[j] is None:
            return 0.0
        return DataStructs.TanimotoSimilarity(fps[i], fps[j])

    for _ in range(min(K, n)):
        if not remaining:
            break
        best_i, best_val = None, -1e9
        for i in remaining:
            sim_pen = max(sim(i, j) for j in selected) if selected else 0.0
            val = lamb * float(scores[i]) - (1.0 - lamb) * sim_pen
            if val > best_val:
                best_val, best_i = val, i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)
    return selected

def basic_metrics(df, mols):
    uniq = df["smiles"].nunique() / len(df) if len(df) > 0 else 0.0
    scaffolds = [mol_to_scaffold_smiles(m) for m in mols]
    scaf_div = len(set([s for s in scaffolds if s is not None])) / len(df) if len(df) > 0 else 0.0
    qed_vals = []
    for m in mols:
        try:
            qed_vals.append(QED.qed(m) if m is not None else float("nan"))
        except Exception:
            qed_vals.append(float("nan"))
    qed_mean = float(pd.Series(qed_vals).mean())
    pains_rate = float(df["pains"].mean()) if "pains" in df.columns else float("nan")
    return uniq, scaf_div, qed_mean, pains_rate

def main():
    ap = argparse.ArgumentParser(description="MMR/diversity-aware selection over governed candidates.")
    ap.add_argument("--in", dest="input_glob", type=str, required=True,
                    help="Input CSV or glob, e.g. results/denovo/govern/*.csv")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")
    ap.add_argument("--k", type=int, default=100, help="Number of molecules to select")
    ap.add_argument("--lambda", dest="lamb", type=float, default=0.8,
                    help="MMR lambda in [0,1], higher = more score emphasis")
    ap.add_argument("--score-col", type=str, default="score", help="Column name for governance score")
    ap.add_argument("--radius", type=int, default=2, help="MorganFP radius")
    ap.add_argument("--nbits", type=int, default=2048, help="MorganFP nBits")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for f in files:
        t0 = time.time()
        df = pd.read_csv(f)
        try:
            smiles_col = get_smiles_col(df)
        except Exception:
            print(f"⚠️ Skip {os.path.basename(f)} (no SMILES column).")
            continue

        if args.score_col not in df.columns:
            print(f"⚠️ Skip {os.path.basename(f)} (no score column '{args.score_col}').")
            continue
        score_series = pd.to_numeric(df[args.score_col], errors="coerce")
        valid_mask = score_series.notna()
        if valid_mask.sum() == 0:
            print(f"⚠️ Skip {os.path.basename(f)} (all scores are NaN).")
            continue
        df = df[valid_mask].reset_index(drop=True)

        mols = ensure_mol(df, smiles_col)
        if smiles_col != "smiles":
            df["smiles"] = df[smiles_col]

        fps = [fp_morgan(m, radius=args.radius, nBits=args.nbits) for m in mols]
        sel_idx = mmr_select(df, score_col=args.score_col, K=args.k, lamb=args.lamb, fps=fps)
        if not sel_idx:
            print(f"⚠️ No indices selected for {os.path.basename(f)} (MMR returned empty).")
            continue

        df_sel = df.iloc[sel_idx].copy()
        mols_sel = [mols[i] for i in sel_idx]
        uniq, scaf_div, qed_mean, pains_rate = basic_metrics(df_sel, mols_sel)
        seconds = time.time() - t0

        base = os.path.basename(f)
        out_csv = os.path.join(args.outdir, f"mmr_{base}")
        df_sel.to_csv(out_csv, index=False)

        rows.append(dict(
            file=base, k=args.k, lamb=args.lamb,
            uniqueness=uniq, scaf_div=scaf_div, qed_mean=qed_mean,
            pains_rate=pains_rate, seconds=seconds, out=out_csv
        ))
        print(f"✅ {base} -> {out_csv} | K={args.k} λ={args.lamb} | "
              f"uniq={uniq:.3f} scaf_div={scaf_div:.3f} QED={qed_mean:.3f} time={seconds:.2f}s")

    if rows:
        summary_path = os.path.join(args.outdir, "mmr_summary.csv")
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"✅ Summary saved at {summary_path}")
    else:
        print("⚠️ No summary written (no valid inputs).")

if __name__ == "__main__":
    main()
