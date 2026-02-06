import os, re, json, sys, math
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.rdmolfiles import SDMolSupplier
from scipy import stats

sys.path.insert(0, os.path.abspath("scripts"))
try:
    import SA_Score
    SA_OK = True
    print("[DIAG] SA_Score loaded from:", SA_Score.__file__)
except Exception as e:
    SA_OK = False
    print("[DIAG] SA_Score import FAILED:", e)

def calc_sa(mol):
    if mol is None or not SA_OK: 
        return np.nan
    try:
        return SA_Score.sascorer.calculateScore(mol)
    except Exception as e:
        return np.nan

SMI_KEYS = ["smiles","SMILES","canonical_smiles","SMI","smi"]
def find_smiles_col(df: pd.DataFrame):
    for k in df.columns:
        if 'smile' in k.lower(): return k
    return None

def extract_tau_from_path(path: str):
    for part in [os.path.basename(path), os.path.basename(os.path.dirname(path))]:
        m = re.search(r'(?:^|[_\-])(?:tau|t)[=_\-]?([01](?:\.\d+)?)', part, re.I)
        if m: return float(m.group(1))
        m2 = re.search(r'(?<!\d)(0\.(?:8|9)|1\.0|1\.2)(?!\d)', part)
        if m2: return float(m2.group(1))
    return np.nan

def infer_method(path: str):
    p = path.lower()
    if "govern" in p or "heavy" in p: return "govern"
    if "filter+rank" in p or "filter_rank" in p or "rank" in p or "rerank" in p: return "filter+rank"
    if "filter" in p: return "filter"
    if "pass" in p or "raw" in p or "orig" in p or "denovo" in p: return "pass"
    return "unknown"

def read_smiles(fp: str):
    rows=[]
    low = fp.lower()
    try:
        if low.endswith(".csv"):
            df = pd.read_csv(fp, on_bad_lines='skip')
            col = find_smiles_col(df)
            if col is None: return []
            vals = df[col].dropna().astype(str).tolist()
            for s in vals:
                mol = Chem.MolFromSmiles(s.strip())
                if mol is not None:
                    rows.append(Chem.MolToSmiles(mol))
        elif low.endswith(".jsonl"):
            with open(fp,"r",encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    obj = json.loads(line)
                    for k in SMI_KEYS:
                        if k in obj:
                            s=str(obj[k]).strip()
                            if Chem.MolFromSmiles(s):
                                rows.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
                            break
        elif low.endswith(".json"):
            obj = json.load(open(fp,"r",encoding="utf-8"))
            if isinstance(obj, list):
                for o in obj:
                    for k in SMI_KEYS:
                        if k in o:
                            s=str(o[k]).strip()
                            if Chem.MolFromSmiles(s):
                                rows.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
                            break
            elif isinstance(obj, dict):
                for k in SMI_KEYS:
                    if k in obj:
                        s=str(obj[k]).strip()
                        if Chem.MolFromSmiles(s):
                            rows.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
                        break
        elif low.endswith(".smi") or low.endswith(".smiles"):
            with open(fp,"r",encoding="utf-8") as f:
                for line in f:
                    s=line.strip().split()[0]
                    if s and Chem.MolFromSmiles(s):
                        rows.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
        elif low.endswith(".sdf"):
            suppl = SDMolSupplier(fp)
            for mol in suppl:
                if mol is None: continue
                rows.append(Chem.MolToSmiles(mol))
    except Exception:
        return []
    return rows

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="results/denovo")
args = ap.parse_args()

scan_ext = re.compile(r'\.(csv|json|jsonl|smi|smiles|sdf)$', re.I)
files=[]
for root,_,fs in os.walk(args.root):
    for fn in fs:
        if scan_ext.search(fn):
            fp = os.path.join(root, fn)
            files.append(fp)

used, skipped = [], []
for fp in files:
    smis = read_smiles(fp)
    if smis:
        used.append((fp, infer_method(fp), extract_tau_from_path(fp), smis))
    else:
        skipped.append(fp)

print(f"[DIAG] files total={len(files)}, used={len(used)}, skipped(no SMILES)={len(skipped)}")
if skipped[:5]:
    print("[DIAG] examples skipped:", skipped[:5])

records=[]
n_sa_ok=0
for fp, method, tau, smis in tqdm(used, desc="Collecting"):
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        sa = calc_sa(mol)
        if not np.isnan(sa): n_sa_ok += 1
        mw = Descriptors.MolWt(mol) if mol else np.nan
        logp = Crippen.MolLogP(mol) if mol else np.nan
        hbd = Lipinski.NumHDonors(mol) if mol else np.nan
        hba = Lipinski.NumHAcceptors(mol) if mol else np.nan
        viol = (int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)) if mol else np.nan
        records.append({"file": fp, "method": method, "tau": tau, "smiles": s,
                        "SA": sa, "MW": mw, "logP": logp, "HBD": hbd, "HBA": hba, "LipinskiViol": viol})

print(f"[DIAG] SA non-NaN count = {n_sa_ok}")

det = pd.DataFrame(records)
if det.empty:
    raise SystemExit("Parsed zero molecules with valid SMILES.")

det["tau_str"] = det["tau"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "NA")
os.makedirs("analysis_out", exist_ok=True)
det.to_csv("analysis_out/druglikeness_detailed.csv", index=False)

def ci95(series: pd.Series):
    s = series.dropna().astype(float)
    if len(s) < 2: return (np.nan, np.nan)
    m = s.mean(); half = stats.t.ppf(0.975, len(s)-1) * stats.sem(s)
    return (m, half)

def agg_block(g):
    sa_m, sa_h = ci95(g["SA"])
    return pd.Series({
        "N": len(g),
        "SA_mean±CI": (f"{sa_m:.2f} ± {sa_h:.2f}" if pd.notna(sa_m) else "nan ± nan"),
        "SA ≤3 (%)": (g["SA"] <= 3).mean() * 100,
        "Lipinski viol. (%)": (g["LipinskiViol"] > 0).mean() * 100,
        "Fully obey (%)": (g["LipinskiViol"] == 0).mean() * 100
    })

out = det.groupby(["tau_str","method"], dropna=False).apply(agg_block).reset_index()
out = out.rename(columns={"tau_str":"τ","method":"Method"})
out.to_csv("analysis_out/druglikeness_summary.csv", index=False)


view = out.copy()
for c in ["SA ≤3 (%)","Lipinski viol. (%)","Fully obey (%)"]:
    view[c] = view[c].map(lambda x: f"{x:.1f}")
print("\n=== Drug-likeness & SA Summary (by τ × Method) ===")
print(view.to_string(index=False))
print("\nSaved:")
print(" - analysis_out/druglikeness_detailed.csv")
print(" - analysis_out/druglikeness_summary.csv")
