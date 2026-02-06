#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-K per (seed,temp) on LiteGov vs HeavyGov, then aggregate mean +/- sd.

Key fixes:
  - Method detection no longer treats 'gov' as HeavyGov (avoid matching '/govemol', '/govmol-lite').
  - If a CSV lacks a score column, use --fallback-score to synthesize one (default: qed).

Outputs:
  figs/topk_audit_files.csv
  figs/topk_availability.csv
  figs/paper_table_topk{K}.csv/.tex   (when intersection not empty)

Usage:
  python -X utf8 build_paper_table_topk.py --k 300
  python -X utf8 build_paper_table_topk.py --list-availability
  python -X utf8 build_paper_table_topk.py --k 300 --auto-k --min-k 80
  python -X utf8 build_paper_table_topk.py --root /root/govmol-lite --k 300
  python -X utf8 build_paper_table_topk.py --k 300 --fallback-score qed
"""

from __future__ import annotations
from pathlib import Path
import argparse, glob, re, sys, hashlib
import numpy as np
import pandas as pd

# -------------------- aliases & column candidates --------------------
ALIASES_LITE  = {
    "lite","litegov","filter","filter_rank","filter+rank","filterrank",
    "rank","lite-gov","lite_gov"
}
ALIASES_HEAVY = {
    "heavy","heavygov","govern","govern_mmr","governmmr"
}

SCORE_CANDS   = {
    "score","docking_score","dockscore","vina_score","affinity",
    "binding_affinity","result","pred","prediction","dock_score","score_vina"
}
SMILES_CANDS  = {
    "smiles_can","canonical_smiles","canon_smiles","smiles_canon",
    "can_smiles","smiles","SMILES"
}
SEED_COLS     = {"seed","s","rep","run"}
TEMP_COLS     = {"temperature","temp","tau","t","kt","kT"}

EXTRA_SCORE_HINTS = ["score", "affin", "vina", "dock", "energy"]  

RE_INT        = r"(\d+)"
RE_FLOAT      = r"(\d+(?:\.\d+)?)"
SEED_PATTERNS = [re.compile(r"(?i)(?:^|[/_\-\.])(seed|s|rep|run)[_=:-]?" + RE_INT   + r"(?:[/_\-\.]|$)")]
TEMP_PATTERNS = [re.compile(r"(?i)(?:^|[/_\-\.])(temperature|temp|tau|t|kt)[_=:-]?" + RE_FLOAT + r"(?:[/_\-\.]|$)")]

# -------------------- RDKit (optional, for descriptors/fallback) --------------------
def _maybe_import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import QED, Descriptors, Crippen, Lipinski, rdMolDescriptors
        sys.path.insert(0, "scripts")
        import sascorer
        return Chem, QED, Descriptors, Crippen, Lipinski, rdMolDescriptors, sascorer
    except Exception:
        return None
RD = _maybe_import_rdkit()

# -------------------- helpers --------------------
def norm_method_from_path(p: str) -> str|None:
    """Check every path part (lower-cased) for alias substrings (no 'gov' generic)."""
    parts = [comp.lower() for comp in Path(p).parts]
    for comp in parts:
        if any(alias in comp for alias in ALIASES_HEAVY):
            return "HeavyGov"
    for comp in parts:
        if any(alias in comp for alias in ALIASES_LITE):
            return "LiteGov"
    return None

def pick_col(cols, candidates:set[str]) -> str|None:
    low = {c.lower(): c for c in cols}
    for name in candidates:
        if name in low:
            return low[name]
    return None

def parse_seed_from_path(p: str):
    s = str(p)
    for pat in SEED_PATTERNS:
        m = pat.search(s)
        if m:
            try: return int(m.group(2))
            except Exception: pass
    return None

def parse_temp_from_path(p: str):
    s = str(p)
    for pat in TEMP_PATTERNS:
        m = pat.search(s)
        if m:
            try: return float(m.group(2))
            except Exception: pass
    return None

def guess_score_col(df: pd.DataFrame) -> str | None:
    """
    Guess a numeric 'score' column if exact candidates are absent.
    Preference:
      1) any numeric column whose name contains one of EXTRA_SCORE_HINTS
      2) numeric 'rank' / 'order' / 'priority'
    """
    for c in df.columns:
        lc = c.lower()
        if any(h in lc for h in EXTRA_SCORE_HINTS):
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    for k in ["rank", "order", "priority"]:
        low = {x.lower(): x for x in df.columns}
        if k in low and pd.api.types.is_numeric_dtype(df[low[k]]):
            return low[k]
    return None

def detect_columns_or_path(df: pd.DataFrame, fpath: str, debug=False):
    seed_col = pick_col(df.columns, {c.lower() for c in SEED_COLS})
    temp_col = pick_col(df.columns, {c.lower() for c in TEMP_COLS})
    score_col= pick_col(df.columns, {c.lower() for c in SCORE_CANDS})
    smiles   = pick_col(df.columns, {c.lower() for c in SMILES_CANDS})
    seed_val = None if seed_col is not None else parse_seed_from_path(fpath)
    temp_val = None if temp_col is not None else parse_temp_from_path(fpath)

    if score_col is None:
        guessed = guess_score_col(df)
        if guessed is not None:
            score_col = guessed

    if debug:
        print(f"[detect] {fpath}")
        print(f"  seed_col={seed_col} seed_val={seed_val}  temp_col={temp_col} temp_val={temp_val}")
        print(f"  score_col={score_col}  smiles={smiles}")
    return seed_col, temp_col, score_col, smiles, seed_val, temp_val

def ensure_descriptors(df: pd.DataFrame):
    need = {
        "sa_ertl","qed","mw","logp","tpsa","heavy_atoms",
        "rings_aromatic","frac_csp3","murcko_scaffold","pains_n","pains_ok","smiles_can"
    }
    missing = [c for c in need if c not in df.columns]
    if not missing:
        return df
    if RD is None:
        raise RuntimeError("Descriptors missing and RDKit/sascorer not available. Provide these columns or install RDKit.")
    Chem, QED, Descriptors, Crippen, Lipinski, rdMolDescriptors, sascorer = RD
    from rdkit.Chem.Scaffolds import MurckoScaffold
    smi_col = pick_col(df.columns, {"smiles_can","canonical_smiles","can_smiles","smiles"})
    if smi_col is None:
        raise RuntimeError("No SMILES column to compute descriptors.")
    cans, sa, qed, mw, logp, tpsa, heavy, arom, sp3, scaf, pains_n, pains_ok = ([] for _ in range(12))
    for s in df[smi_col].astype(str):
        try: m = Chem.MolFromSmiles(s)
        except Exception: m = None
        if m is None:
            cans.append(np.nan); sa.append(np.nan); qed.append(np.nan); mw.append(np.nan)
            logp.append(np.nan); tpsa.append(np.nan); heavy.append(np.nan)
            arom.append(np.nan); sp3.append(np.nan); scaf.append(np.nan)
            pains_n.append(np.nan); pains_ok.append(np.nan); continue
        try: cans.append(Chem.MolToSmiles(m, isomericSmiles=True, canonical=True))
        except Exception: cans.append(np.nan)
        try: sa.append(float(sascorer.calculateScore(m)))
        except Exception: sa.append(np.nan)
        try: qed.append(float(QED.qed(m)))
        except Exception: qed.append(np.nan)
        try: mw.append(float(Descriptors.MolWt(m)))
        except Exception: mw.append(np.nan)
        try: logp.append(float(Crippen.MolLogP(m)))
        except Exception: logp.append(np.nan)
        try: tpsa.append(float(rdMolDescriptors.CalcTPSA(m)))
        except Exception: tpsa.append(np.nan)
        try: heavy.append(int(m.GetNumHeavyAtoms()))
        except Exception: heavy.append(np.nan)
        try: arom.append(int(rdMolDescriptors.CalcNumAromaticRings(m)))
        except Exception: arom.append(np.nan)
        try: sp3.append(float(rdMolDescriptors.CalcFractionCSP3(m)))
        except Exception: sp3.append(np.nan)
        try: scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception: scaf.append(np.nan)
        pains_n.append(np.nan); pains_ok.append(np.nan)
    if "smiles_can" not in df.columns: df["smiles_can"] = cans
    if "sa_ertl" not in df.columns:    df["sa_ertl"]    = sa
    if "qed" not in df.columns:        df["qed"]        = qed
    if "mw" not in df.columns:         df["mw"]         = mw
    if "logp" not in df.columns:       df["logp"]       = logp
    if "tpsa" not in df.columns:       df["tpsa"]       = tpsa
    if "heavy_atoms" not in df.columns:df["heavy_atoms"]= heavy
    if "rings_aromatic" not in df.columns: df["rings_aromatic"] = arom
    if "frac_csp3" not in df.columns:  df["frac_csp3"]  = sp3
    if "murcko_scaffold" not in df.columns: df["murcko_scaffold"] = scaf
    if "pains_n" not in df.columns:    df["pains_n"]    = pains_n
    if "pains_ok" not in df.columns:   df["pains_ok"]   = pains_ok
    return df

def replicate_stats(df: pd.DataFrame) -> dict:
    n = int(len(df))
    valid_mask = df["qed"].notna() & df["sa_ertl"].notna()
    valid_n = int(valid_mask.sum())
    valid_pct = 100.0 * valid_n / n if n>0 else np.nan
    uniq_n = int(df["smiles_can"].dropna().nunique()) if "smiles_can" in df.columns else valid_n
    uniq_pct = 100.0 * uniq_n / max(valid_n,1)
    scaf_n = int(df["murcko_scaffold"].dropna().nunique()) if "murcko_scaffold" in df.columns else np.nan
    scaf_pct = 100.0 * scaf_n / max(uniq_n,1) if uniq_n>0 else np.nan
    if "pains_ok" in df.columns and df["pains_ok"].notna().any():
        pains_pct = 100.0 * (~df["pains_ok"].astype(bool)).mean()
    elif "pains_n" in df.columns and df["pains_n"].notna().any():
        pains_pct = 100.0 * (df["pains_n"].fillna(0)>0).mean()
    else:
        pains_pct = np.nan
    def m(col): return float(df[col].mean()) if col in df.columns else np.nan
    return {
        "n": n, "valid_pct": valid_pct, "unique_pct": uniq_pct, "scaf_div_pct": scaf_pct,
        "qed_mean": m("qed"), "sa_mean": m("sa_ertl"),
        "mw_mean": m("mw"), "logp_mean": m("logp"), "tpsa_mean": m("tpsa"),
        "heavy_atoms_mean": m("heavy_atoms"), "rings_aromatic_mean": m("rings_aromatic"),
        "frac_csp3_mean": m("frac_csp3"), "pains_pct": pains_pct,
    }

def agg_mean_sd(df: pd.DataFrame, by: str) -> pd.DataFrame:
    metrics = ["valid_pct","unique_pct","scaf_div_pct","qed_mean","sa_mean","pains_pct",
               "mw_mean","logp_mean","tpsa_mean","heavy_atoms_mean","rings_aromatic_mean","frac_csp3_mean"]
    out=[]
    for m, g in df.groupby(by):
        row = {"Method": m, "Replicates": int(g.shape[0])}
        for col in metrics:
            mu = float(g[col].mean())
            sd = float(g[col].std(ddof=1)) if g.shape[0]>1 else 0.0
            row[col+"_mu"] = mu; row[col+"_sd"] = sd
        out.append(row)
    return pd.DataFrame(out)

def fmt_pm(mu, sd, k=3): return f"{mu:.{k}f} +/- {sd:.{k}f}"

def build_paper_table(agg: pd.DataFrame) -> pd.DataFrame:
    def pick(mu, sd, k): return fmt_pm(mu, sd, k)
    tbl = pd.DataFrame({
        "Method": agg["Method"],
        "Replicates": agg["Replicates"],
        "Valid (%)":       [pick(agg["valid_pct_mu"][i],        agg["valid_pct_sd"][i],        1) for i in agg.index],
        "Unique (%)":      [pick(agg["unique_pct_mu"][i],       agg["unique_pct_sd"][i],       1) for i in agg.index],
        "Scaffold (%)":    [pick(agg["scaf_div_pct_mu"][i],     agg["scaf_div_pct_sd"][i],     1) for i in agg.index],
        "QED (mean +/- sd)": [pick(agg["qed_mean_mu"][i],       agg["qed_mean_sd"][i],         3) for i in agg.index],
        "SA (mean +/- sd)":  [pick(agg["sa_mean_mu"][i],        agg["sa_mean_sd"][i],          3) for i in agg.index],
        "PAINS (%)":       [pick(agg["pains_pct_mu"][i],        agg["pains_pct_sd"][i],        2) for i in agg.index],
        "MW":              [pick(agg["mw_mean_mu"][i],          agg["mw_mean_sd"][i],          1) for i in agg.index],
        "logP":            [pick(agg["logp_mean_mu"][i],        agg["logp_mean_sd"][i],        2) for i in agg.index],
        "TPSA":            [pick(agg["tpsa_mean_mu"][i],        agg["tpsa_mean_sd"][i],        1) for i in agg.index],
        "Heavy atoms":     [pick(agg["heavy_atoms_mean_mu"][i], agg["heavy_atoms_mean_sd"][i], 1) for i in agg.index],
        "Arom. rings":     [pick(agg["rings_aromatic_mean_mu"][i], agg["rings_aromatic_mean_sd"][i], 2) for i in agg.index],
        "Frac. sp3":       [pick(agg["frac_csp3_mean_mu"][i],   agg["frac_csp3_mean_sd"][i],   2) for i in agg.index],
    })
    order = {"LiteGov":0,"HeavyGov":1}
    tbl["__o__"]=tbl["Method"].map(lambda x:order.get(x,9))
    return tbl.sort_values(["__o__","Method"]).drop(columns="__o__").reset_index(drop=True)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="repo root (contains results/)")
    ap.add_argument("--k", type=int, default=300, help="Top-K per (seed,temp,method)")
    ap.add_argument("--higher-better", action="store_true", help="descending score")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--list-availability", action="store_true", help="only list availability and exit")
    ap.add_argument("--auto-k", action="store_true", help="per-pair K=min(uniq_lite, uniq_heavy, targetK)")
    ap.add_argument("--min-k", type=int, default=0, help="when --auto-k, require K>=min-k")
    ap.add_argument("--fallback-score", type=str, default="qed", choices=["qed","sa","random","none"],
                    help="when a CSV lacks score column: use -QED, SA, or stable random; 'none' to disallow")
    args = ap.parse_args()

    ROOT = Path(args.root)
    OUT  = ROOT / "figs"
    OUT.mkdir(parents=True, exist_ok=True)

    # recursive glob
    GLOB = str(ROOT / "results" / "**" / "*.csv")
    files_all = glob.glob(GLOB, recursive=True)

    # filter out known aggregates that do not contain molecule rows
    files = []
    for f in files_all:
        fl = str(f).lower()
        if any(fl.endswith(x) for x in ("summary.csv","aggregate_means.csv","metrics.csv")):
            continue
        if "/figs/" in fl or fl.endswith(".tex") or "paper_table" in fl:
            continue
        files.append(f)

    audit_rows = []
    kept_rows  = []

    for f in files:
        method = norm_method_from_path(f)
        audit = {"path": f, "method": method, "seed_col": None, "temp_col": None,
                 "score_col": None, "smiles_col": None, "seed_val": None, "temp_val": None,
                 "kept": False, "skip_reason": ""}
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            audit["skip_reason"] = f"read_csv_error:{type(e).__name__}"
            audit_rows.append(audit); continue
        if df.empty:
            audit["skip_reason"] = "empty_csv"; audit_rows.append(audit); continue
        if method is None:
            audit["skip_reason"] = "method_not_detected"; audit_rows.append(audit); continue

        seed_col, temp_col, score_col, smiles_col, seed_val, temp_val = detect_columns_or_path(df, f, debug=args.debug)
        audit.update({"seed_col": seed_col, "temp_col": temp_col, "score_col": score_col,
                      "smiles_col": smiles_col, "seed_val": seed_val, "temp_val": temp_val})

        if (seed_col is None and seed_val is None) or (temp_col is None and temp_val is None) or (smiles_col is None):
            audit["skip_reason"] = "missing_seed/temp/smiles"
            audit_rows.append(audit); continue
        if score_col is None and args.fallback_score == "none":
            audit["skip_reason"] = "missing_score_and_fallback_none"
            audit_rows.append(audit); continue

        keep = set([smiles_col])
        if score_col: keep.add(score_col)
        if seed_col:  keep.add(seed_col)
        if temp_col:  keep.add(temp_col)
        keep |= {
            "sa_ertl","qed","mw","logp","tpsa","heavy_atoms","rings_aromatic",
            "frac_csp3","murcko_scaffold","pains_n","pains_ok","smiles_can"
        }
        keep = [c for c in df.columns if c in keep]
        df = df[keep].copy()

        if score_col:
            df.rename(columns={score_col:"score"}, inplace=True)
        df.rename(columns={smiles_col:"smiles"}, inplace=True)

        if "smiles_can" not in df.columns and "canonical_smiles" in df.columns:
            df.rename(columns={"canonical_smiles":"smiles_can"}, inplace=True)
        if seed_col: df.rename(columns={seed_col:"seed"}, inplace=True)
        else: df["seed"] = seed_val
        if temp_col: df.rename(columns={temp_col:"temp"}, inplace=True)
        else: df["temp"] = temp_val

        df = df.dropna(subset=["seed","temp","smiles"])
        if df.empty:
            audit["skip_reason"] = "rows_dropped_after_na_filter"
            audit_rows.append(audit); continue

        df["Method"] = method
        # unify smiles key for uniq counts
        if "smiles_can" in df.columns and df["smiles_can"].notna().any():
            df["key_smiles"] = df["smiles_can"].astype(str)
        else:
            df["key_smiles"] = df["smiles"].astype(str)

        kept_rows.append(df)
        audit["kept"] = True
        if score_col is None:
            audit["score_col"] = f"[fallback:{args.fallback_score}]"
        audit_rows.append(audit)

    # write audit file (always)
    audit_df = pd.DataFrame(audit_rows)
    audit_csv = OUT / "topk_audit_files.csv"
    audit_df.to_csv(audit_csv, index=False)

    if not kept_rows:
        print("[!] No usable CSVs after audit. See:", audit_csv)
        return

    all_df = pd.concat(kept_rows, ignore_index=True)

    # ensure descriptors if needed (for fallback qed/sa)
    try:
        all_df = ensure_descriptors(all_df)
    except Exception as e:
        if args.fallback_score in ("qed","sa"):
            print("[!] Need descriptors for fallback-score:", args.fallback_score, "Error:", e)
            print("    See audit:", audit_csv)
            return
        pass

    # synthesize score if missing
    if "score" not in all_df.columns or all_df["score"].isna().all():
        fs = args.fallback_score
        if fs == "qed":
            if "qed" not in all_df.columns:
                print("[!] Fallback qed requested but 'qed' column missing.")
                return
            all_df["score"] = -all_df["qed"]  # higher QED better -> use negative so lower-better
        elif fs == "sa":
            if "sa_ertl" not in all_df.columns:
                print("[!] Fallback sa requested but 'sa_ertl' column missing.")
                return
            all_df["score"] = all_df["sa_ertl"]  # lower SA better
        elif fs == "random":
            # stable pseudo-random by SMILES hash
            def hash_to_float(s):
                h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
                return int(h[:8], 16) / 0xFFFFFFFF
            all_df["score"] = all_df["key_smiles"].map(hash_to_float)
        else:  # none
            print("[!] No score column and fallback-score=none; cannot proceed.")
            return
    else:
        # fill only missing scores using fallback
        if all_df["score"].isna().any() and args.fallback_score != "none":
            m = all_df["score"].isna()
            fs = args.fallback_score
            if fs == "qed" and "qed" in all_df.columns:
                all_df.loc[m, "score"] = -all_df.loc[m, "qed"]
            elif fs == "sa" and "sa_ertl" in all_df.columns:
                all_df.loc[m, "score"] = all_df.loc[m, "sa_ertl"]
            elif fs == "random":
                def hash_to_float(s):
                    h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
                    return int(h[:8], 16) / 0xFFFFFFFF
                all_df.loc[m, "score"] = all_df.loc[m, "key_smiles"].map(hash_to_float)

    # availability (no groupby.apply warning)
    avail_df = (all_df
        .groupby(["Method","seed","temp"])["key_smiles"]
        .nunique()
        .reset_index(name="uniq_n"))

    piv = avail_df.pivot_table(index=["seed","temp"], columns="Method", values="uniq_n", fill_value=0).reset_index()
    if "LiteGov" not in piv.columns:  piv["LiteGov"]  = 0
    if "HeavyGov" not in piv.columns: piv["HeavyGov"] = 0
    piv["max_common_k"] = np.minimum(piv["LiteGov"], piv["HeavyGov"]).astype(int)
    piv = piv.sort_values("max_common_k", ascending=False).reset_index(drop=True)

    avail_csv = OUT / "topk_availability.csv"
    piv.to_csv(avail_csv, index=False)

    print("[+] Availability by (seed,temp): top rows")
    print(piv.head(50).to_string(index=False))
    print(f"[+] wrote {avail_csv}")
    print(f"[+] wrote {audit_csv}")

    if args.list_availability:
        return

    # choose pairs and per-pair K
    target_k = int(args.k)
    if args.auto_k:
        pairs = piv[piv["max_common_k"] >= max(1, int(args.min_k))][["seed","temp","max_common_k"]].copy()
        if pairs.empty:
            print("[!] No (seed,temp) pairs meet --min-k under --auto-k.")
            print("    Inspect:", audit_csv)
            return
        pairs["pair_k"] = pairs["max_common_k"].clip(upper=target_k).astype(int)
    else:
        pairs = piv[piv["max_common_k"] >= target_k][["seed","temp"]].copy()
        if pairs.empty:
            print(f"[!] Intersection empty for K={target_k}. Inspect:", audit_csv)
            return
        pairs["pair_k"] = target_k

    # replicate stats
    reps=[]
    lower_better = not args.higher_better
    for _, r in pairs.iterrows():
        seed, temp, kk = r["seed"], r["temp"], int(r["pair_k"])
        for method in ["LiteGov","HeavyGov"]:
            sub = all_df[(all_df["Method"]==method) & (all_df["seed"]==seed) & (all_df["temp"]==temp)].copy()
            sub = sub.sort_values("score", ascending=lower_better).drop_duplicates(subset=["key_smiles"], keep="first").head(kk).copy()
            if len(sub) < kk:
                continue
            stats = replicate_stats(sub)
            stats.update({"Method":method,"seed":seed,"temp":float(temp),"K":kk})
            reps.append(stats)

    if not reps:
        print("[!] No replicates after Top-K selection. Inspect:", audit_csv)
        return

    reps_df = pd.DataFrame(reps)
    agg = agg_mean_sd(reps_df, by="Method")
    paper = build_paper_table(agg)

    tag = f"topk{target_k}" if not args.auto_k else f"autok{target_k}_min{int(args.min_k)}"
    csv_path = OUT / f"paper_table_{tag}.csv"
    tex_path = OUT / f"paper_table_{tag}.tex"
    paper.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        cols = list(paper.columns)
        colfmt = "l" + "r" * (len(cols)-1)
        f.write("\\begin{tabular}{%s}\n\\toprule\n" % colfmt)
        f.write(" & ".join(c.replace("%","\\%") for c in cols) + " \\\\\n\\midrule\n")
        for _, rw in paper.iterrows():
            vals = [str(rw[c]).replace("%","\\%") for c in cols]
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    def line(mu, sd, k): return f"{mu:.{k}f} +/- {sd:.{k}f}"
    for _, rr in agg.iterrows():
        print(f"\n{rr['Method']}  (replicates={int(rr['Replicates'])})")
        print("  Valid (%):     ", line(rr["valid_pct_mu"], rr["valid_pct_sd"], 1))
        print("  Unique (%):    ", line(rr["unique_pct_mu"], rr["unique_pct_sd"], 1))
        print("  Scaffold (%):  ", line(rr["scaf_div_pct_mu"], rr["scaf_div_pct_sd"], 1))
        print("  QED:           ", line(rr["qed_mean_mu"], rr["qed_mean_sd"], 3))
        print("  SA:            ", line(rr["sa_mean_mu"], rr["sa_mean_sd"], 3))
        print("  PAINS (%):     ", line(rr["pains_pct_mu"], rr["pains_pct_sd"], 2))
        print("  MW:            ", line(rr["mw_mean_mu"], rr["mw_mean_sd"], 1))
        print("  logP:          ", line(rr["logp_mean_mu"], rr["logp_mean_sd"], 2))
        print("  TPSA:          ", line(rr["tpsa_mean_mu"], rr["tpsa_mean_sd"], 1))
        print("  Heavy atoms:   ", line(rr["heavy_atoms_mean_mu"], rr["heavy_atoms_mean_sd"], 1))
        print("  Arom. rings:   ", line(rr["rings_aromatic_mean_mu"], rr["rings_aromatic_mean_sd"], 2))
        print("  Frac. sp3:     ", line(rr["frac_csp3_mean_mu"], rr["frac_csp3_mean_sd"], 2))

    print(f"\n[+] wrote {csv_path}")
    print(f"[+] wrote {tex_path}")

if __name__ == "__main__":
    main()
