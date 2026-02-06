#!/usr/bin/env python
import os
import numpy as np
import pandas as pd


MODES = {
    "pass": "results/denovo/pass/summary.csv",
    "filter": "results/denovo/filter/summary.csv",
    "filter+rank": "results/denovo/filter_rank/summary.csv",
    "govern": "results/denovo/govern/summary.csv",
}

# Metrics we try to aggregate if present in each summary.csv
METRICS = [
    "validity",
    "uniqueness",
    "scaf_div",
    "qed_mean",
    "sa_mean",
    "pains_rate",
    "seconds",
]


def to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Convert column to numeric; return empty series if missing."""
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def main() -> None:
    rows = []
    summaries: dict[str, pd.DataFrame] = {}

    # Aggregate per-mode statistics from each summary.csv
    for mode, path in MODES.items():
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        summaries[mode] = df

        row: dict[str, float | str] = {"mode": mode}

        for m in METRICS:
            if m not in df.columns:
                continue
            s = to_numeric_series(df, m)
            row[f"{m}_mean"] = float(s.mean())
            row[f"{m}_std"] = float(s.std())

        rows.append(row)

    if not rows:
        raise RuntimeError("No summaries found for any mode.")

    out = pd.DataFrame(rows)

    # Helper to fetch mean seconds per mode
    def seconds_mean(mode: str) -> float:
        if mode not in summaries:
            return float("nan")
        df = summaries[mode]
        if "seconds" not in df.columns:
            return float("nan")
        s = to_numeric_series(df, "seconds")
        return float(s.mean())

    # End-to-end pipeline times:
    # - Raw / pass:          pass
    # - Filter-only:         pass + filter
    # - LiteGov (filter+rank): pass + filter + filter+rank
    # - HeavyGov (govern):   pass + filter + govern
    pass_e2e = seconds_mean("pass")
    filter_e2e = pass_e2e + seconds_mean("filter")
    lite_e2e = filter_e2e + seconds_mean("filter+rank")
    heavy_e2e = filter_e2e + seconds_mean("govern")

    # Ensure the column exists
    if "e2e_seconds_mean" not in out.columns:
        out["e2e_seconds_mean"] = np.nan

    out.loc[out["mode"] == "pass", "e2e_seconds_mean"] = pass_e2e
    out.loc[out["mode"] == "filter", "e2e_seconds_mean"] = filter_e2e
    out.loc[out["mode"] == "filter+rank", "e2e_seconds_mean"] = lite_e2e
    out.loc[out["mode"] == "govern", "e2e_seconds_mean"] = heavy_e2e

    os.makedirs("figs", exist_ok=True)
    out_path = os.path.join("figs", "paper_metrics_agg.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved aggregated metrics to: {out_path}")


if __name__ == "__main__":
    main()
