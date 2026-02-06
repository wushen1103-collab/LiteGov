#!/usr/bin/env python

import pandas as pd
from pathlib import Path


def build_table1() -> pd.DataFrame:
    """
    Build Table 1: overview of datasets, generators and governance configs.
    Edit descriptions (e.g. training set name and size) before final use.
    """

    rows = []

    # Datasets
    rows.append(
        dict(
            category="Dataset",
            component="GenMol training set",
            code="train_genmol",
            description="Dataset used to train the GenMol generator (e.g., ChEMBL subset, N molecules).",
        )
    )
    rows.append(
        dict(
            category="Dataset",
            component="MoLeR training set",
            code="train_moler",
            description="Dataset used to train the MoLeR generator.",
        )
    )
    rows.append(
        dict(
            category="Dataset",
            component="Docking targets",
            code="1M17, 6LU7, CDK2, ERa, HIVPR",
            description="Protein targets used for docking-based evaluation.",
        )
    )

    # Generators
    rows.append(
        dict(
            category="Generator",
            component="GenMol",
            code="genmol",
            description="Base molecular generator; main model used in this work.",
        )
    )
    rows.append(
        dict(
            category="Generator",
            component="MoLeR",
            code="moler",
            description="Alternative generator used to test transfer of governance effects.",
        )
    )

    # Governance configs
    rows.append(
        dict(
            category="Governance",
            component="Raw",
            code="raw",
            description="No governance; direct samples from the generator.",
        )
    )
    rows.append(
        dict(
            category="Governance",
            component="QED filter",
            code="qed",
            description="Post-hoc filtering/reranking based only on QED score.",
        )
    )
    rows.append(
        dict(
            category="Governance",
            component="RuleKit",
            code="rulekit",
            description="Hand-crafted medicinal chemistry rules / structural alerts.",
        )
    )
    rows.append(
        dict(
            category="Governance",
            component="Lite governance",
            code="lite",
            description="LiteGov: lightweight learned governance filter; main method.",
        )
    )
    rows.append(
        dict(
            category="Governance",
            component="Heavy governance",
            code="heavy",
            description="HeavyGov: full governance stack with stronger but more expensive checks.",
        )
    )

    df = pd.DataFrame(rows)
    order = dict(Dataset=0, Generator=1, Governance=2)
    df["category_order"] = df["category"].map(order)
    df = df.sort_values(["category_order", "category", "component"]).drop(
        columns=["category_order"]
    )
    return df


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "tables"
    out_dir.mkdir(exist_ok=True)

    df = build_table1()

    csv_path = out_dir / "table1_setup_overview.csv"
    df.to_csv(csv_path, index=False)

    print(f"[INFO] Wrote CSV to {csv_path}")
    print("[INFO] Preview:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
