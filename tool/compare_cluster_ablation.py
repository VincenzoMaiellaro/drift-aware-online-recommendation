"""Build a compact comparison table for the clustering activation ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHOD_ORDER = [
    "drift_aware_kl",
    "drift_aware_kl_clustered",
    "drift_aware_kl_clustered_no_force",
    "drift_aware_jsd",
    "drift_aware_jsd_clustered",
    "drift_aware_jsd_clustered_no_force",
    "mf_rerank_always",
]

OUTPUT_COLUMNS = {
    "method": "method",
    "ctr_at_k": "HitRate@10",
    "ndcg_at_k": "NDCG@10",
    "category_top1_accuracy": "category_top1_accuracy",
    "cumulative_regret": "cumulative_regret",
    "rerank_active_rate": "rerank_rate",
    "cluster_rule_active_rate": "cluster_rule_rate",
}


def main() -> None:
    """Parse arguments and write the ablation comparison table."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="experiment-results/synthetic/cluster_no_force_ablation",
    )
    parser.add_argument("--csv", help="Optional CSV output path.")
    args = parser.parse_args()

    table = build_table(Path(args.output_dir))
    print(dataframe_to_text_table(table))
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)


def build_table(output_dir: Path) -> pd.DataFrame:
    """Read summary.csv and keep the metrics needed for the ablation."""

    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.csv found in {output_dir}")

    summary = pd.read_csv(summary_path)
    available_methods = [method for method in METHOD_ORDER if method in set(summary["method"])]
    columns = [column for column in OUTPUT_COLUMNS if column in summary.columns]
    table = summary[summary["method"].isin(available_methods)][columns].copy()
    table["method"] = pd.Categorical(table["method"], available_methods, ordered=True)
    table = table.sort_values("method").reset_index(drop=True)
    return table.rename(columns=OUTPUT_COLUMNS)


def dataframe_to_text_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a dependency-free text table."""

    if df.empty:
        return "_No rows._"
    rounded = df.copy()
    for column in rounded.columns:
        if pd.api.types.is_float_dtype(rounded[column]):
            rounded[column] = rounded[column].map(lambda value: f"{value:.4f}")
    rows = [[str(value) for value in row] for row in rounded.to_numpy()]
    table = [
        "| " + " | ".join(str(column) for column in rounded.columns) + " |",
        "| " + " | ".join(["---"] * len(rounded.columns)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


if __name__ == "__main__":
    main()
