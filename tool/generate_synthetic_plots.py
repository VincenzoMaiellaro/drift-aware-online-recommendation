"""Generate plots from experiment result tables.

The script reads CSV outputs produced by the single prequential experiment
runner and plots performance over time, drift detection, recovery, and regret.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    """Parse plotting arguments and generate plot files."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Experiment output directory, e.g. experiment-results/synthetic/latest.",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Path to events.csv. If used, output-dir is inferred from its parent.",
    )
    args = parser.parse_args()

    output_dir = infer_output_dir(args.output_dir, args.results)
    plot_paths = generate_plots(output_dir)
    print(f"Wrote {len(plot_paths)} plots to {output_dir / 'plots'}")


def infer_output_dir(output_dir: str | None, results: str | None) -> Path:
    """Resolve the result directory from explicit arguments or defaults."""

    if output_dir:
        return Path(output_dir)
    if results:
        return Path(results).parent
    return Path("experiment-results/synthetic/latest")


def generate_plots(output_dir: Path) -> list[Path]:
    """Load saved experiment tables and generate the standard plots."""

    if (output_dir / "final_summary_overall.csv").exists():
        return generate_protocol_plots(output_dir)

    events = pd.read_csv(
        output_dir / "events.csv",
        dtype={"rerank_reason": "string", "cluster_rule_detector": "string"},
    )
    summary = pd.read_csv(output_dir / "summary.csv")
    detections = pd.read_csv(output_dir / "detections.csv")
    detection_summary = pd.read_csv(output_dir / "detection_summary.csv")
    recovery = pd.read_csv(output_dir / "recovery.csv")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    style = (
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.style.use(style)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    drift_points = [
        int(v) for v in metadata["stream"]["user_drift_points"].values()
    ]
    global_drift = int(pd.Series(drift_points).median())
    plot_paths = [
        plot_performance(events, plot_dir, global_drift),
        plot_detection(detections, detection_summary, plot_dir, drift_points),
        plot_recovery_regret(summary, recovery, plot_dir),
    ]
    return plot_paths


def generate_protocol_plots(output_dir: Path) -> list[Path]:
    """Generate plots for aggregated final-test protocol outputs."""

    summary = pd.read_csv(output_dir / "final_summary_overall.csv")
    by_scenario = pd.read_csv(output_dir / "final_summary_by_scenario.csv")
    detection = pd.read_csv(output_dir / "final_detection_overall.csv")

    style = (
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.style.use(style)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_protocol_quality_regret(summary, plot_dir),
        plot_protocol_scenario_top1(by_scenario, plot_dir),
        plot_protocol_detection(detection, plot_dir),
    ]


def plot_protocol_quality_regret(summary: pd.DataFrame, plot_dir: Path) -> Path:
    """Plot final-test category quality against cumulative regret."""

    path = plot_dir / "protocol_quality_regret.png"
    methods = {
        "base_mf",
        "time_decay_mf",
        "recent_popularity",
        "mf_rerank_always",
        "mf_rerank_oracle",
        "drift_aware_kl",
        "drift_aware_jsd",
        "drift_aware_kl_clustered",
        "drift_aware_jsd_clustered",
    }
    data = summary[summary["method"].isin(methods)].copy()
    data = data.dropna(
        subset=[
            "category_top1_accuracy_mean",
            "cumulative_regret_mean",
            "rerank_active_rate_mean",
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    sizes = 120 + 900 * data["rerank_active_rate_mean"].clip(lower=0.0)
    ax.scatter(
        data["category_top1_accuracy_mean"],
        data["cumulative_regret_mean"],
        s=sizes,
        alpha=0.75,
        color="#4f81bd",
        edgecolor="white",
        linewidth=1.0,
    )
    for _, row in data.iterrows():
        ax.annotate(
            row["method"],
            (row["category_top1_accuracy_mean"], row["cumulative_regret_mean"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("Final-Test Quality and Oracle Regret")
    ax.set_xlabel("Category top-1 accuracy")
    ax.set_ylabel("Cumulative regret (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_protocol_scenario_top1(by_scenario: pd.DataFrame, plot_dir: Path) -> Path:
    """Plot category top-1 accuracy by final-test scenario."""

    path = plot_dir / "protocol_scenario_top1.png"
    methods = [
        "base_mf",
        "mf_rerank_always",
        "drift_aware_kl",
        "drift_aware_jsd",
        "drift_aware_kl_clustered",
        "drift_aware_jsd_clustered",
    ]
    data = by_scenario[by_scenario["method"].isin(methods)].copy()
    pivot = data.pivot_table(
        index="scenario",
        columns="method",
        values="category_top1_accuracy_mean",
    )
    pivot = pivot[[method for method in methods if method in pivot.columns]]
    fig, ax = plt.subplots(figsize=(13, 7))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Category Top-1 Accuracy by Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Category top-1 accuracy")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_protocol_detection(detection: pd.DataFrame, plot_dir: Path) -> Path:
    """Plot final-test detector coverage and false positives."""

    path = plot_dir / "protocol_detection_summary.png"
    data = detection.dropna(subset=["detection_rate_mean"]).copy()
    data["label"] = data["method"] + " / " + data["detector"]
    data = data.sort_values("detection_rate_mean")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    axes[0].barh(data["label"], data["detection_rate_mean"], color="#5aa17f")
    axes[0].set_title("Detection Rate")
    axes[0].set_xlabel("Rate")
    axes[1].barh(data["label"], data["false_positives_mean"], color="#d98c5f")
    axes[1].set_title("False Positives")
    axes[1].set_xlabel("Mean count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_performance(events: pd.DataFrame, plot_dir: Path, global_drift: int) -> Path:
    """Plot moving recommendation quality and mark the median true drift point."""

    path = plot_dir / "performance_over_time.png"
    selected = events[events["method"].map(_include_in_performance_plot)]
    fig, ax = plt.subplots(figsize=(14, 7))
    for method, group in selected.groupby("method"):
        group = group.sort_values("event_id")
        ax.plot(
            group["event_id"],
            group["category_top1_hit_ma"],
            linewidth=2,
            label=method,
        )
    ax.axvline(global_drift, color="black", linestyle="--", label="median true drift")
    ax.set_title("Recommendation Performance Over Time")
    ax.set_xlabel("Stream event")
    ax.set_ylabel("Moving category top-1 accuracy")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _include_in_performance_plot(method: str) -> bool:
    """Keep the main baselines and ablations visible in the time-series plot."""

    return (
        method
        in {
            "base_mf",
            "recent_popularity",
            "global_popularity",
            "mf_rerank_always",
            "mf_rerank_oracle",
            "drift_aware_kl",
            "drift_aware_jsd",
        }
        or method.startswith("time_decay_mf")
    )


def plot_detection(
    detections: pd.DataFrame,
    detection_summary: pd.DataFrame,
    plot_dir: Path,
    drift_points: list[int],
) -> Path:
    """Plot true drift-point distribution and actual detector firings."""

    path = plot_dir / "drift_detection_behavior.png"
    detected = detections[detections["drift_detected"] == 1].copy()
    detected["method_detector"] = detected["method"] + " / " + detected["detector"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    axes[0].hist(drift_points, bins=20, color="#6a8caf", alpha=0.8)
    axes[0].set_title("True User Drift Points")
    axes[0].set_xlabel("Event")
    axes[0].set_ylabel("Users")

    if detected.empty:
        axes[1].text(0.5, 0.5, "No detections", ha="center", va="center")
    else:
        labels = list(detected["method_detector"].drop_duplicates())
        label_to_y = {label: idx for idx, label in enumerate(labels)}
        colors = {"kl": "#356c9b", "jsd": "#c46a4a"}
        for detector, group in detected.groupby("detector"):
            axes[1].scatter(
                group["timestamp"],
                [label_to_y[label] for label in group["method_detector"]],
                label=detector,
                s=45,
                alpha=0.85,
                color=colors.get(detector, None),
            )
        axes[1].set_yticks(range(len(labels)))
        axes[1].set_yticklabels(labels)
        axes[1].legend(loc="best")
    axes[1].set_title("Detected Drift Events")
    axes[1].set_xlabel("Event")
    axes[1].set_ylabel("")

    if not detection_summary.empty:
        summary_text = detection_summary[
            ["method", "detection_rate", "mean_delay", "false_positives"]
        ].round(2).to_string(index=False)
        axes[1].text(
            1.01,
            0.5,
            summary_text,
            transform=axes[1].transAxes,
            va="center",
            fontsize=9,
            family="monospace",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_recovery_regret(
    summary: pd.DataFrame, recovery: pd.DataFrame, plot_dir: Path
) -> Path:
    """Plot cumulative regret and estimated recovery time by method."""

    path = plot_dir / "recovery_regret_comparison.png"
    merged = summary.merge(recovery, on="method", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    regret_sorted = merged.sort_values("cumulative_regret")
    axes[0].barh(
        regret_sorted["method"],
        regret_sorted["cumulative_regret"],
        color="#d98c5f",
    )
    axes[0].set_title("Cumulative Regret")
    axes[0].set_xlabel("Lower is better")
    axes[0].set_ylabel("")
    recovery_sorted = merged.sort_values("recovery_time_events")
    axes[1].barh(
        recovery_sorted["method"],
        recovery_sorted["recovery_time_events"],
        color="#5aa17f",
    )
    axes[1].set_title("Recovery Time")
    axes[1].set_xlabel("Events after aligned user drift")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
