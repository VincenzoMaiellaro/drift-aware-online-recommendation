"""Generate MIND-specific behavior-shift plots.

Unlike the synthetic plots, these figures do not assume labelled drift points.
They describe history-vs-recent divergence activations, category shifts, and
the effect of temporary reranking on real clicked-impression streams.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    """Parse arguments and generate MIND behavior-shift figures."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiment-results/mind/latest")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plot_paths = generate_mind_plots(output_dir)
    print(f"Wrote {len(plot_paths)} MIND plots to {output_dir / 'mind_plots'}")


def generate_mind_plots(output_dir: Path) -> list[Path]:
    """Generate all MIND-specific plots from one result directory."""

    if (output_dir / "final_summary_overall.csv").exists():
        return generate_mind_protocol_plots(output_dir)

    detections = pd.read_csv(output_dir / "detections.csv")
    events = pd.read_csv(
        output_dir / "events.csv",
        dtype={"rerank_reason": "string", "cluster_rule_detector": "string"},
    )
    behavior_shifts_path = output_dir / "behavior_shifts.csv"
    if behavior_shifts_path.exists():
        behavior_shifts = pd.read_csv(behavior_shifts_path)
    else:
        behavior_shifts = pd.DataFrame()

    style = (
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.style.use(style)
    plot_dir = output_dir / "mind_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    return [
        plot_divergence_scores(detections, plot_dir),
        plot_shift_counts(behavior_shifts, plot_dir),
        plot_category_shift_pairs(behavior_shifts, plot_dir),
        plot_active_event_method_comparison(events, plot_dir),
    ]


def generate_mind_protocol_plots(output_dir: Path) -> list[Path]:
    """Generate MIND plots from detector-protocol aggregate outputs."""

    summary = pd.read_csv(output_dir / "final_summary_overall.csv")
    detection = pd.read_csv(output_dir / "final_detection_overall.csv")
    tuning_path = output_dir / "tuning_summary_overall.csv"
    tuning = pd.read_csv(tuning_path) if tuning_path.exists() else pd.DataFrame()
    behavior_shifts = read_protocol_behavior_shifts(output_dir)

    style = (
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.style.use(style)
    plot_dir = output_dir / "mind_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    return [
        plot_protocol_quality(summary, plot_dir),
        plot_protocol_activation_counts(detection, plot_dir),
        plot_category_shift_pairs(behavior_shifts, plot_dir),
        plot_protocol_threshold_sensitivity(tuning, plot_dir),
    ]


def read_protocol_behavior_shifts(output_dir: Path) -> pd.DataFrame:
    """Read final-test behavior-shift files across protocol seeds."""

    parts: list[pd.DataFrame] = []
    for path in sorted((output_dir / "final").glob("seed_*/behavior_shifts.csv")):
        frame = pd.read_csv(path)
        frame["seed"] = path.parent.name.replace("seed_", "")
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def plot_protocol_quality(summary: pd.DataFrame, plot_dir: Path) -> Path:
    """Plot final-test MIND ranking metrics by method."""

    path = plot_dir / "mind_protocol_quality.png"
    metrics = {
        "ctr_at_k_mean": "CTR@10",
        "ndcg_at_k_mean": "nDCG@10",
        "category_top1_accuracy_mean": "Category top-1",
        "rerank_active_rate_mean": "Rerank active",
    }
    methods = [
        "base_mf",
        "time_decay_mf",
        "recent_popularity",
        "mf_rerank_always",
        "drift_aware_kl",
        "drift_aware_jsd",
        "drift_aware_kl_clustered",
        "drift_aware_jsd_clustered",
    ]
    data = summary[summary["method"].isin(methods)].copy()
    data = data.set_index("method")[[col for col in metrics if col in data.columns]]
    data = data.rename(columns=metrics)
    fig, ax = plt.subplots(figsize=(13, 7))
    data.plot(kind="bar", ax=ax)
    ax.set_title("MIND Final-Test Quality by Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean metric value")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_protocol_activation_counts(detection: pd.DataFrame, plot_dir: Path) -> Path:
    """Plot final-test behavior-shift activation counts."""

    path = plot_dir / "mind_protocol_activation_counts.png"
    data = detection.dropna(subset=["n_detections_mean"]).copy()
    data["label"] = data["method"] + " / " + data["detector"]
    data = data.sort_values("n_detections_mean")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(data["label"], data["n_detections_mean"], color="#5aa17f")
    ax.set_title("MIND Behavior-Shift Activations")
    ax.set_xlabel("Mean detections over final seeds")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_protocol_threshold_sensitivity(
    tuning: pd.DataFrame, plot_dir: Path
) -> Path:
    """Plot tuning quality across detector thresholds when available."""

    path = plot_dir / "mind_protocol_threshold_sensitivity.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    threshold_cols = [col for col in ("kl_threshold", "jsd_threshold") if col in tuning]
    if tuning.empty or not threshold_cols:
        ax.text(0.5, 0.5, "No tuning table available", ha="center", va="center")
    else:
        threshold_col = threshold_cols[0]
        candidate = tuning[
            tuning["method"].str.contains("drift_aware", na=False)
            & tuning["method"].str.contains(
                "kl" if threshold_col == "kl_threshold" else "jsd",
                na=False,
            )
        ].copy()
        for method, group in candidate.groupby("method"):
            group = group.sort_values(threshold_col)
            ax.plot(
                group[threshold_col],
                group["category_top1_accuracy_mean"],
                marker="o",
                linewidth=2,
                label=method,
            )
        ax.legend(loc="best")
        ax.set_xlabel(threshold_col.replace("_", " "))
        ax.set_ylabel("Category top-1 accuracy")
    ax.set_title("MIND Tuning Threshold Sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_divergence_scores(detections: pd.DataFrame, plot_dir: Path) -> Path:
    """Plot the distribution of history-vs-recent divergence scores."""

    path = plot_dir / "divergence_score_distribution.png"
    scores = detections[
        detections["detector"].isin(["history_kl", "history_jsd"])
        & detections["score"].notna()
    ][["detector", "score"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    if scores.empty:
        ax.text(0.5, 0.5, "No finite divergence scores", ha="center", va="center")
    else:
        colors = {"history_kl": "#356c9b", "history_jsd": "#b279a2"}
        for detector, group in scores.groupby("detector"):
            ax.hist(
                group["score"],
                bins=40,
                alpha=0.65,
                label=detector,
                color=colors.get(detector, "#777777"),
            )
            ax.axvline(
                group["score"].median(),
                color=colors.get(detector, "#777777"),
                linestyle="--",
            )
        ax.legend(loc="best")
    ax.set_title("MIND History-vs-Recent Divergence Scores")
    ax.set_xlabel("Divergence score")
    ax.set_ylabel("Detector updates")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_shift_counts(behavior_shifts: pd.DataFrame, plot_dir: Path) -> Path:
    """Plot detected behavior-shift counts over stream time."""

    path = plot_dir / "behavior_shift_counts_over_time.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    if behavior_shifts.empty:
        ax.text(0.5, 0.5, "No behavior shifts detected", ha="center", va="center")
    else:
        shifts = behavior_shifts.copy()
        bin_size = max(1, int(shifts["event_id"].max() // 40))
        shifts["event_bin"] = (shifts["event_id"] // bin_size) * bin_size
        counts = shifts.groupby(["method", "event_bin"]).size().reset_index(name="n")
        for method, group in counts.groupby("method"):
            ax.plot(group["event_bin"], group["n"], linewidth=2, label=method)
        ax.legend(loc="best")
    ax.set_title("Detected MIND Behavior Shifts Over Time")
    ax.set_xlabel("Stream event")
    ax.set_ylabel("Detected shifts per bin")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_category_shift_pairs(behavior_shifts: pd.DataFrame, plot_dir: Path) -> Path:
    """Plot the most frequent baseline-to-recent category shift pairs."""

    path = plot_dir / "category_shift_pairs.png"
    fig, ax = plt.subplots(figsize=(11, 7))
    required = {"baseline_top_category", "recent_top_category"}
    if behavior_shifts.empty or not required.issubset(behavior_shifts.columns):
        ax.text(0.5, 0.5, "No category shift pairs", ha="center", va="center")
    else:
        shifts = behavior_shifts.copy()
        baseline_col = (
            "baseline_top_category_name"
            if "baseline_top_category_name" in shifts.columns
            else "baseline_top_category"
        )
        recent_col = (
            "recent_top_category_name"
            if "recent_top_category_name" in shifts.columns
            else "recent_top_category"
        )
        shifts["pair"] = (
            shifts[baseline_col].astype(str)
            + " -> "
            + shifts[recent_col].astype(str)
        )
        top_pairs = shifts["pair"].value_counts().head(12).sort_values()
        ax.barh(top_pairs.index, top_pairs.values, color="#5aa17f")
    ax.set_title("Most Frequent MIND Category Shifts")
    ax.set_xlabel("Detected shifts")
    ax.set_ylabel("History top label -> recent top label")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_active_event_method_comparison(events: pd.DataFrame, plot_dir: Path) -> Path:
    """Compare methods on events where adaptive divergence reranking is active."""

    path = plot_dir / "rerank_active_effect.png"
    fig, ax = plt.subplots(figsize=(11, 6))
    required = {
        "method",
        "event_id",
        "rerank_active",
        "hit",
        "ndcg",
        "category_hit",
        "category_top1_hit",
    }
    if not required.issubset(events.columns):
        ax.text(0.5, 0.5, "Missing matched-event metrics", ha="center", va="center")
    else:
        active_ids = events.loc[
            events["method"].isin(["drift_aware_kl", "drift_aware_jsd"])
            & (events["rerank_active"].astype(int) == 1),
            "event_id",
        ].drop_duplicates()
        matched = events[
            events["event_id"].isin(active_ids)
            & events["method"].isin(["base_mf", "drift_aware_kl", "drift_aware_jsd"])
        ].copy()
        if active_ids.empty or matched["method"].nunique() < 2:
            ax.text(
                0.5,
                0.5,
                "No matched active KL events",
                ha="center",
                va="center",
            )
        else:
            metrics = {
                "hit": "CTR@10",
                "ndcg": "nDCG@10",
                "category_hit": "Category hit@10",
                "category_top1_hit": "Category top-1",
            }
            grouped = matched.groupby("method")[list(metrics)].mean()
            methods = [
                method
                for method in ["base_mf", "drift_aware_kl", "drift_aware_jsd"]
                if method in grouped.index
            ]
            x = np.arange(len(metrics))
            width = 0.72 / max(1, len(methods))
            colors = {
                "base_mf": "#6a8caf",
                "drift_aware_kl": "#d98c5f",
                "drift_aware_jsd": "#b279a2",
            }
            for offset, method in enumerate(methods):
                values = grouped.loc[method, list(metrics)]
                ax.bar(
                    x + (offset - (len(methods) - 1) / 2) * width,
                    values,
                    width,
                    label=method,
                    color=colors[method],
                )
            for idx, metric in enumerate(metrics):
                base_value = float(grouped.loc["base_mf", metric])
                adaptive_value = float(grouped.loc["drift_aware_kl", metric])
                delta = adaptive_value - base_value
                ax.text(
                    idx,
                    max(base_value, adaptive_value) + 0.015,
                    f"{delta:+.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            max_value = float(grouped[list(metrics)].to_numpy().max())
            ax.set_ylim(0, min(1.0, max_value + 0.1))
            ax.set_xticks(x)
            ax.set_xticklabels(metrics.values())
            ax.legend(loc="best")
            ax.text(
                0.01,
                0.98,
                f"Subset: {len(active_ids)} events where KL or JSD reranking is active",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
    ax.set_title("MIND Same Active Events: Base MF vs Drift-Aware Divergence")
    ax.set_ylabel("Mean metric value on matched events")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
