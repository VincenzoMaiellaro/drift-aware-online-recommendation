"""Prequential metrics and experiment summarisation utilities.

Metrics from the online predict-evaluate-update loop include
CTR@K, NDCG@K, category-level adaptation quality, oracle regret, detector delay,
and recovery time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MOVING_METRICS = [
    "hit",
    "ndcg",
    "category_hit",
    "category_top1_hit",
    "category_regret",
    "behavior_category_regret",
    "cluster_rule_active",
]

EVENT_AGGREGATIONS = {
    "hit": "mean",
    "ndcg": "mean",
    "category_hit": "mean",
    "category_top1_hit": "mean",
    "category_regret": ["mean", "sum"],
    "behavior_category_regret": ["mean", "sum"],
    "rerank_active": "mean",
    "cluster_rule_active": "mean",
    "event_id": "count",
}

SUMMARY_COLUMN_NAMES = {
    "hit": "ctr_at_k",
    "hit_mean": "ctr_at_k",
    "ndcg": "ndcg_at_k",
    "ndcg_mean": "ndcg_at_k",
    "category_hit": "category_hit_at_k",
    "category_hit_mean": "category_hit_at_k",
    "category_top1_hit": "category_top1_accuracy",
    "category_top1_hit_mean": "category_top1_accuracy",
    "category_regret_mean": "mean_regret",
    "category_regret_sum": "cumulative_regret",
    "behavior_category_regret_mean": "mean_behavior_regret",
    "behavior_category_regret_sum": "cumulative_behavior_regret",
    "rerank_active": "rerank_active_rate",
    "rerank_active_mean": "rerank_active_rate",
    "cluster_rule_active": "cluster_rule_active_rate",
    "cluster_rule_active_mean": "cluster_rule_active_rate",
    "event_id": "n_events",
    "event_id_count": "n_events",
}

DETECTION_COLUMNS = [
    "method",
    "detector",
    "detection_rate",
    "mean_delay",
    "median_delay",
    "mean_user_interaction_delay",
    "median_user_interaction_delay",
    "pre_drift_false_positives",
    "late_detections",
    "repeat_detections",
    "false_positives",
    "n_detections",
]


@dataclass
class _DetectionOutcomes:
    """Counters for labelled detector firings."""

    true_positive_delays: list[int]
    user_interaction_delays: list[int]
    pre_drift_false_positives: int = 0
    late_detections: int = 0
    repeat_detections: int = 0


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for moving averages, recovery windows, and delay limits."""

    moving_window: int = 250
    recovery_window: int = 300
    recovery_threshold_ratio: float = 0.9
    max_events_after_drift: int = 1500
    recovery_bin_size: int = 50
    recovery_min_bin_events: int = 10
    recovery_sustain_bins: int = 2
    absolute_recovery_threshold: float = 0.15


def evaluate_ranking(
    ranked_items: list[int],
    actual_item: int,
    actual_category: int,
    item_categories: np.ndarray,
    true_distribution: tuple[float, ...],
    behavior_distribution: tuple[float, ...] | None = None,
    candidate_item_ids: tuple[int, ...] | list[int] | np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate a single prequential recommendation."""

    ranked_categories = [int(item_categories[item]) for item in ranked_items]
    rank, hit, ndcg = _item_ranking_metrics(ranked_items, actual_item)
    category_hit, category_top1_hit = _category_hit_metrics(
        ranked_categories,
        actual_category,
    )
    distribution = np.asarray(true_distribution, dtype=float)
    if distribution.size:
        oracle_category = _candidate_constrained_oracle_category(
            distribution,
            item_categories,
            candidate_item_ids,
        )
        oracle_category_top1_hit = 1.0 if actual_category == oracle_category else 0.0
        oracle_utility, recommended_utility, regret = _category_regret_metrics(
            ranked_categories,
            distribution,
            oracle_category,
        )
    else:
        oracle_category_top1_hit = np.nan
        oracle_utility = np.nan
        recommended_utility = np.nan
        regret = np.nan
    if behavior_distribution is None:
        behavior_distribution = true_distribution
    behavior = np.asarray(behavior_distribution, dtype=float)
    if behavior.size:
        behavior_oracle_category = _candidate_constrained_oracle_category(
            behavior,
            item_categories,
            candidate_item_ids,
        )
        (
            behavior_oracle_utility,
            behavior_recommended_utility,
            behavior_regret,
        ) = _category_regret_metrics(
            ranked_categories,
            behavior,
            behavior_oracle_category,
        )
    else:
        behavior_oracle_utility = np.nan
        behavior_recommended_utility = np.nan
        behavior_regret = np.nan

    return {
        "hit": hit,
        "ndcg": float(ndcg),
        "rank": float(rank),
        "category_hit": category_hit,
        "category_top1_hit": category_top1_hit,
        "oracle_category_top1_hit": oracle_category_top1_hit,
        "oracle_category_utility": oracle_utility,
        "recommended_category_utility": recommended_utility,
        "category_regret": regret,
        "behavior_oracle_category_utility": behavior_oracle_utility,
        "behavior_recommended_category_utility": behavior_recommended_utility,
        "behavior_category_regret": behavior_regret,
    }


def _item_ranking_metrics(
    ranked_items: list[int],
    actual_item: int,
) -> tuple[int, float, float]:
    """Return item rank, hit, and NDCG for one recommendation list."""

    try:
        rank = ranked_items.index(actual_item) + 1
    except ValueError:
        return 0, 0.0, 0.0
    return rank, 1.0, float(1.0 / np.log2(rank + 1))


def _category_hit_metrics(
    ranked_categories: list[int],
    actual_category: int,
) -> tuple[float, float]:
    """Return category hit@K and top-1 category hit."""

    category_hit = 1.0 if actual_category in ranked_categories else 0.0
    category_top1_hit = (
        1.0 if ranked_categories and ranked_categories[0] == actual_category else 0.0
    )
    return category_hit, category_top1_hit


def _category_regret_metrics(
    ranked_categories: list[int],
    true_distribution: np.ndarray,
    oracle_category: int,
) -> tuple[float, float, float]:
    """Compare recommended top category utility against oracle utility."""

    recommended_category = ranked_categories[0] if ranked_categories else None
    recommended_utility = (
        float(true_distribution[recommended_category])
        if recommended_category is not None
        else 0.0
    )
    oracle_utility = float(true_distribution[oracle_category])
    regret = max(0.0, oracle_utility - recommended_utility)
    return oracle_utility, recommended_utility, regret


def _candidate_constrained_oracle_category(
    distribution: np.ndarray,
    item_categories: np.ndarray,
    candidate_item_ids: tuple[int, ...] | list[int] | np.ndarray | None,
) -> int:
    """Return the best oracle category among available candidate items."""

    if candidate_item_ids is None:
        return int(np.argmax(distribution))
    candidates = np.asarray(candidate_item_ids, dtype=int)
    if candidates.size == 0:
        return int(np.argmax(distribution))
    available_categories = np.unique(item_categories[candidates])
    valid_categories = available_categories[
        (available_categories >= 0) & (available_categories < distribution.size)
    ]
    if valid_categories.size == 0:
        return int(np.argmax(distribution))
    utilities = distribution[valid_categories]
    return int(valid_categories[int(np.argmax(utilities))])


def add_moving_metrics(
    events: pd.DataFrame, window: int, metrics: list[str] | None = None
) -> pd.DataFrame:
    """Add per-method rolling averages used in time-series plots."""

    if events.empty:
        return events
    if metrics is None:
        metrics = DEFAULT_MOVING_METRICS
    frames = []
    for method, group in events.sort_values("event_id").groupby("method"):
        group = group.copy()
        for metric in metrics:
            if metric in group:
                group[f"{metric}_ma"] = (
                    group[metric].rolling(window=window, min_periods=1).mean()
                )
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def summarise_events(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level metrics into one row per method."""

    if events.empty:
        return pd.DataFrame()
    aggregations = {
        column: aggregation
        for column, aggregation in EVENT_AGGREGATIONS.items()
        if column in events.columns
    }
    summary = events.groupby("method").agg(aggregations)
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary = summary.rename(columns=SUMMARY_COLUMN_NAMES)
    return summary.reset_index()


def summarise_events_by_phase(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics separately for pre, transition, and post phases."""

    if events.empty or "drift_phase" not in events.columns:
        return pd.DataFrame()
    parts = []
    for phase, group in events.groupby("drift_phase", sort=False):
        summary = summarise_events(group)
        if not summary.empty:
            summary.insert(0, "drift_phase", phase)
            parts.append(summary)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def summarise_detection(
    detections: pd.DataFrame,
    user_drift_points: dict[int, int | dict[str, int]],
    max_events_after_drift: int,
) -> pd.DataFrame:
    """Summarise detector speed, coverage, and false positives.

    A detection is counted as a true positive only if it is the first detection
    for that user after the known synthetic drift onset and within the configured
    maximum delay window. For gradual drift, the onset is the start of the
    transition, not the midpoint.
    """

    if detections.empty:
        return pd.DataFrame(columns=DETECTION_COLUMNS)

    drift_windows = _normalise_drift_windows(user_drift_points)
    if not any(window["start"] >= 0 for window in drift_windows.values()):
        return _summarise_unlabelled_detections(detections)
    rows = []
    detected = detections[detections["drift_detected"] == 1]
    method_pairs = detections[["method", "detector"]].drop_duplicates()
    for _, pair in method_pairs.iterrows():
        method = str(pair["method"])
        detector = str(pair["detector"])
        group = detected[
            (detected["method"] == pair["method"])
            & (detected["detector"] == pair["detector"])
        ]
        (
            true_positive_delays,
            user_interaction_delays,
            pre_drift_false_positives,
            late_detections,
            repeat_detections,
        ) = _detection_outcomes(
            group,
            drift_windows,
            max_events_after_drift,
        )
        false_positives = (
            pre_drift_false_positives + late_detections + repeat_detections
        )
        rows.append(
            {
                "method": method,
                "detector": detector,
                "detection_rate": len(true_positive_delays)
                / max(1, len(drift_windows)),
                "mean_delay": float(np.mean(true_positive_delays))
                if true_positive_delays
                else np.nan,
                "median_delay": float(np.median(true_positive_delays))
                if true_positive_delays
                else np.nan,
                "mean_user_interaction_delay": float(np.mean(user_interaction_delays))
                if user_interaction_delays
                else np.nan,
                "median_user_interaction_delay": float(
                    np.median(user_interaction_delays)
                )
                if user_interaction_delays
                else np.nan,
                "pre_drift_false_positives": pre_drift_false_positives,
                "late_detections": late_detections,
                "repeat_detections": repeat_detections,
                "false_positives": false_positives,
                "n_detections": len(group),
            }
        )
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def _summarise_unlabelled_detections(detections: pd.DataFrame) -> pd.DataFrame:
    """Summarise detector activations when no ground-truth drift labels exist."""

    rows = []
    detected = detections[detections["drift_detected"] == 1]
    method_pairs = detections[["method", "detector"]].drop_duplicates()
    for _, pair in method_pairs.iterrows():
        group = detected[
            (detected["method"] == pair["method"])
            & (detected["detector"] == pair["detector"])
        ]
        rows.append(
            {
                "method": str(pair["method"]),
                "detector": str(pair["detector"]),
                "detection_rate": np.nan,
                "mean_delay": np.nan,
                "median_delay": np.nan,
                "mean_user_interaction_delay": np.nan,
                "median_user_interaction_delay": np.nan,
                "pre_drift_false_positives": np.nan,
                "late_detections": np.nan,
                "repeat_detections": np.nan,
                "false_positives": np.nan,
                "n_detections": len(group),
            }
        )
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def _normalise_drift_windows(
    drift_reference: dict[int, int | dict[str, int]],
) -> dict[int, dict[str, int]]:
    """Accept legacy drift points or explicit start/mid/end windows."""

    windows: dict[int, dict[str, int]] = {}
    for user_id, value in drift_reference.items():
        user = int(user_id)
        if isinstance(value, dict):
            start = int(value.get("start", value.get("midpoint", value.get("end", 0))))
            midpoint = int(value.get("midpoint", start))
            end = int(value.get("end", midpoint))
            windows[user] = {"start": start, "midpoint": midpoint, "end": end}
        else:
            point = int(value)
            windows[user] = {"start": point, "midpoint": point, "end": point}
    return windows


def _detection_outcomes(
    detections: pd.DataFrame,
    user_drift_windows: dict[int, dict[str, int]],
    max_events_after_drift: int,
) -> tuple[list[int], list[int], int, int, int]:
    """Classify detector firings into true positives and alarm categories."""

    outcomes = _DetectionOutcomes([], [])
    detected_users: set[int] = set()
    for _, row in detections.sort_values("timestamp").iterrows():
        _record_detection_outcome(
            row,
            user_drift_windows,
            max_events_after_drift,
            detected_users,
            outcomes,
        )
    return (
        outcomes.true_positive_delays,
        outcomes.user_interaction_delays,
        outcomes.pre_drift_false_positives,
        outcomes.late_detections,
        outcomes.repeat_detections,
    )


def _record_detection_outcome(
    row: pd.Series,
    user_drift_windows: dict[int, dict[str, int]],
    max_events_after_drift: int,
    detected_users: set[int],
    outcomes: _DetectionOutcomes,
) -> None:
    """Update outcome counters for one detector firing."""

    user_id = int(row["user_id"])
    timestamp = int(row["timestamp"])
    drift_start = int(user_drift_windows[user_id]["start"])
    delay = timestamp - drift_start
    if user_id in detected_users:
        outcomes.repeat_detections += 1
        return
    if delay < 0:
        outcomes.pre_drift_false_positives += 1
        return
    if delay > max_events_after_drift:
        outcomes.late_detections += 1
        return

    outcomes.true_positive_delays.append(delay)
    if "user_events_since_drift_start" in row.index:
        outcomes.user_interaction_delays.append(
            int(row["user_events_since_drift_start"])
        )
    detected_users.add(user_id)


def summarise_recovery(
    events: pd.DataFrame,
    user_drift_points: dict[int, int | dict[str, int]],
    config: EvaluationConfig,
) -> pd.DataFrame:
    """Estimate sustained recovery after each user's drift transition completes."""

    if events.empty:
        return pd.DataFrame()
    drift_windows = _normalise_drift_windows(user_drift_points)
    if not any(window["end"] >= 0 for window in drift_windows.values()):
        return _not_applicable_recovery(events)
    rows = []
    for method, group in events.sort_values("event_id").groupby("method"):
        aligned = _add_user_drift_alignment(group, drift_windows)
        before, after = _recovery_windows(aligned, config)
        if before.empty or after.empty:
            recovery_time = np.nan
            absolute_recovery_time = np.nan
            threshold = np.nan
            pre_perf = np.nan
            post_perf = np.nan
            status = "insufficient_data"
        else:
            pre_perf = float(before["category_top1_hit"].mean())
            post_perf = float(after["category_top1_hit"].mean())
            threshold = max(0.05, pre_perf * config.recovery_threshold_ratio)
            recovery_time = _first_recovery_time(
                after, threshold, config, "category_top1_hit"
            )
            absolute_recovery_time = _first_recovery_time(
                after,
                config.absolute_recovery_threshold,
                config,
                "category_top1_hit",
            )
            if pd.isna(recovery_time):
                status = "not_recovered"
            else:
                status = "recovered"
        rows.append(
            {
                "method": method,
                "aligned_drift_event": 0,
                "pre_drift_category_top1": pre_perf,
                "post_drift_category_top1": post_perf,
                "recovery_threshold": threshold,
                "recovery_time_events": recovery_time,
                "absolute_recovery_threshold": config.absolute_recovery_threshold,
                "absolute_recovery_time_events": absolute_recovery_time,
                "recovery_status": status,
            }
        )
    return pd.DataFrame(rows)


def _not_applicable_recovery(events: pd.DataFrame) -> pd.DataFrame:
    """Return recovery rows for streams without labelled drift points."""

    rows = []
    for method in events["method"].drop_duplicates():
        rows.append(
            {
                "method": method,
                "aligned_drift_event": np.nan,
                "pre_drift_category_top1": np.nan,
                "post_drift_category_top1": np.nan,
                "recovery_threshold": np.nan,
                "recovery_time_events": np.nan,
                "absolute_recovery_threshold": np.nan,
                "absolute_recovery_time_events": np.nan,
                "recovery_status": "not_applicable",
            }
        )
    return pd.DataFrame(rows)


def _add_user_drift_alignment(
    events: pd.DataFrame,
    user_drift_windows: dict[int, dict[str, int]],
) -> pd.DataFrame:
    """Add event distances from the user's drift start and completed endpoint."""

    aligned = events.copy()
    if "drift_start" in aligned:
        drift_starts = aligned["drift_start"].astype(int)
    else:
        drift_starts = aligned["user_id"].map(
            lambda user_id: user_drift_windows[int(user_id)]["start"]
        )
    if "drift_end" in aligned:
        drift_ends = aligned["drift_end"].astype(int)
    else:
        drift_ends = aligned["user_id"].map(
            lambda user_id: user_drift_windows[int(user_id)]["end"]
        )
    aligned["events_until_drift_start"] = aligned["event_id"].astype(int) - drift_starts
    aligned["events_since_user_drift"] = aligned["event_id"].astype(int) - drift_ends
    return aligned


def _recovery_windows(
    events: pd.DataFrame,
    config: EvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split aligned events into pre-drift reference and post-drift recovery."""

    before_elapsed = events["events_until_drift_start"]
    after_elapsed = events["events_since_user_drift"]
    before = events[
        (before_elapsed >= -config.recovery_window) & (before_elapsed < 0)
    ]
    after = events[
        (after_elapsed >= 0)
        & (after_elapsed <= config.max_events_after_drift)
    ].copy()
    return before, after


def _first_recovery_time(
    after: pd.DataFrame,
    threshold: float,
    config: EvaluationConfig,
    metric: str,
) -> int | float:
    """Return the first sustained post-drift bin at or above a threshold."""

    if after.empty or metric not in after.columns:
        return np.nan
    bin_size = max(1, int(config.recovery_bin_size))
    min_bin_events = max(1, int(config.recovery_min_bin_events))
    sustain_bins = max(1, int(config.recovery_sustain_bins))

    post_drift = after.copy()
    post_drift["recovery_bin"] = (
        post_drift["events_since_user_drift"].astype(int) // bin_size
    ) * bin_size
    binned = (
        post_drift.groupby("recovery_bin", as_index=False)[metric]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("recovery_bin")
    )
    binned = binned[binned["count"] >= min_bin_events].reset_index(drop=True)
    if binned.empty:
        return np.nan

    meets_threshold = binned["mean"] >= threshold
    for idx in range(0, len(binned) - sustain_bins + 1):
        window = binned.iloc[idx : idx + sustain_bins]
        if not bool(meets_threshold.iloc[idx : idx + sustain_bins].all()):
            continue
        expected_bins = np.arange(
            int(window.iloc[0]["recovery_bin"]),
            int(window.iloc[0]["recovery_bin"]) + sustain_bins * bin_size,
            bin_size,
        )
        if np.array_equal(window["recovery_bin"].to_numpy(dtype=int), expected_bins):
            return int(window.iloc[0]["recovery_bin"])
    return np.nan


def save_tables(
    output_dir: Path,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    detections: pd.DataFrame,
    detection_summary: pd.DataFrame,
    recovery: pd.DataFrame,
) -> None:
    """Persist all event-level and aggregate result tables as CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "events.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    phase_summary.to_csv(output_dir / "phase_summary.csv", index=False)
    detections.to_csv(output_dir / "detections.csv", index=False)
    detection_summary.to_csv(output_dir / "detection_summary.csv", index=False)
    recovery.to_csv(output_dir / "recovery.csv", index=False)
