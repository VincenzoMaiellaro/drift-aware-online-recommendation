"""Run prequential drift-aware recommendation experiments.

The runner builds the configured stream, instantiates the comparison methods,
runs the online predict-evaluate-update loop, and saves CSV result tables.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.core.events import InteractionEvent
from src.data.stream_factory import build_stream
from src.experiment.evaluation import (
    EvaluationConfig,
    add_moving_metrics,
    evaluate_ranking,
    save_tables,
    summarise_detection,
    summarise_events,
    summarise_events_by_phase,
    summarise_recovery,
)
from src.experiment.protocol import set_nested
from src.methods.factory import MethodState, build_methods
from src.methods.recommender import Recommendation

BEHAVIOR_SHIFT_COLUMNS = [
    "method",
    "detector",
    "event_id",
    "timestamp",
    "user_id",
    "source_event_id",
    "source_timestamp",
    "source_history_size",
    "category_level",
    "divergence_score",
    "score_name",
    "kl_score",
    "jsd_score",
    "threshold",
    "baseline_top_category",
    "baseline_top_category_name",
    "recent_top_category",
    "recent_top_category_name",
    "baseline_top_probability",
    "recent_top_probability",
    "recent_window_size",
    "rule_named",
    "rule",
]


@dataclass(frozen=True)
class RunContext:
    """Resolved objects shared across one experiment run."""

    stream: Any
    output_dir: Path
    methods: list[MethodState]
    k: int
    rerank_pool: int


def main() -> None:
    """Parse command-line arguments and launch one experiment."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_synthetic_mind_shaped.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from the config.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values, for example data.n_events=5000.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    for override in args.set:
        apply_override(config, override)

    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir

    run_experiment(config, Path(args.config))


def run_experiment(
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, Path]:
    """Execute one complete prequential experiment.

    The loop first asks every method for a recommendation, evaluates that
    recommendation on the current event, then updates recommenders, detectors,
    pattern histories, and reranker activation state. This preserves the
    test-then-train protocol.
    """

    context = build_run_context(config)
    rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    behavior_shift_rows: list[dict[str, Any]] = []
    for event in context.stream:
        event_rows = evaluate_methods_before_update(context, event)
        update_methods_after_evaluation(
            context.methods, event, detection_rows, behavior_shift_rows
        )
        rows.extend(event_rows)

    stream_metadata = context.stream.metadata()
    events = build_events_table(rows, config)
    if not bool(stream_metadata.get("has_ground_truth_drift", True)):
        events = drop_unlabelled_oracle_metrics(events)
    detections = pd.DataFrame(detection_rows)
    user_drift_windows = stream_user_drift_windows(context.stream)
    eval_cfg = EvaluationConfig(**config["evaluation"])
    summary = summarise_events(events)
    phase_summary = summarise_events_by_phase(events)
    detection_summary = summarise_detection(
        detections, user_drift_windows, eval_cfg.max_events_after_drift
    )
    recovery = summarise_recovery(events, user_drift_windows, eval_cfg)
    save_run_outputs(
        context=context,
        config=config,
        config_path=config_path,
        events=events,
        summary=summary,
        phase_summary=phase_summary,
        detections=detections,
        detection_summary=detection_summary,
        recovery=recovery,
        behavior_shifts=build_behavior_shifts_table(
            behavior_shift_rows,
            stream_metadata,
        ),
    )

    print("Experiment complete")
    print(f"Results: {context.output_dir}")
    return {"output_dir": context.output_dir}


def build_run_context(config: dict[str, Any]) -> RunContext:
    """Resolve stream, output paths, methods, and ranking sizes for a run."""

    seed = int(config["experiment"].get("seed", 42))
    stream = build_stream(config, seed)
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    k = int(config["recommendation"]["k"])
    rerank_pool = max(k, int(config["recommendation"].get("candidate_pool_size", k)))
    return RunContext(
        stream=stream,
        output_dir=output_dir,
        methods=build_methods(config, stream, seed),
        k=k,
        rerank_pool=rerank_pool,
    )


def evaluate_methods_before_update(
    context: RunContext,
    event: InteractionEvent,
) -> list[dict[str, Any]]:
    """Evaluate every method on the event before any model sees it."""

    return [
        evaluate_method_before_update(context, method, event)
        for method in context.methods
    ]


def evaluate_method_before_update(
    context: RunContext,
    method: MethodState,
    event: InteractionEvent,
) -> dict[str, Any]:
    """Generate one recommendation row for the prequential event table."""

    candidate_items = event.candidate_item_ids
    recommendation_k = context.rerank_pool
    if candidate_items:
        recommendation_k = min(recommendation_k, len(candidate_items))
    base_rec = method.recommender.recommend(
        event.user_id,
        recommendation_k,
        candidate_items=candidate_items,
    )
    rec, rerank_details = apply_optional_reranking(
        method,
        base_rec,
        event,
        context.k,
    )
    metric_row = evaluate_ranking(
        rec.item_ids,
        event.item_id,
        event.category_id,
        context.stream.item_categories,
        event.true_user_distribution,
        event.behavior_user_distribution,
        event.candidate_item_ids,
    )
    return event_result_row(
        method=method,
        event=event,
        recommendation=rec,
        item_categories=context.stream.item_categories,
        rerank_details=rerank_details,
        metric_row=metric_row,
    )


def apply_optional_reranking(
    method: MethodState,
    base_rec: Recommendation,
    event: InteractionEvent,
    k: int,
) -> tuple[Recommendation, dict[str, Any]]:
    """Apply adaptive reranking when the method has an active reranker."""

    if method.reranker and method.pattern_extractor:
        features = method.pattern_extractor.features(event.user_id)
        cluster_details: dict[str, Any] = {}
        force_active = False
        active_reason: str | None = None
        if method.clusterer is not None:
            features, cluster_details = method.clusterer.features_for_user(
                event.user_id,
                event.timestamp,
                features,
            )
            force_active = bool(cluster_details.get("cluster_rule_active", 0)) and bool(
                method.cluster_force_activation
            )
            if force_active:
                active_reason = "cluster"
        rec, rerank_details = method.reranker.rerank(
            base_rec,
            event.user_id,
            event.timestamp,
            features,
            k,
            force_active=force_active,
            active_reason=active_reason,
        )
        rerank_details.update(cluster_details)
        return rec, rerank_details
    return truncate_recommendation(base_rec, k), {"rerank_active": 0}


def truncate_recommendation(recommendation: Recommendation, k: int) -> Recommendation:
    """Keep the first K recommendations while preserving the dataclass type."""

    return type(recommendation)(
        item_ids=recommendation.item_ids[:k],
        scores=recommendation.scores[:k],
    )


def event_result_row(
    method: MethodState,
    event: InteractionEvent,
    recommendation: Recommendation,
    item_categories: Any,
    rerank_details: dict[str, Any],
    metric_row: dict[str, float],
) -> dict[str, Any]:
    """Build the event-level result row saved to events.csv."""

    top_item = recommendation.item_ids[0] if recommendation.item_ids else -1
    top_category = int(item_categories[top_item]) if top_item >= 0 else -1
    return {
        "method": method.name,
        "event_id": event.event_id,
        "timestamp": event.timestamp,
        "user_id": event.user_id,
        "user_event_index": event.user_event_index,
        "user_events_since_drift_start": event.user_events_since_drift_start,
        "item_id": event.item_id,
        "category_id": event.category_id,
        "user_drift_point": event.user_drift_point,
        "drift_start": event.drift_start,
        "drift_end": event.drift_end,
        "drift_phase": event.drift_phase,
        "is_after_drift": int(event.is_after_drift),
        "drift_progress": event.drift_progress,
        "candidate_count": (
            len(event.candidate_item_ids) if event.candidate_item_ids else 0
        ),
        "source_event_id": event.source_event_id or "",
        "source_timestamp": event.source_timestamp or "",
        "source_history_size": event.source_history_size,
        "rerank_active": int(rerank_details.get("rerank_active", 0)),
        "rerank_reason": rerank_details.get("rerank_reason", ""),
        "adaptive_mean": rerank_details.get("adaptive_mean", 0.0),
        "base_mean": rerank_details.get("base_mean", 0.0),
        "cluster_id": int(rerank_details.get("cluster_id", -1)),
        "cluster_rule_active": int(rerank_details.get("cluster_rule_active", 0)),
        "cluster_rule_weight": float(rerank_details.get("cluster_rule_weight", 0.0)),
        "cluster_rule_id": int(rerank_details.get("cluster_rule_id", -1)),
        "cluster_similarity": float(rerank_details.get("cluster_similarity", 0.0)),
        "cluster_rule_age": int(rerank_details.get("cluster_rule_age", -1)),
        "cluster_rule_source_user": int(
            rerank_details.get("cluster_rule_source_user", -1)
        ),
        "cluster_rule_support": int(rerank_details.get("cluster_rule_support", 0)),
        "cluster_rule_detector": rerank_details.get("cluster_rule_detector", ""),
        "top_item": top_item,
        "top_category": top_category,
        **metric_row,
    }


def update_methods_after_evaluation(
    methods: list[MethodState],
    event: InteractionEvent,
    detection_rows: list[dict[str, Any]],
    behavior_shift_rows: list[dict[str, Any]],
) -> None:
    """Update recommenders, detectors, reranker state, and pattern histories."""

    for method in methods:
        method.recommender.update(event)

        signal = update_detector_and_reranker(
            method, event, detection_rows, behavior_shift_rows
        )
        if method.pattern_extractor is not None:
            method.pattern_extractor.update(event)
        if method.clusterer is not None and method.pattern_extractor is not None:
            features = method.pattern_extractor.features(event.user_id)
            if signal is not None and signal.drift_detected:
                method.clusterer.publish_rule(
                    event.user_id,
                    event.timestamp,
                    features,
                    signal.detector,
                )
            else:
                method.clusterer.update_user(event.user_id, event.timestamp, features)


def update_detector_and_reranker(
    method: MethodState,
    event: InteractionEvent,
    detection_rows: list[dict[str, Any]],
    behavior_shift_rows: list[dict[str, Any]],
) -> Any | None:
    """Record detector output and activate reranking after detected drift."""

    if method.detector is None:
        return None
    signal = method.detector.update(event)
    detection_rows.append(
        {
            "method": method.name,
            "detector": signal.detector,
            "timestamp": signal.timestamp,
            "event_id": event.event_id,
            "user_id": signal.user_id,
            "user_event_index": event.user_event_index,
            "user_events_since_drift_start": event.user_events_since_drift_start,
            "drift_start": event.drift_start,
            "drift_end": event.drift_end,
            "drift_phase": event.drift_phase,
            "drift_detected": int(signal.drift_detected),
            "score": signal.score,
            "threshold": signal.threshold,
            **{f"detail_{key}": value for key, value in signal.details.items()},
        }
    )
    if signal.drift_detected and method.reranker is not None:
        method.reranker.activate(event.user_id, event.timestamp)
    if signal.drift_detected:
        maybe_record_behavior_shift(method, event, signal, behavior_shift_rows)
    return signal


def maybe_record_behavior_shift(
    method: MethodState,
    event: InteractionEvent,
    signal: Any,
    behavior_shift_rows: list[dict[str, Any]],
) -> None:
    """Save interpretable MIND-style divergence behavior shifts when available."""

    if signal.detector not in {"history_kl", "history_jsd"}:
        return
    details = signal.details
    score_name = "kl_score" if signal.detector == "history_kl" else "jsd_score"
    behavior_shift_rows.append(
        {
            "method": method.name,
            "detector": signal.detector,
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "user_id": event.user_id,
            "source_event_id": event.source_event_id or "",
            "source_timestamp": event.source_timestamp or "",
            "source_history_size": event.source_history_size,
            "divergence_score": signal.score,
            "score_name": score_name,
            "kl_score": signal.score if signal.detector == "history_kl" else "",
            "jsd_score": signal.score if signal.detector == "history_jsd" else "",
            "threshold": signal.threshold,
            "baseline_top_category": details.get("baseline_top_category"),
            "recent_top_category": details.get("recent_top_category"),
            "baseline_top_probability": details.get("baseline_top_probability"),
            "recent_top_probability": details.get("recent_top_probability"),
            "recent_window_size": details.get("recent_window_size"),
            "rule": details.get("rule", ""),
        }
    )


def build_events_table(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Create the events table and add configured moving averages."""

    events = pd.DataFrame(rows)
    return add_moving_metrics(
        events,
        int(config["evaluation"].get("moving_window", 250)),
    )


UNLABELLED_ORACLE_METRIC_COLUMNS = {
    "oracle_category_top1_hit",
    "oracle_category_utility",
    "recommended_category_utility",
    "category_regret",
    "behavior_oracle_category_utility",
    "behavior_recommended_category_utility",
    "behavior_category_regret",
    "category_regret_ma",
    "behavior_category_regret_ma",
}


def drop_unlabelled_oracle_metrics(events: pd.DataFrame) -> pd.DataFrame:
    """Remove oracle/regret metrics when the stream has no preference truth."""

    return events.drop(
        columns=[
            column
            for column in UNLABELLED_ORACLE_METRIC_COLUMNS
            if column in events.columns
        ],
        errors="ignore",
    )


def build_behavior_shifts_table(
    rows: list[dict[str, Any]],
    stream_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Attach category names to interpretable KL shifts."""

    shifts = pd.DataFrame(rows)
    if shifts.empty:
        return pd.DataFrame(columns=BEHAVIOR_SHIFT_COLUMNS)
    category_names = list(stream_metadata.get("category_names", []))
    category_level = str(stream_metadata.get("category_level", "category"))
    shifts["category_level"] = category_level
    shifts["baseline_top_category_name"] = shifts["baseline_top_category"].apply(
        lambda value: category_name(value, category_names)
    )
    shifts["recent_top_category_name"] = shifts["recent_top_category"].apply(
        lambda value: category_name(value, category_names)
    )
    shifts["rule_named"] = shifts.apply(
        lambda row: (
            f"IF recent behavior shifts from {category_level} "
            f"{row['baseline_top_category_name']} to "
            f"{row['recent_top_category_name']} THEN boost "
            f"{category_level} {row['recent_top_category_name']}"
        ),
        axis=1,
    )
    return shifts.reindex(columns=BEHAVIOR_SHIFT_COLUMNS)


def category_name(value: Any, category_names: list[str]) -> str:
    """Return a display label for a compact category ID."""

    if pd.isna(value):
        return ""
    index = int(value)
    if 0 <= index < len(category_names):
        return category_names[index]
    return str(index)


def stream_user_drift_windows(stream: Any) -> dict[int, dict[str, int]]:
    """Return user drift start/midpoint/end windows from stream metadata."""

    metadata = stream.metadata()
    points = metadata["user_drift_points"]
    starts = metadata.get("user_drift_starts", points)
    ends = metadata.get("user_drift_ends", points)
    return {
        int(user_id): {
            "start": int(starts[user_id]),
            "midpoint": int(drift_point),
            "end": int(ends[user_id]),
        }
        for user_id, drift_point in points.items()
    }


def save_run_outputs(
    context: RunContext,
    config: dict[str, Any],
    config_path: Path,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    detections: pd.DataFrame,
    detection_summary: pd.DataFrame,
    recovery: pd.DataFrame,
    behavior_shifts: pd.DataFrame | None = None,
) -> None:
    """Save CSV tables, metadata, and the exact configuration used."""

    saved_config_path = context.output_dir / "config_used.yaml"
    saved_config_path.unlink(missing_ok=True)
    save_tables(
        context.output_dir,
        events,
        summary,
        phase_summary,
        detections,
        detection_summary,
        recovery,
    )
    metadata = {
        "config_path": str(config_path),
        "stream": context.stream.metadata(),
        "methods": [method.name for method in context.methods],
    }
    (context.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    if behavior_shifts is not None:
        behavior_shifts.to_csv(context.output_dir / "behavior_shifts.csv", index=False)
    saved_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML experiment configuration from disk."""

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def apply_override(config: dict[str, Any], override: str) -> None:
    """Apply a CLI override like `data.n_events=5000` to a nested config."""

    if "=" not in override:
        raise ValueError(f"Override must use key=value syntax: {override}")
    key, raw_value = override.split("=", 1)
    set_nested(config, key, parse_scalar(raw_value))


def parse_scalar(value: str) -> Any:
    """Parse CLI override values into bool, int, float, or string."""

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
