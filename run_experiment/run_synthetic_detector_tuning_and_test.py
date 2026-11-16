"""Run synthetic detector-threshold tuning and final-test validation.

This protocol mirrors the detector-specific MIND validation runner, but keeps
the labelled synthetic scenarios used for supervised detection, recovery, and
regret claims. Thresholds are selected on tuning scenarios/seeds only and then
frozen for final-test seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from run_experiment.run_single_prequential_experiment import load_config, run_experiment
from src.experiment.protocol import (
    ProtocolAggregation,
    SCENARIOS,
    append_run_tables,
    build_protocol_run_config,
    concat_table_parts,
    empty_protocol_tables,
    empty_table_parts,
    parse_floats,
    parse_ints,
    reuse_completed_run,
    select_threshold,
    threshold_label,
    write_protocol_phase,
)


TUNING_SEEDS = (1, 2, 3)
TEST_SEEDS = (10, 11, 12, 13, 14)
TUNING_SCENARIOS = ("sudden_strong", "gradual_strong")
TEST_SCENARIOS = (
    "sudden_strong",
    "sudden_weak",
    "gradual_strong",
    "noisy_sudden",
    "stable_no_drift",
)
KL_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60)
JSD_THRESHOLDS = (0.05, 0.08, 0.10, 0.12, 0.16)
DETECTOR_SETTINGS = {
    "kl": {
        "thresholds": KL_THRESHOLDS,
        "threshold_column": "kl_threshold",
        "candidate_methods": ("drift_aware_kl", "drift_aware_kl_clustered"),
        "name_prefix": "synthetic_kl",
        "threshold_name": "KL",
        "output_root": "experiment-results/synthetic/kl_protocol",
    },
    "jsd": {
        "thresholds": JSD_THRESHOLDS,
        "threshold_column": "jsd_threshold",
        "candidate_methods": ("drift_aware_jsd", "drift_aware_jsd_clustered"),
        "name_prefix": "synthetic_jsd",
        "threshold_name": "JSD",
        "output_root": "experiment-results/synthetic/jsd_protocol",
    },
}
SUMMARY_METRICS = (
    "ctr_at_k",
    "ndcg_at_k",
    "category_hit_at_k",
    "category_top1_accuracy",
    "mean_regret",
    "cumulative_regret",
    "mean_behavior_regret",
    "cumulative_behavior_regret",
    "rerank_active_rate",
)
DETECTION_METRICS = (
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
)
RECOVERY_METRICS = (
    "pre_drift_category_top1",
    "post_drift_category_top1",
    "recovery_threshold",
    "recovery_time_events",
    "absolute_recovery_threshold",
    "absolute_recovery_time_events",
)
FINAL_METHODS = (
    "base_mf",
    "time_decay_mf",
    "recent_popularity",
    "global_popularity",
    "mf_rerank_always",
    "mf_rerank_always_preference_only",
    "mf_rerank_always_transition_only",
    "mf_rerank_oracle",
    "drift_aware_kl",
    "drift_aware_jsd",
    "drift_aware_kl_clustered",
    "drift_aware_jsd_clustered",
)
TABLE_FILES = {
    "summary": "summary.csv",
    "detection": "detection_summary.csv",
    "recovery": "recovery.csv",
}


def main() -> None:
    """Parse CLI arguments and run the requested synthetic detector protocol."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_synthetic_mind_shaped.yaml")
    parser.add_argument(
        "--detector",
        choices=sorted(DETECTOR_SETTINGS),
        default="kl",
        help="Which synthetic detector threshold to tune.",
    )
    parser.add_argument("--phase", choices=["tuning", "test", "both"], default="both")
    parser.add_argument("--output-root")
    parser.add_argument("--n-events-tuning", type=int, default=8000)
    parser.add_argument("--n-events-test", type=int, default=12000)
    parser.add_argument("--thresholds", help="Comma-separated thresholds for tuning.")
    parser.add_argument(
        "--tuning-seeds",
        default=",".join(str(seed) for seed in TUNING_SEEDS),
    )
    parser.add_argument(
        "--test-seeds",
        default=",".join(str(seed) for seed in TEST_SEEDS),
    )
    args = parser.parse_args()

    settings = detector_settings(args.detector)
    thresholds = (
        parse_floats(args.thresholds)
        if args.thresholds
        else list(settings["thresholds"])
    )
    output_root = (
        Path(args.output_root) if args.output_root else Path(str(settings["output_root"]))
    )

    run_synthetic_detector_tuning_and_test(
        config_path=Path(args.config),
        detector=args.detector,
        phase=args.phase,
        output_root=output_root,
        n_events_tuning=args.n_events_tuning,
        n_events_test=args.n_events_test,
        thresholds=thresholds,
        tuning_seeds=parse_ints(args.tuning_seeds),
        test_seeds=parse_ints(args.test_seeds),
    )


def run_synthetic_detector_tuning_and_test(
    config_path: Path,
    detector: str,
    phase: str,
    output_root: Path,
    n_events_tuning: int,
    n_events_test: int,
    thresholds: list[float],
    tuning_seeds: list[int],
    test_seeds: list[int],
) -> None:
    """Run tuning/final-test synthetic detector validation and write tables."""

    settings = detector_settings(detector)
    base_config = load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)

    tuning = empty_protocol_tables(TABLE_FILES)
    frozen_threshold = float(base_config["drift_detection"][detector]["threshold"])
    if phase in {"tuning", "both"}:
        tuning = run_tuning(
            base_config=base_config,
            detector=detector,
            config_path=config_path,
            output_root=output_root / "tuning",
            n_events=n_events_tuning,
            thresholds=thresholds,
            seeds=tuning_seeds,
        )
        frozen_threshold = select_threshold(tuning["summary"], settings)

    final = empty_protocol_tables(TABLE_FILES)
    if phase in {"test", "both"}:
        final = run_final(
            base_config=base_config,
            detector=detector,
            config_path=config_path,
            output_root=output_root / "final",
            n_events=n_events_test,
            threshold=frozen_threshold,
            seeds=test_seeds,
        )

    write_outputs(output_root, settings, tuning, final)


def run_tuning(
    base_config: dict[str, Any],
    detector: str,
    config_path: Path,
    output_root: Path,
    n_events: int,
    thresholds: list[float],
    seeds: list[int],
) -> dict[str, pd.DataFrame]:
    """Run every threshold/scenario/seed pair for synthetic tuning."""

    settings = detector_settings(detector)
    threshold_column = str(settings["threshold_column"])
    name_prefix = str(settings["name_prefix"])
    parts = empty_table_parts(TABLE_FILES)
    for threshold in thresholds:
        for scenario in TUNING_SCENARIOS:
            for seed in seeds:
                label = threshold_label(threshold)
                run_dir = output_root / label / scenario / f"seed_{seed}"
                outputs = reuse_completed_run(
                    run_dir=run_dir,
                    required_files=(*TABLE_FILES.values(), "metadata.json"),
                    config=build_run_config(
                        base_config=base_config,
                        detector=detector,
                        name=f"{name_prefix}_tuning_{label}_{scenario}_seed_{seed}",
                        seed=seed,
                        n_events=n_events,
                        output_dir=run_dir,
                        scenario=scenario,
                        threshold=threshold,
                    ),
                    config_path=config_path,
                    runner=run_experiment,
                )
                annotations = [
                    ("phase", "tuning"),
                    (threshold_column, threshold),
                    ("scenario", scenario),
                    ("seed", seed),
                ]
                append_run_tables(
                    parts,
                    outputs["output_dir"],
                    annotations,
                    TABLE_FILES,
                )
    return concat_table_parts(parts)


def run_final(
    base_config: dict[str, Any],
    detector: str,
    config_path: Path,
    output_root: Path,
    n_events: int,
    threshold: float,
    seeds: list[int],
) -> dict[str, pd.DataFrame]:
    """Run frozen-threshold final synthetic scenarios and seeds."""

    settings = detector_settings(detector)
    threshold_column = str(settings["threshold_column"])
    name_prefix = str(settings["name_prefix"])
    parts = empty_table_parts(TABLE_FILES)
    for scenario in TEST_SCENARIOS:
        for seed in seeds:
            run_dir = output_root / scenario / f"seed_{seed}"
            outputs = reuse_completed_run(
                run_dir=run_dir,
                required_files=(*TABLE_FILES.values(), "metadata.json"),
                config=build_run_config(
                    base_config=base_config,
                    detector=detector,
                    name=f"{name_prefix}_final_{scenario}_seed_{seed}",
                    seed=seed,
                    n_events=n_events,
                    output_dir=run_dir,
                    scenario=scenario,
                    threshold=threshold,
                ),
                config_path=config_path,
                runner=run_experiment,
            )
            annotations = [
                ("phase", "test"),
                (threshold_column, threshold),
                ("scenario", scenario),
                ("seed", seed),
            ]
            append_run_tables(
                parts,
                outputs["output_dir"],
                annotations,
                TABLE_FILES,
            )
    return concat_table_parts(parts)


def build_run_config(
    base_config: dict[str, Any],
    detector: str,
    name: str,
    seed: int,
    n_events: int,
    output_dir: Path,
    scenario: str,
    threshold: float,
) -> dict[str, Any]:
    """Create one concrete synthetic protocol config."""

    return build_protocol_run_config(
        base_config=base_config,
        name=name,
        seed=seed,
        n_events=n_events,
        output_dir=output_dir,
        scenario_overrides=SCENARIOS[scenario],
        extra_overrides={f"drift_detection.{detector}.threshold": float(threshold)},
    )


def write_outputs(
    output_root: Path,
    settings: dict[str, Any],
    tuning: dict[str, pd.DataFrame],
    final: dict[str, pd.DataFrame],
) -> None:
    """Save raw and aggregate protocol CSV tables."""

    threshold_column = str(settings["threshold_column"])
    write_protocol_phase(
        output_root,
        "tuning",
        tuning,
        protocol_aggregations("tuning", threshold_column),
    )
    write_protocol_phase(
        output_root,
        "final",
        final,
        protocol_aggregations("final"),
    )


def protocol_aggregations(
    phase: str,
    threshold_column: str | None = None,
) -> list[ProtocolAggregation]:
    """Return aggregate table definitions for one synthetic protocol phase."""

    prefix = ("phase",) + ((threshold_column,) if threshold_column else ())
    metrics_by_table = {
        "summary": SUMMARY_METRICS,
        "detection": DETECTION_METRICS,
        "recovery": RECOVERY_METRICS,
    }
    suffix_by_table = {
        "summary": ("method",),
        "detection": ("method", "detector"),
        "recovery": ("method",),
    }
    aggregations = []
    for table, metrics in metrics_by_table.items():
        suffix = suffix_by_table[table]
        aggregations.extend(
            [
                ProtocolAggregation(
                    table,
                    f"{phase}_{table}_by_scenario.csv",
                    prefix + ("scenario",) + suffix,
                    metrics,
                ),
                ProtocolAggregation(
                    table,
                    f"{phase}_{table}_overall.csv",
                    prefix + suffix,
                    metrics,
                ),
            ]
        )
    return aggregations


def detector_settings(detector: str) -> dict[str, Any]:
    """Return settings for a supported synthetic detector."""

    try:
        return DETECTOR_SETTINGS[detector]
    except KeyError as exc:
        raise ValueError(f"Unsupported detector: {detector}") from exc


if __name__ == "__main__":
    main()
