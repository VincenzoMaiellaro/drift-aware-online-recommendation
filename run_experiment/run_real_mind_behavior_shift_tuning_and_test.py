"""Run real-MIND history-divergence tuning and final-test validation.

MINDsmall has no labelled drift points, so this protocol aggregates ranking
quality and behavior-shift activation counts across seeds. It does not report
supervised detection delay, false positives, recovery, or regret claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from run_experiment.run_single_prequential_experiment import load_config, run_experiment
from src.experiment.protocol import (
    ProtocolAggregation,
    append_run_tables,
    build_protocol_run_config,
    concat_table_parts,
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
KL_THRESHOLDS = (0.45, 0.55, 0.65, 0.75, 0.85)
JSD_THRESHOLDS = (0.05, 0.08, 0.10, 0.12, 0.16)
DETECTOR_SETTINGS = {
    "kl": {
        "thresholds": KL_THRESHOLDS,
        "threshold_column": "kl_threshold",
        "candidate_methods": ("drift_aware_kl", "drift_aware_kl_clustered"),
        "name_prefix": "mind_kl",
        "threshold_name": "KL",
        "output_root": "experiment-results/mind/kl_protocol",
    },
    "jsd": {
        "thresholds": JSD_THRESHOLDS,
        "threshold_column": "jsd_threshold",
        "candidate_methods": ("drift_aware_jsd", "drift_aware_jsd_clustered"),
        "name_prefix": "mind_jsd",
        "threshold_name": "JSD",
        "output_root": "experiment-results/mind/jsd_protocol",
    },
}
SUMMARY_METRICS = (
    "ctr_at_k",
    "ndcg_at_k",
    "category_hit_at_k",
    "category_top1_accuracy",
    "rerank_active_rate",
)
DETECTION_METRICS = ("n_detections",)
FINAL_METHODS = (
    "base_mf",
    "time_decay_mf",
    "recent_popularity",
    "mf_rerank_always",
    "drift_aware_kl",
    "drift_aware_jsd",
    "drift_aware_kl_clustered",
    "drift_aware_jsd_clustered",
)
TABLE_FILES = {
    "summary": "summary.csv",
    "detection": "detection_summary.csv",
}


def main() -> None:
    """Parse command-line options and run the requested MIND protocol."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_real_mind_behavior_shift.yaml")
    parser.add_argument(
        "--detector",
        choices=sorted(DETECTOR_SETTINGS),
        default="jsd",
        help="Which real-MIND history divergence threshold to tune.",
    )
    parser.add_argument(
        "--phase",
        choices=["tuning", "test", "both"],
        default="both",
    )
    parser.add_argument("--output-root")
    parser.add_argument(
        "--n-events",
        type=int,
        default=0,
        help="MIND clicked impressions per run. Use 0 for all events.",
    )
    parser.add_argument(
        "--thresholds",
        help="Comma-separated history-divergence thresholds for tuning.",
    )
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

    run_real_mind_behavior_shift_tuning_and_test(
        config_path=Path(args.config),
        detector=args.detector,
        phase=args.phase,
        output_root=output_root,
        n_events=args.n_events,
        thresholds=thresholds,
        tuning_seeds=parse_ints(args.tuning_seeds),
        test_seeds=parse_ints(args.test_seeds),
    )


def run_real_mind_behavior_shift_tuning_and_test(
    config_path: Path,
    detector: str,
    phase: str,
    output_root: Path,
    n_events: int,
    thresholds: list[float],
    tuning_seeds: list[int],
    test_seeds: list[int],
) -> None:
    """Run tuning/final-test real-MIND validation and write aggregate tables."""

    settings = detector_settings(detector)
    base_config = load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)

    tuning_summary = pd.DataFrame()
    tuning_detection = pd.DataFrame()
    frozen_threshold = float(base_config["drift_detection"][detector]["threshold"])
    if phase in {"tuning", "both"}:
        tuning_summary, tuning_detection = run_tuning(
            base_config=base_config,
            detector=detector,
            config_path=config_path,
            output_root=output_root / "tuning",
            n_events=n_events,
            thresholds=thresholds,
            seeds=tuning_seeds,
        )
        frozen_threshold = select_threshold(tuning_summary, settings)

    final_summary = pd.DataFrame()
    final_detection = pd.DataFrame()
    if phase in {"test", "both"}:
        final_summary, final_detection = run_final(
            base_config=base_config,
            detector=detector,
            config_path=config_path,
            output_root=output_root / "final",
            n_events=n_events,
            threshold=frozen_threshold,
            seeds=test_seeds,
        )

    write_outputs(
        output_root=output_root,
        settings=settings,
        tuning_summary=tuning_summary,
        tuning_detection=tuning_detection,
        final_summary=final_summary,
        final_detection=final_detection,
    )


def run_tuning(
    base_config: dict[str, Any],
    detector: str,
    config_path: Path,
    output_root: Path,
    n_events: int,
    thresholds: list[float],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every threshold/seed pair for MIND history-divergence tuning."""

    settings = detector_settings(detector)
    threshold_column = str(settings["threshold_column"])
    name_prefix = str(settings["name_prefix"])
    parts = empty_table_parts(TABLE_FILES)
    for threshold in thresholds:
        for seed in seeds:
            run_dir = output_root / threshold_label(threshold) / f"seed_{seed}"
            outputs = reuse_completed_run(
                run_dir=run_dir,
                required_files=(*TABLE_FILES.values(), "metadata.json", "behavior_shifts.csv"),
                config=build_mind_config(
                    base_config,
                    detector=detector,
                    name=f"{name_prefix}_tuning_{threshold_label(threshold)}_seed_{seed}",
                    seed=seed,
                    n_events=n_events,
                    output_dir=run_dir,
                    threshold=threshold,
                ),
                config_path=config_path,
                runner=run_experiment,
            )
            append_run_tables(
                parts,
                outputs["output_dir"],
                [("phase", "tuning"), (threshold_column, threshold), ("seed", seed)],
                TABLE_FILES,
            )
    tables = concat_table_parts(parts)
    return tables["summary"], tables["detection"]


def run_final(
    base_config: dict[str, Any],
    detector: str,
    config_path: Path,
    output_root: Path,
    n_events: int,
    threshold: float,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run frozen-threshold final MIND seeds."""

    settings = detector_settings(detector)
    threshold_column = str(settings["threshold_column"])
    name_prefix = str(settings["name_prefix"])
    parts = empty_table_parts(TABLE_FILES)
    for seed in seeds:
        run_dir = output_root / f"seed_{seed}"
        outputs = reuse_completed_run(
            run_dir=run_dir,
            required_files=(*TABLE_FILES.values(), "metadata.json", "behavior_shifts.csv"),
            config=build_mind_config(
                base_config,
                detector=detector,
                name=f"{name_prefix}_final_seed_{seed}",
                seed=seed,
                n_events=n_events,
                output_dir=run_dir,
                threshold=threshold,
            ),
            config_path=config_path,
            runner=run_experiment,
        )
        append_run_tables(
            parts,
            outputs["output_dir"],
            [("phase", "test"), (threshold_column, threshold), ("seed", seed)],
            TABLE_FILES,
        )
    tables = concat_table_parts(parts)
    return tables["summary"], tables["detection"]


def build_mind_config(
    base_config: dict[str, Any],
    detector: str,
    name: str,
    seed: int,
    n_events: int,
    output_dir: Path,
    threshold: float,
) -> dict[str, Any]:
    """Create one concrete MIND run config."""

    return build_protocol_run_config(
        base_config=base_config,
        name=name,
        seed=seed,
        n_events=n_events,
        output_dir=output_dir,
        extra_overrides={f"drift_detection.{detector}.threshold": float(threshold)},
        cap_detection_horizon=False,
    )


def write_outputs(
    output_root: Path,
    settings: dict[str, Any],
    tuning_summary: pd.DataFrame,
    tuning_detection: pd.DataFrame,
    final_summary: pd.DataFrame,
    final_detection: pd.DataFrame,
) -> None:
    """Save raw and aggregate protocol CSV tables."""

    threshold_column = str(settings["threshold_column"])
    write_protocol_phase(
        output_root,
        "tuning",
        {"summary": tuning_summary, "detection": tuning_detection},
        protocol_aggregations("tuning", threshold_column),
    )
    write_protocol_phase(
        output_root,
        "final",
        {"summary": final_summary, "detection": final_detection},
        protocol_aggregations("final"),
    )


def protocol_aggregations(
    phase: str,
    threshold_column: str | None = None,
) -> list[ProtocolAggregation]:
    """Return aggregate table definitions for one real-MIND phase."""

    prefix = ("phase",) + ((threshold_column,) if threshold_column else ())
    return [
        ProtocolAggregation(
            "summary",
            f"{phase}_summary_overall.csv",
            prefix + ("method",),
            SUMMARY_METRICS,
        ),
        ProtocolAggregation(
            "detection",
            f"{phase}_detection_overall.csv",
            prefix + ("method", "detector"),
            DETECTION_METRICS,
        ),
    ]


def detector_settings(detector: str) -> dict[str, Any]:
    """Return settings for a supported MIND history-divergence detector."""

    try:
        return DETECTOR_SETTINGS[detector]
    except KeyError as exc:
        raise ValueError(f"Unsupported detector: {detector}") from exc


if __name__ == "__main__":
    main()
