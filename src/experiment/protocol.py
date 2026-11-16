"""Shared scenario definitions and config helpers for experiment protocols."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SCENARIOS: dict[str, dict[str, Any]] = {
    "sudden_strong": {
        "data.drift_mode": "sudden",
        "data.drift_strength": 0.75,
        "data.noise": 0.08,
    },
    "sudden_weak": {
        "data.drift_mode": "sudden",
        "data.drift_strength": 0.55,
        "data.noise": 0.10,
    },
    "gradual_strong": {
        "data.drift_mode": "gradual",
        "data.drift_strength": 0.75,
        "data.noise": 0.08,
        "data.gradual_width": 1200,
    },
    "noisy_sudden": {
        "data.drift_mode": "sudden",
        "data.drift_strength": 0.75,
        "data.noise": 0.18,
    },
    "stable_no_drift": {
        "data.drift_mode": "none",
        "data.drift_strength": 0.0,
        "data.noise": 0.08,
    },
}


@dataclass(frozen=True)
class ProtocolAggregation:
    """Describe one aggregate CSV produced from a protocol table."""

    table: str
    filename: str
    group_columns: tuple[str, ...]
    metrics: tuple[str, ...]


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested dictionary value using dot notation."""

    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def build_protocol_run_config(
    base_config: dict[str, Any],
    name: str,
    seed: int,
    n_events: int,
    output_dir: Path,
    scenario_overrides: dict[str, Any] | None = None,
    extra_overrides: dict[str, Any] | None = None,
    cap_detection_horizon: bool = True,
) -> dict[str, Any]:
    """Create one concrete config for a seeded protocol run."""

    config = deepcopy(base_config)
    config["experiment"]["name"] = name
    config["experiment"]["seed"] = int(seed)
    config["experiment"]["output_dir"] = str(output_dir)
    config["data"]["n_events"] = int(n_events)
    if cap_detection_horizon:
        config["evaluation"]["max_events_after_drift"] = min(
            int(config["evaluation"]["max_events_after_drift"]),
            max(500, n_events // 4),
        )
    for key, value in (scenario_overrides or {}).items():
        set_nested(config, key, value)
    for key, value in (extra_overrides or {}).items():
        set_nested(config, key, value)
    return config


def reuse_completed_run(
    run_dir: Path,
    required_files: Sequence[str],
    config: dict[str, Any],
    config_path: Path,
    runner: Callable[[dict[str, Any], Path], dict[str, Path]],
) -> dict[str, Path]:
    """Reuse completed outputs only when they match the requested run config."""

    saved_config_path = run_dir / "config_used.yaml"
    if saved_config_path.is_file() and all(
        (run_dir / filename).is_file() for filename in required_files
    ):
        try:
            saved_config = yaml.safe_load(saved_config_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, yaml.YAMLError):
            saved_config = None
        if saved_config == config:
            return {"output_dir": run_dir}
    return runner(config, config_path)


def empty_protocol_tables(table_names: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Return an empty table mapping for a protocol phase."""

    return {name: pd.DataFrame() for name in table_names}


def empty_table_parts(table_names: Iterable[str]) -> dict[str, list[pd.DataFrame]]:
    """Return mutable collectors for protocol tables."""

    return {name: [] for name in table_names}


def append_run_tables(
    parts: dict[str, list[pd.DataFrame]],
    output_dir: Path,
    annotations: list[tuple[str, Any]],
    table_files: Mapping[str, str],
) -> None:
    """Read one run's tables and append the identifying annotations."""

    for name, filename in table_files.items():
        path = output_dir / filename
        table = (
            read_summary_table(path, annotations)
            if name == "summary"
            else read_annotated_table(path, annotations)
        )
        parts[name].append(table)


def concat_table_parts(
    parts: Mapping[str, list[pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Concatenate collected protocol tables, preserving mapping order."""

    return {
        name: pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        for name, tables in parts.items()
    }


def write_protocol_phase(
    output_root: Path,
    phase: str,
    tables: Mapping[str, pd.DataFrame],
    aggregations: Sequence[ProtocolAggregation],
) -> None:
    """Write raw phase tables and their configured aggregate CSVs."""

    summary = tables.get("summary")
    if summary is None or summary.empty:
        return
    for name, table in tables.items():
        table.to_csv(output_root / f"{phase}_{name}_runs.csv", index=False)
    for spec in aggregations:
        aggregate_metrics(
            tables[spec.table],
            list(spec.group_columns),
            spec.metrics,
            include_count_ci=True,
        ).to_csv(output_root / spec.filename, index=False)


def read_annotated_table(
    path: Path,
    annotations: list[tuple[str, Any]],
) -> pd.DataFrame:
    """Read a CSV table and prepend run-identifying columns."""

    table = pd.read_csv(path)
    prepend_annotations(table, annotations)
    return table


def read_summary_table(path: Path, annotations: list[tuple[str, Any]]) -> pd.DataFrame:
    """Read a summary table and normalize legacy single-aggregation names."""

    table = read_annotated_table(path, annotations)
    aliases = {
        "hit": "ctr_at_k",
        "ndcg": "ndcg_at_k",
        "category_hit": "category_hit_at_k",
        "category_top1_hit": "category_top1_accuracy",
        "rerank_active": "rerank_active_rate",
        "cluster_rule_active": "cluster_rule_active_rate",
        "event_id": "n_events",
    }
    return table.rename(
        columns={
            source: target
            for source, target in aliases.items()
            if source in table.columns and target not in table.columns
        }
    )


def prepend_annotations(
    table: pd.DataFrame,
    annotations: list[tuple[str, Any]],
) -> None:
    """Prepend annotation columns in the caller-provided order."""

    for column, value in reversed(annotations):
        table.insert(0, column, value)


def aggregate_metrics(
    table: pd.DataFrame,
    group_columns: list[str],
    metrics: tuple[str, ...],
    include_count_ci: bool = False,
) -> pd.DataFrame:
    """Aggregate available numeric metrics for protocol reports."""

    available = [metric for metric in metrics if metric in table.columns]
    grouped = table.groupby(group_columns, dropna=False)[available]
    if include_count_ci:
        aggregated = grouped.agg(["count", "mean", "std"]).reset_index()
        aggregated.columns = [
            "_".join(str(part) for part in column if part)
            if isinstance(column, tuple)
            else str(column)
            for column in aggregated.columns
        ]
        for metric in available:
            count_col = f"{metric}_count"
            std_col = f"{metric}_std"
            if count_col not in aggregated or std_col not in aggregated:
                continue
            count = aggregated[count_col].astype(float).replace(0.0, pd.NA)
            aggregated[f"{metric}_ci95"] = (
                1.96 * aggregated[std_col] / count.pow(0.5)
            )
        return aggregated

    mean = grouped.mean(numeric_only=True).add_suffix("_mean")
    std = grouped.std(numeric_only=True).add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()


def select_threshold(tuning_summary: pd.DataFrame, settings: dict[str, Any]) -> float:
    """Choose a frozen detector threshold from tuning rows."""

    threshold_column = str(settings["threshold_column"])
    candidate_methods = tuple(settings["candidate_methods"])
    candidate_rows = tuning_summary[tuning_summary["method"].isin(candidate_methods)]
    if candidate_rows.empty:
        threshold_name = str(settings["threshold_name"])
        raise ValueError(f"No drift-aware {threshold_name} rows found in tuning output.")
    grouped = aggregate_metrics(
        candidate_rows,
        [threshold_column, "method"],
        ("category_top1_accuracy", "ndcg_at_k", "rerank_active_rate"),
    )
    if "rerank_active_rate_mean" in grouped:
        filtered = grouped[
            (grouped["rerank_active_rate_mean"] > 0.0)
            & (grouped["rerank_active_rate_mean"] < 0.95)
        ]
        if not filtered.empty:
            grouped = filtered
    if grouped.empty:
        grouped = aggregate_metrics(
            candidate_rows,
            [threshold_column, "method"],
            ("category_top1_accuracy", "ndcg_at_k"),
        )
    if grouped.empty:
        threshold_name = str(settings["threshold_name"])
        raise ValueError(
            f"No aggregate metrics available to select a {threshold_name} threshold."
        )
    sort_columns = [
        column
        for column in ("category_top1_accuracy_mean", "ndcg_at_k_mean")
        if column in grouped.columns
    ]
    if not sort_columns:
        threshold_name = str(settings["threshold_name"])
        raise ValueError(
            f"No ranking-quality metric columns available to select a "
            f"{threshold_name} threshold."
        )
    best = grouped.sort_values(
        sort_columns,
        ascending=False,
    ).iloc[0]
    return float(best[threshold_column])


def parse_ints(raw: str) -> list[int]:
    """Parse a comma-separated integer list."""

    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return [int(value) for value in values]


def parse_floats(raw: str) -> list[float]:
    """Parse a comma-separated float list."""

    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one threshold is required.")
    return [float(value) for value in values]


def threshold_label(value: float) -> str:
    """Return a filesystem-friendly threshold label."""

    return f"{value:g}".replace(".", "p")


def round_numeric(table: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    """Round numeric columns for readable Markdown."""

    rounded = table.copy()
    numeric_columns = rounded.select_dtypes(include="number").columns
    rounded[numeric_columns] = rounded[numeric_columns].round(digits)
    return rounded
