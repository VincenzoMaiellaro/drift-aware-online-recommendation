"""Shared event contract for online recommendation streams."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InteractionEvent:
    """Single interaction consumed by recommenders, detectors, and metrics."""

    event_id: int
    user_id: int
    item_id: int
    category_id: int
    timestamp: int
    drift_progress: float
    user_drift_point: int
    drift_start: int
    drift_end: int
    drift_phase: str
    is_after_drift: bool
    true_user_distribution: tuple[float, ...]
    behavior_user_distribution: tuple[float, ...]
    user_event_index: int
    user_events_since_drift_start: int
    candidate_item_ids: tuple[int, ...] | None = None
    reference_user_distribution: tuple[float, ...] | None = None
    source_event_id: str | None = None
    source_timestamp: str | None = None
    source_history_size: int = 0
