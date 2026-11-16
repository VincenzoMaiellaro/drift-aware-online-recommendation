"""MIND stream adapter for real-data behavior-shift experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from src.core.events import InteractionEvent
from src.data.mind_common import (
    load_mind_news_metadata,
    map_mind_history_categories,
    normalise_mind_category_level,
    parse_mind_impressions,
)


@dataclass(frozen=True)
class MINDStreamConfig:
    """Configuration for reading a local MIND split."""

    path: str = "MINDsmall_train"
    n_events: int = 0
    category_level: str = "category"
    smoothing: float = 0.05
    n_users: int = 0
    n_categories: int = 0


@dataclass(frozen=True)
class MINDRecord:
    """Parsed clicked impression with its behavior-row context."""

    sort_time: datetime
    behavior_row: int
    impression_index: int
    behavior_id: str
    user_key: str
    item_id: int
    category_id: int
    candidate_item_ids: tuple[int, ...]
    history_category_ids: tuple[int, ...]
    history_distribution: tuple[float, ...]
    source_timestamp: str


class MINDBehaviorStream:
    """Convert MIND behaviors/news TSV files into online click events."""

    def __init__(self, config: MINDStreamConfig) -> None:
        """Load news metadata and clicked impressions from a local split."""

        if not np.isfinite(config.smoothing) or config.smoothing <= 0.0:
            raise ValueError("MIND smoothing must be a positive finite number.")

        self.path = Path(config.path)
        self.category_level = normalise_mind_category_level(config.category_level)
        self.smoothing = float(config.smoothing)
        news_metadata = load_mind_news_metadata(self.path, self.category_level)
        self.news_ids = news_metadata.news_ids
        self.news_id_to_item_id = news_metadata.news_id_to_item_id
        self.category_names = news_metadata.category_names
        self.item_categories = news_metadata.item_categories
        self.n_items = len(self.news_ids)
        self.records = self._load_records(config.n_events)
        self.user_id_to_index = self._build_user_index(self.records)
        self.user_keys = sorted(
            self.user_id_to_index,
            key=self.user_id_to_index.__getitem__,
        )
        self.config = MINDStreamConfig(
            path=str(self.path),
            n_events=len(self.records),
            category_level=self.category_level,
            smoothing=self.smoothing,
            n_users=len(self.user_id_to_index),
            n_categories=len(self.category_names),
        )

    def __iter__(self) -> Iterator[InteractionEvent]:
        """Yield clicked impression events in chronological order."""

        user_event_counts = np.zeros(self.config.n_users, dtype=int)
        for event_id, record in enumerate(self.records):
            user_id = self.user_id_to_index[record.user_key]
            user_event_index = int(user_event_counts[user_id])
            user_event_counts[user_id] += 1
            yield InteractionEvent(
                event_id=event_id,
                user_id=user_id,
                item_id=record.item_id,
                category_id=record.category_id,
                timestamp=event_id,
                drift_progress=0.0,
                user_drift_point=-1,
                drift_start=-1,
                drift_end=-1,
                drift_phase="observed",
                is_after_drift=False,
                true_user_distribution=(),
                behavior_user_distribution=record.history_distribution,
                user_event_index=user_event_index,
                user_events_since_drift_start=-1,
                candidate_item_ids=record.candidate_item_ids,
                reference_user_distribution=record.history_distribution,
                source_event_id=record.behavior_id,
                source_timestamp=record.source_timestamp,
                source_history_size=len(record.history_category_ids),
            )

    def metadata(self) -> dict[str, object]:
        """Return serialisable stream metadata saved with experiment results."""

        return {
            "source": "mind",
            "path": str(self.path),
            "n_users": self.config.n_users,
            "n_categories": self.config.n_categories,
            "n_items": self.n_items,
            "n_events": self.config.n_events,
            "category_level": self.config.category_level,
            "category_names": list(self.category_names),
            "has_ground_truth_drift": False,
            "user_drift_points": {
                str(i): -1 for i in range(self.config.n_users)
            },
            "user_drift_starts": {
                str(i): -1 for i in range(self.config.n_users)
            },
            "user_drift_ends": {
                str(i): -1 for i in range(self.config.n_users)
            },
            "item_categories": [int(x) for x in self.item_categories],
        }

    def _load_records(self, n_events: int) -> list[MINDRecord]:
        """Read clicked impressions from `behaviors.tsv`."""

        behavior_path = self.path / "behaviors.tsv"
        if not behavior_path.exists():
            raise FileNotFoundError(f"MIND behaviors file not found: {behavior_path}")
        records: list[MINDRecord] = []
        with behavior_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row_number, row in enumerate(reader, start=1):
                if len(row) < 5:
                    raise ValueError(
                        f"Malformed behaviors.tsv row {row_number}: expected 5 columns."
                    )
                behavior_id, user_key, timestamp, history, impressions = row[:5]
                history_categories = map_mind_history_categories(
                    history,
                    self.news_id_to_item_id,
                    self.item_categories,
                )
                history_distribution = self._history_distribution(history_categories)
                candidate_items, clicked = parse_mind_impressions(
                    impressions,
                    self.news_id_to_item_id,
                )
                if not candidate_items:
                    continue
                for impression_index, item_id in clicked:
                    records.append(
                        MINDRecord(
                            sort_time=_parse_mind_timestamp(timestamp),
                            behavior_row=row_number,
                            impression_index=impression_index,
                            behavior_id=behavior_id,
                            user_key=user_key,
                            item_id=item_id,
                            category_id=int(self.item_categories[item_id]),
                            candidate_item_ids=candidate_items,
                            history_category_ids=history_categories,
                            history_distribution=history_distribution,
                            source_timestamp=timestamp,
                        )
                    )

        records.sort(
            key=lambda record: (
                record.sort_time,
                record.behavior_row,
                record.impression_index,
            )
        )
        if n_events > 0:
            return records[:n_events]
        return records

    def _history_distribution(
        self,
        categories: tuple[int, ...],
    ) -> tuple[float, ...]:
        """Build the KL reference distribution from behavior history."""

        counts = np.full(len(self.category_names), self.smoothing, dtype=float)
        for category in categories:
            counts[int(category)] += 1.0
        distribution = counts / counts.sum()
        return tuple(float(x) for x in distribution)

    @staticmethod
    def _build_user_index(records: list[MINDRecord]) -> dict[str, int]:
        """Map raw MIND user IDs to compact row indices."""

        user_id_to_index: dict[str, int] = {}
        for record in records:
            if record.user_key not in user_id_to_index:
                user_id_to_index[record.user_key] = len(user_id_to_index)
        return user_id_to_index


def _parse_mind_timestamp(value: str) -> datetime:
    """Parse the timestamp format used by MIND behavior files."""

    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse MIND timestamp: {value!r}")
