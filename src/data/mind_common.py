"""Shared helpers for reading MIND news metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_CATEGORY_COLUMN_BY_LEVEL = {"category": 1}


@dataclass(frozen=True)
class MINDNewsMetadata:
    """Compact item/category mapping loaded from a MIND split."""

    news_ids: list[str]
    news_id_to_item_id: dict[str, int]
    category_names: list[str]
    item_categories: np.ndarray


def load_mind_news_metadata(path: str | Path, category_level: str) -> MINDNewsMetadata:
    """Read `news.tsv` and build compact item/category IDs."""

    dataset_path = Path(path)
    news_path = dataset_path / "news.tsv"
    if not news_path.exists():
        raise FileNotFoundError(f"MIND news file not found: {news_path}")

    category_index = mind_category_column_index(category_level)
    news_ids: list[str] = []
    news_id_to_item_id: dict[str, int] = {}
    category_names: list[str] = []
    category_to_id: dict[str, int] = {}
    item_categories: list[int] = []

    with news_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if len(row) <= category_index:
                raise ValueError(
                    f"Malformed news.tsv row {row_number}: expected at least "
                    f"{category_index + 1} columns."
                )
            news_id = row[0].strip()
            if not news_id:
                continue
            if news_id in news_id_to_item_id:
                raise ValueError(
                    f"Duplicate news ID {news_id!r} in news.tsv row {row_number}."
                )
            category_name = row[category_index].strip().lower() or "unknown"
            category_id = category_to_id.get(category_name)
            if category_id is None:
                category_id = len(category_names)
                category_to_id[category_name] = category_id
                category_names.append(category_name)
            news_id_to_item_id[news_id] = len(news_ids)
            news_ids.append(news_id)
            item_categories.append(category_id)

    if not news_ids:
        raise ValueError(f"No news items found in MIND news file: {news_path}")

    return MINDNewsMetadata(
        news_ids=news_ids,
        news_id_to_item_id=news_id_to_item_id,
        category_names=category_names,
        item_categories=np.asarray(item_categories, dtype=int),
    )


def mind_category_column_index(category_level: str) -> int:
    """Return the supported `news.tsv` category column."""

    level = normalise_mind_category_level(category_level)
    return _CATEGORY_COLUMN_BY_LEVEL[level]


def normalise_mind_category_level(category_level: str) -> str:
    """Validate and normalise the MIND category granularity."""

    level = str(category_level).strip().lower()
    if level in _CATEGORY_COLUMN_BY_LEVEL:
        return level
    raise ValueError(
        f"Unknown MIND category_level={category_level!r}; "
        "expected 'category'."
    )


def map_mind_history_categories(
    history: str,
    news_id_to_item_id: dict[str, int],
    item_categories: np.ndarray,
) -> tuple[int, ...]:
    """Map known news IDs in a MIND history to compact category IDs."""

    categories: list[int] = []
    for news_id in history.split():
        item_id = news_id_to_item_id.get(news_id)
        if item_id is not None:
            categories.append(int(item_categories[item_id]))
    return tuple(categories)


def parse_mind_impressions(
    impressions: str,
    news_id_to_item_id: dict[str, int],
) -> tuple[tuple[int, ...], list[tuple[int, int]]]:
    """Return unique known candidates and clicked impression positions."""

    candidate_items: list[int] = []
    clicked_items: list[tuple[int, int]] = []
    seen_candidates: set[int] = set()
    for impression_index, token in enumerate(impressions.split()):
        if "-" not in token:
            continue
        news_id, label = token.rsplit("-", 1)
        item_id = news_id_to_item_id.get(news_id)
        if item_id is None:
            continue
        if item_id not in seen_candidates:
            candidate_items.append(item_id)
            seen_candidates.add(item_id)
        if label == "1":
            clicked_items.append((impression_index, item_id))
    return tuple(candidate_items), clicked_items
