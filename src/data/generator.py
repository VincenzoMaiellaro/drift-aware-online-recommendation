"""Synthetic streaming data for controlled recommendation drift experiments.

The generator creates a controlled event stream. Each user has known pre-drift
and post-drift category preference distributions. Each event includes the true
drift point for measuring detector delay, recovery, and regret. The item catalog,
category labels, popularity prior, and user profile priors come from the local
MIND dataset so the benchmark is comparable to the real-data MIND pipeline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
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
class SyntheticStreamConfig:
    """All knobs that define one synthetic stream.

    The most important parameters are the MIND path, item/profile sampling
    limits, drift type, drift strength, observation noise, and random seed.
    Experiment runners build this object from `config_synthetic_mind_shaped.yaml`.
    """

    n_users: int = 60
    n_events: int = 12_000
    drift_fraction: float = 0.5
    drift_mode: str = "sudden"
    drift_strength: float = 0.75
    noise: float = 0.08
    gradual_width: int = 1_500
    user_drift_jitter: int = 450
    mind_path: str = "MINDsmall_train"
    category_level: str = "category"
    mind_item_limit: int = 0
    mind_profile_events: int = 0
    mind_smoothing: float = 0.05
    candidate_set_size: int = 0
    # Derived after loading MIND news metadata and kept for downstream methods.
    n_categories: int = 0
    seed: int = 42


@dataclass(frozen=True)
class _MINDCatalog:
    """MIND-derived catalog and behavior priors used by synthetic streams."""

    category_names: list[str]
    item_categories: np.ndarray
    item_sampling_weights: np.ndarray
    category_distribution: np.ndarray
    user_profiles: np.ndarray
    raw_n_items: int
    raw_n_users: int
    raw_n_behavior_rows: int


@dataclass(frozen=True)
class _MINDBehaviorStats:
    """Aggregated MIND behavior priors used to construct a catalog."""

    click_counts: np.ndarray
    category_distribution: np.ndarray
    user_profiles: np.ndarray
    raw_n_users: int
    raw_n_behavior_rows: int


class SyntheticDriftStream:
    """Generate user-item-category interactions with controlled drift.

    Each user has a pre-drift and post-drift category distribution. The drift can
    be sudden, gradual, or disabled for stable-preference controls. Drift points
    are jittered around a global event index so that the stream is not
    unrealistically synchronized.
    """

    def __init__(self, config: SyntheticStreamConfig) -> None:
        """Create deterministic stream state from the supplied configuration."""

        _validate_synthetic_config(config)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.category_level = normalise_mind_category_level(config.category_level)

        catalog = _load_mind_catalog(config, self.rng)
        self.config = replace(
            config,
            n_categories=len(catalog.category_names),
            category_level=self.category_level,
        )
        self.category_names = list(catalog.category_names)
        self.n_items = len(catalog.item_categories)
        self.item_categories = catalog.item_categories
        self.item_sampling_weights = catalog.item_sampling_weights
        self.background_category_distribution = catalog.category_distribution
        self.mind_user_profiles = catalog.user_profiles
        self.catalog_metadata: dict[str, int | str] = {
            "mind_path": str(config.mind_path),
            "raw_n_items": int(catalog.raw_n_items),
            "raw_n_users": int(catalog.raw_n_users),
            "raw_n_behavior_rows": int(catalog.raw_n_behavior_rows),
        }

        if self.config.n_categories < 2:
            raise ValueError("Synthetic streams need at least two categories for drift.")
        if self.n_items < 2:
            raise ValueError("Synthetic streams need at least two items.")

        self.items_by_category = {
            c: np.flatnonzero(self.item_categories == c)
            for c in range(self.config.n_categories)
        }
        empty_categories = [
            c for c, items in self.items_by_category.items() if len(items) == 0
        ]
        if empty_categories:
            raise ValueError(
                "Every synthetic category must have at least one item; "
                f"empty category IDs: {empty_categories}."
            )

        self.item_popularity = self._category_item_popularity(
            self.item_sampling_weights
        )
        self.all_item_ids = np.arange(self.n_items)
        self.pre_preferences, self.post_preferences = self._make_preferences()
        self.user_drift_points = self._make_user_drift_points()

    def __iter__(self) -> Iterator[InteractionEvent]:
        """Yield events in chronological order.

        At every timestamp the generator samples a user, computes that user's
        current hidden category distribution, adds configurable noise, samples a
        category, and finally samples an item from that category.
        """

        user_event_counts = np.zeros(self.config.n_users, dtype=int)
        user_drift_event_counts = np.zeros(self.config.n_users, dtype=int)

        for event_id in range(self.config.n_events):
            user_id = int(self.rng.integers(0, self.config.n_users))
            dist, progress = self.user_distribution(user_id, event_id)
            noisy_dist = self._with_noise(dist)
            category_id = int(self.rng.choice(self.config.n_categories, p=noisy_dist))
            item_id = int(
                self.rng.choice(
                    self.items_by_category[category_id],
                    p=self.item_popularity[category_id],
                )
            )
            candidate_item_ids = self._candidate_item_ids(item_id)
            drift_start, drift_point, drift_end = self.drift_window(user_id)
            user_event_index = int(user_event_counts[user_id])
            if drift_start >= 0 and event_id >= drift_start:
                user_events_since_drift_start = int(user_drift_event_counts[user_id])
                user_drift_event_counts[user_id] += 1
            else:
                user_events_since_drift_start = -1
            user_event_counts[user_id] += 1
            yield InteractionEvent(
                event_id=event_id,
                user_id=user_id,
                item_id=item_id,
                category_id=category_id,
                timestamp=event_id,
                drift_progress=float(progress),
                user_drift_point=drift_point,
                drift_start=drift_start,
                drift_end=drift_end,
                drift_phase=self.drift_phase(event_id, drift_start, drift_end),
                is_after_drift=drift_end >= 0 and event_id >= drift_end,
                true_user_distribution=tuple(float(x) for x in dist),
                behavior_user_distribution=tuple(float(x) for x in noisy_dist),
                user_event_index=user_event_index,
                user_events_since_drift_start=user_events_since_drift_start,
                candidate_item_ids=candidate_item_ids,
            )

    def user_distribution(
        self, user_id: int, timestamp: int
    ) -> tuple[np.ndarray, float]:
        """Return the hidden preference distribution at a timestamp."""

        pre = self.pre_preferences[user_id]
        post = self.post_preferences[user_id]
        drift_point = self.user_drift_points[user_id]

        if self.config.drift_mode == "none":
            progress = 0.0
        elif self.config.drift_mode == "sudden":
            progress = 1.0 if timestamp >= drift_point else 0.0
        elif self.config.drift_mode == "gradual":
            half_width = max(1, self.config.gradual_width // 2)
            start = drift_point - half_width
            progress = (timestamp - start) / max(1, self.config.gradual_width)
            progress = float(np.clip(progress, 0.0, 1.0))
        else:
            raise ValueError(
                f"Unknown drift_mode={self.config.drift_mode!r}; "
                "expected 'sudden', 'gradual', or 'none'."
            )

        dist = (1.0 - progress) * pre + progress * post
        return _normalise_distribution(dist), progress

    def drift_window(self, user_id: int) -> tuple[int, int, int]:
        """Return the drift start, midpoint, and completed-post-drift timestamp."""

        midpoint = int(self.user_drift_points[user_id])
        if self.config.drift_mode == "none":
            return -1, -1, -1
        if self.config.drift_mode == "sudden":
            return midpoint, midpoint, midpoint
        if self.config.drift_mode == "gradual":
            width = max(1, int(self.config.gradual_width))
            half_width = max(1, width // 2)
            start = midpoint - half_width
            end = start + width
            return int(start), midpoint, int(end)
        raise ValueError(
            f"Unknown drift_mode={self.config.drift_mode!r}; "
            "expected 'sudden', 'gradual', or 'none'."
        )

    def drift_phase(self, timestamp: int, drift_start: int, drift_end: int) -> str:
        """Classify a timestamp as pre-drift, transition, or post-drift."""

        if self.config.drift_mode == "none":
            return "stable"
        if timestamp < drift_start:
            return "pre"
        if self.config.drift_mode == "gradual" and timestamp < drift_end:
            return "transition"
        return "post"

    def true_top_categories(
        self, user_id: int, timestamp: int, k: int
    ) -> list[int]:
        """Categories preferred by the hidden oracle at this point in time."""

        dist, _ = self.user_distribution(user_id, timestamp)
        top = np.argsort(dist)[::-1][:k]
        return [int(x) for x in top]

    def metadata(self) -> dict[str, object]:
        """Return serialisable stream metadata saved with experiment results."""

        return {
            "source": "synthetic",
            "catalog_source": "mind",
            "n_users": self.config.n_users,
            "n_categories": self.config.n_categories,
            "n_items": self.n_items,
            "n_events": self.config.n_events,
            "drift_mode": self.config.drift_mode,
            "drift_strength": self.config.drift_strength,
            "noise": self.config.noise,
            "category_level": self.category_level,
            "category_names": list(self.category_names),
            "candidate_set_size": self.config.candidate_set_size,
            "has_ground_truth_drift": self.config.drift_mode != "none",
            "user_drift_points": {
                str(i): int(v) for i, v in enumerate(self.user_drift_points)
            },
            "user_drift_starts": {
                str(i): int(self.drift_window(i)[0]) for i in range(self.config.n_users)
            },
            "user_drift_ends": {
                str(i): int(self.drift_window(i)[2]) for i in range(self.config.n_users)
            },
            "item_categories": [int(x) for x in self.item_categories],
            "item_count_by_category": {
                str(category): int(len(items))
                for category, items in self.items_by_category.items()
            },
            **self.catalog_metadata,
        }

    def _make_preferences(self) -> tuple[np.ndarray, np.ndarray]:
        """Create each user's pre-drift and post-drift category preferences."""

        cfg = self.config
        pre = np.zeros((cfg.n_users, cfg.n_categories), dtype=float)
        post = np.zeros_like(pre)

        for user in range(cfg.n_users):
            base = self._sample_mind_user_profile()
            target_category = self._sample_shift_target(base)
            target_preference = self._category_focus(target_category)
            pre[user] = base
            post[user] = target_preference

        return pre, post

    def _sample_mind_user_profile(self) -> np.ndarray:
        """Sample and slightly jitter one empirical MIND user profile."""

        if len(self.mind_user_profiles) > 0:
            profile = self.mind_user_profiles[
                int(self.rng.integers(0, len(self.mind_user_profiles)))
            ]
        else:
            profile = self.background_category_distribution
        alpha = np.maximum(profile, 1e-6) * 80.0 + self.config.mind_smoothing
        jitter = self.rng.dirichlet(alpha)
        return _normalise_distribution(0.85 * profile + 0.15 * jitter)

    def _sample_shift_target(self, pre_distribution: np.ndarray) -> int:
        """Choose a plausible post-drift category from the MIND global mix."""

        weights = self.background_category_distribution * np.square(
            1.0 - pre_distribution
        )
        main_category = int(np.argmax(pre_distribution))
        weights = _normalise_excluding_category(weights, main_category)
        return int(self.rng.choice(self.config.n_categories, p=weights))

    def _category_focus(self, main_category: int) -> np.ndarray:
        """Build a MIND-shaped distribution centered on one target category."""

        target_mass = float(np.clip(self.config.drift_strength, 0.0, 1.0))
        tail = _normalise_excluding_category(
            self.background_category_distribution,
            main_category,
        )
        preference = (1.0 - target_mass) * tail
        preference[main_category] = target_mass
        return _normalise_distribution(preference)

    def _make_user_drift_points(self) -> np.ndarray:
        """Assign a drift timestamp to each user around the global drift center."""

        cfg = self.config
        if cfg.drift_mode == "none":
            return np.full(cfg.n_users, -1, dtype=int)
        center = int(cfg.n_events * cfg.drift_fraction)
        low, high = self._drift_point_bounds()
        if cfg.user_drift_jitter <= 0:
            return np.full(cfg.n_users, np.clip(center, low, high), dtype=int)
        jitter = self.rng.integers(
            -cfg.user_drift_jitter, cfg.user_drift_jitter + 1, size=cfg.n_users
        )
        return np.clip(center + jitter, low, high).astype(int)

    def _drift_point_bounds(self) -> tuple[int, int]:
        """Return safe midpoint bounds for the configured drift mode."""

        cfg = self.config
        if cfg.drift_mode == "gradual":
            width = max(1, int(cfg.gradual_width))
            half_width = max(1, width // 2)
            low = max(1, half_width)
            high = cfg.n_events - 1 - width + half_width
        else:
            low = 1
            high = cfg.n_events - 1
        return low, max(low, high)

    def _category_item_popularity(
        self,
        item_weights: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Convert global item weights into per-category sampling weights."""

        popularity: dict[int, np.ndarray] = {}
        for category, items in self.items_by_category.items():
            weights = item_weights[items]
            popularity[category] = _normalise_distribution(weights)
        return popularity

    def _with_noise(self, dist: np.ndarray) -> np.ndarray:
        """Mix a preference distribution with background random behavior."""

        noise = float(np.clip(self.config.noise, 0.0, 1.0))
        return _normalise_distribution(
            (1.0 - noise) * dist + noise * self.background_category_distribution
        )

    def _candidate_item_ids(self, item_id: int) -> tuple[int, ...] | None:
        """Return a MIND-like impression set containing the clicked item."""

        candidate_size = int(self.config.candidate_set_size)
        if candidate_size <= 1:
            return None
        candidate_size = min(candidate_size, self.n_items)
        negative_count = candidate_size - 1
        weights = self.item_sampling_weights.copy()
        weights[item_id] = 0.0
        weights = _normalise_distribution(weights)
        negatives = self.rng.choice(
            self.all_item_ids,
            size=negative_count,
            replace=False,
            p=weights,
        )
        candidates = np.concatenate(([item_id], negatives))
        self.rng.shuffle(candidates)
        return tuple(int(x) for x in candidates)


def _load_mind_catalog(
    config: SyntheticStreamConfig,
    rng: np.random.Generator,
) -> _MINDCatalog:
    """Load MIND category labels, item popularity, and user profile priors."""

    path = Path(config.mind_path)
    behavior_path = path / "behaviors.tsv"
    news_metadata = load_mind_news_metadata(path, config.category_level)
    raw_categories = news_metadata.item_categories
    behavior_stats = _load_mind_behavior_stats(
        behavior_path,
        news_metadata.news_id_to_item_id,
        raw_categories,
        len(news_metadata.category_names),
        config,
    )
    selected_items = _select_mind_items(
        raw_categories,
        behavior_stats.click_counts,
        int(config.mind_item_limit),
        float(config.mind_smoothing),
        rng,
    )
    item_categories = raw_categories[selected_items]
    selected_click_counts = behavior_stats.click_counts[selected_items]
    item_sampling_weights = _normalise_distribution(
        selected_click_counts + float(config.mind_smoothing)
    )

    return _MINDCatalog(
        category_names=news_metadata.category_names,
        item_categories=item_categories,
        item_sampling_weights=item_sampling_weights,
        category_distribution=behavior_stats.category_distribution,
        user_profiles=behavior_stats.user_profiles,
        raw_n_items=len(news_metadata.news_ids),
        raw_n_users=behavior_stats.raw_n_users,
        raw_n_behavior_rows=behavior_stats.raw_n_behavior_rows,
    )


def _load_mind_behavior_stats(
    behavior_path: Path,
    news_id_to_item_id: dict[str, int],
    raw_item_categories: np.ndarray,
    n_categories: int,
    config: SyntheticStreamConfig,
) -> _MINDBehaviorStats:
    """Read MIND behavior rows as priors for synthetic generation."""

    click_counts = np.zeros(len(raw_item_categories), dtype=float)
    category_counts = np.full(n_categories, float(config.mind_smoothing), dtype=float)
    user_counts: dict[str, np.ndarray] = {}
    behavior_rows = 0

    if not behavior_path.exists():
        raise FileNotFoundError(f"MIND behaviors file not found: {behavior_path}")

    with behavior_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if config.mind_profile_events > 0 and (
                behavior_rows >= config.mind_profile_events
            ):
                break
            if len(row) < 5:
                raise ValueError(
                    f"Malformed behaviors.tsv row {row_number}: expected 5 columns."
                )
            _, user_key, _, history, impressions = row[:5]
            behavior_rows += 1
            user_category_counts = user_counts.setdefault(
                user_key,
                np.zeros(n_categories, dtype=float),
            )
            history_categories = map_mind_history_categories(
                history,
                news_id_to_item_id,
                raw_item_categories,
            )
            for category in history_categories:
                user_category_counts[category] += 1.0
            _, clicked_items = parse_mind_impressions(
                impressions,
                news_id_to_item_id,
            )
            for _, item_id in clicked_items:
                category = int(raw_item_categories[item_id])
                click_counts[item_id] += 1.0
                category_counts[category] += 1.0
                user_category_counts[category] += 1.0

    if float(category_counts.sum()) <= n_categories * float(config.mind_smoothing):
        news_category_counts = np.bincount(
            raw_item_categories,
            minlength=n_categories,
        ).astype(float)
        category_counts += news_category_counts

    category_distribution = _normalise_distribution(category_counts)
    user_profiles = _build_user_profiles(
        user_counts,
        category_distribution,
        float(config.mind_smoothing),
    )
    return _MINDBehaviorStats(
        click_counts=click_counts,
        category_distribution=category_distribution,
        user_profiles=user_profiles,
        raw_n_users=len(user_counts),
        raw_n_behavior_rows=behavior_rows,
    )


def _validate_synthetic_config(config: SyntheticStreamConfig) -> None:
    """Reject values that cannot produce valid probability distributions."""

    if config.n_users <= 0:
        raise ValueError("Synthetic n_users must be positive.")
    if not np.isfinite(config.mind_smoothing) or config.mind_smoothing < 0.0:
        raise ValueError("Synthetic mind_smoothing must be finite and non-negative.")


def _build_user_profiles(
    user_counts: dict[str, np.ndarray],
    category_distribution: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    """Convert raw MIND user counts into category distributions."""

    if not user_counts:
        return np.empty((0, len(category_distribution)), dtype=float)
    profiles = []
    for counts in user_counts.values():
        smoothed = counts + smoothing * category_distribution
        profiles.append(_normalise_distribution(smoothed))
    return np.asarray(profiles, dtype=float)


def _select_mind_items(
    raw_item_categories: np.ndarray,
    click_counts: np.ndarray,
    item_limit: int,
    smoothing: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a compact MIND item catalog while preserving category coverage."""

    n_items = len(raw_item_categories)
    if item_limit <= 0 or item_limit >= n_items:
        return np.arange(n_items, dtype=int)

    n_categories = int(raw_item_categories.max()) + 1
    if item_limit < n_categories:
        raise ValueError(
            "mind_item_limit must be at least the number of MIND categories "
            f"({n_categories})."
        )

    category_counts = np.bincount(raw_item_categories, minlength=n_categories)
    allocation = np.zeros(n_categories, dtype=int)
    available_categories = np.flatnonzero(category_counts > 0)
    allocation[available_categories] = 1
    remaining = item_limit - int(allocation.sum())

    if remaining > 0:
        capacity = category_counts - allocation
        proportions = category_counts / max(1, int(category_counts.sum()))
        ideal_extra = proportions * remaining
        extra = np.minimum(np.floor(ideal_extra).astype(int), capacity)
        allocation += extra
        while int(allocation.sum()) < item_limit:
            capacity = category_counts - allocation
            candidates = np.flatnonzero(capacity > 0)
            if len(candidates) == 0:
                break
            remainders = ideal_extra[candidates] - np.floor(ideal_extra[candidates])
            category = int(candidates[int(np.argmax(remainders))])
            allocation[category] += 1

    selected: list[int] = []
    for category, count in enumerate(allocation):
        if count <= 0:
            continue
        category_items = np.flatnonzero(raw_item_categories == category)
        if count >= len(category_items):
            selected.extend(int(x) for x in category_items)
            continue
        weights = _normalise_distribution(click_counts[category_items] + smoothing)
        sample = rng.choice(category_items, size=int(count), replace=False, p=weights)
        selected.extend(int(x) for x in sample)
    return np.asarray(sorted(selected), dtype=int)


def _normalise_distribution(values: np.ndarray) -> np.ndarray:
    """Convert non-negative values into a probability distribution."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot normalise an empty distribution.")
    total = float(values.sum())
    if total <= 0.0:
        return np.full(values.shape, 1.0 / len(values), dtype=float)
    return values / total


def _normalise_excluding_category(
    values: np.ndarray,
    excluded_category: int,
) -> np.ndarray:
    """Normalise values after removing one category, with a nonzero fallback."""

    weights = np.asarray(values, dtype=float).copy()
    weights[int(excluded_category)] = 0.0
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights, dtype=float)
        weights[int(excluded_category)] = 0.0
    return _normalise_distribution(weights)
