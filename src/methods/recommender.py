"""Online recommendation models used in the streaming experiments.

The experiment runner treats every recommender through the same minimal API:
`recommend(user_id, k)` before seeing the current event and `update(event)` after
evaluation. This file contains the MF backbone and two baselines.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.core.events import InteractionEvent


@dataclass(frozen=True)
class MFConfig:
    """Hyperparameters for online matrix factorization."""

    embedding_dim: int = 16
    learning_rate: float = 0.035
    regularization: float = 0.01
    negative_samples: int = 4
    init_std: float = 0.08
    category_weight: float = 0.18


@dataclass(frozen=True)
class Recommendation:
    """Ranked recommendation output returned by each model."""

    item_ids: list[int]
    scores: list[float]


class OnlineMatrixFactorization:
    """Incremental implicit-feedback matrix factorization with BPR updates."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        item_categories: Iterable[int],
        config: MFConfig,
        seed: int = 42,
        name: str = "online_mf",
    ) -> None:
        """Initialise user/item embeddings, biases, and category counters."""

        self.n_users = n_users
        self.n_items = n_items
        self.item_categories = np.asarray(list(item_categories), dtype=int)
        self.n_categories = int(self.item_categories.max()) + 1
        self.config = config
        self.name = name
        self.rng = np.random.default_rng(seed)

        self.user_factors = self.rng.normal(
            0.0, config.init_std, size=(n_users, config.embedding_dim)
        )
        self.item_factors = self.rng.normal(
            0.0, config.init_std, size=(n_items, config.embedding_dim)
        )
        self.item_bias = np.zeros(n_items, dtype=float)
        self.user_category_counts = np.full(
            (n_users, self.n_categories), 1e-3, dtype=float
        )
        self.seen_items: list[set[int]] = [set() for _ in range(n_users)]

    def recommend(
        self,
        user_id: int,
        k: int,
        exclude_seen: bool = False,
        candidate_items: np.ndarray | None = None,
    ) -> Recommendation:
        """Score candidate items and return the top-K ranking for one user."""

        candidate_items = _candidate_array(candidate_items, self.n_items)
        if exclude_seen:
            candidate_items = self._filter_seen_items(user_id, candidate_items)

        scores = self.score_items(user_id, candidate_items)
        top_idx = _top_k_indices(scores, k)
        top_items = candidate_items[top_idx]
        return Recommendation(
            item_ids=[int(x) for x in top_items],
            scores=[float(x) for x in scores[top_idx]],
        )

    def score_items(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Compute MF scores plus a small learned category-preference signal."""

        category_scores = self._category_distribution(user_id)[
            self.item_categories[item_ids]
        ]
        return (
            self.item_bias[item_ids]
            + self.item_factors[item_ids] @ self.user_factors[user_id]
            + self.config.category_weight * category_scores
        )

    def update(self, event: InteractionEvent, weight: float = 1.0) -> None:
        """Train on one implicit positive interaction with sampled negatives."""

        user = event.user_id
        pos_item = event.item_id
        lr = self.config.learning_rate * weight
        reg = self.config.regularization

        if self.n_items > 1:
            for _ in range(self.config.negative_samples):
                neg_item = self._sample_negative(user, pos_item)
                self._bpr_step(user, pos_item, neg_item, lr, reg)

        self.user_category_counts[user, event.category_id] += weight
        self.seen_items[user].add(pos_item)

    def category_distribution(self, user_id: int) -> np.ndarray:
        """Expose the model's current category preference estimate for a user."""

        return self._category_distribution(user_id).copy()

    def _filter_seen_items(
        self,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        """Remove items already consumed by a user, falling back to all items."""

        if not self.seen_items[user_id]:
            return candidate_items
        mask = np.array(
            [item not in self.seen_items[user_id] for item in candidate_items]
        )
        filtered = candidate_items[mask]
        if len(filtered) == 0:
            return np.arange(self.n_items)
        return filtered

    def _bpr_step(
        self, user: int, pos_item: int, neg_item: int, lr: float, reg: float
    ) -> None:
        """Apply one Bayesian Personalized Ranking gradient update."""

        user_vec = self.user_factors[user].copy()
        pos_vec = self.item_factors[pos_item].copy()
        neg_vec = self.item_factors[neg_item].copy()

        x_uij = (
            self.item_bias[pos_item]
            - self.item_bias[neg_item]
            + user_vec @ (pos_vec - neg_vec)
        )
        gradient = 1.0 / (1.0 + np.exp(np.clip(x_uij, -35.0, 35.0)))

        self.user_factors[user] += lr * (
            gradient * (pos_vec - neg_vec) - reg * user_vec
        )
        self.item_factors[pos_item] += lr * (gradient * user_vec - reg * pos_vec)
        self.item_factors[neg_item] += lr * (-gradient * user_vec - reg * neg_vec)
        self.item_bias[pos_item] += lr * (gradient - reg * self.item_bias[pos_item])
        self.item_bias[neg_item] += lr * (-gradient - reg * self.item_bias[neg_item])

    def _sample_negative(self, user_id: int, pos_item: int) -> int:
        """Sample an item the user has not interacted with as negative feedback."""

        if self.n_items <= 1:
            raise ValueError("Cannot sample a negative item from a catalog of size 1.")
        for _ in range(50):
            candidate = int(self.rng.integers(0, self.n_items))
            if candidate != pos_item and candidate not in self.seen_items[user_id]:
                return candidate
        excluded = self.seen_items[user_id] | {pos_item}
        if len(excluded) < self.n_items:
            excluded_array = np.fromiter(excluded, dtype=int)
            unseen = np.setdiff1d(
                np.arange(self.n_items), excluded_array, assume_unique=False
            )
            return int(self.rng.choice(unseen))
        candidate = int(self.rng.integers(0, self.n_items - 1))
        if candidate >= pos_item:
            candidate += 1
        return candidate

    def _category_distribution(self, user_id: int) -> np.ndarray:
        """Normalise user category counts into a probability vector."""

        counts = self.user_category_counts[user_id]
        total = float(counts.sum())
        if total <= 0.0:
            return np.full(self.n_categories, 1.0 / self.n_categories)
        return counts / total


class TimeDecayMatrixFactorization(OnlineMatrixFactorization):
    """Online MF baseline that emphasizes recent user behavior."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        item_categories: Iterable[int],
        config: MFConfig,
        decay_rate: float = 0.015,
        drift_boost: float = 1.0,
        max_elapsed: int = 200,
        user_factor_decay_floor: float = 0.985,
        seed: int = 42,
    ) -> None:
        """Initialise the MF baseline with extra recency parameters."""

        super().__init__(
            n_users=n_users,
            n_items=n_items,
            item_categories=item_categories,
            config=config,
            seed=seed,
            name="time_decay_mf",
        )
        self.decay_rate = decay_rate
        self.drift_boost = drift_boost
        self.max_elapsed = max_elapsed
        self.user_factor_decay_floor = user_factor_decay_floor
        self.last_update = np.zeros(n_users, dtype=int)

    def update(self, event: InteractionEvent, weight: float = 1.0) -> None:
        """Apply decay to old user state before the normal MF update."""

        user = event.user_id
        elapsed = max(1, event.timestamp - int(self.last_update[user]))
        decay = float(np.exp(-self.decay_rate * min(elapsed, self.max_elapsed)))
        self.user_category_counts[user] *= decay
        factor_floor = float(np.clip(self.user_factor_decay_floor, 0.0, 1.0))
        self.user_factors[user] *= factor_floor + (1.0 - factor_floor) * decay
        recency_weight = weight * (1.0 + self.drift_boost * (1.0 - decay))
        super().update(event, weight=recency_weight)
        self.last_update[user] = event.timestamp


class _PopularityRecommender:
    """Shared scoring and category-distribution logic for popularity baselines."""

    def __init__(
        self,
        n_items: int,
        item_categories: Iterable[int],
        name: str,
    ) -> None:
        """Initialise cumulative counters shared by popularity recommenders."""

        self.n_items = n_items
        self.item_categories = np.asarray(list(item_categories), dtype=int)
        self.n_categories = int(self.item_categories.max()) + 1
        self.name = name
        self.item_counts: Counter[int] = Counter()
        self.category_counts: Counter[int] = Counter()

    def recommend(
        self,
        user_id: int,
        k: int,
        exclude_seen: bool = False,
        candidate_items: np.ndarray | None = None,
    ) -> Recommendation:
        """Rank items by item popularity plus category popularity."""

        del user_id, exclude_seen
        candidate_items = _candidate_array(candidate_items, self.n_items)

        scores = np.array(
            [
                self.item_counts[int(item)]
                + 0.1 * self.category_counts[int(self.item_categories[item])]
                for item in candidate_items
            ],
            dtype=float,
        )
        top_idx = _top_k_indices(scores, k)
        return Recommendation(
            item_ids=[int(x) for x in candidate_items[top_idx]],
            scores=[float(x) for x in scores[top_idx]],
        )

    def category_distribution(self, user_id: int) -> np.ndarray:
        """Return the current global category distribution."""

        del user_id
        counts = np.array(
            [self.category_counts[c] for c in range(self.n_categories)],
            dtype=float,
        )
        if counts.sum() <= 0.0:
            return np.full(self.n_categories, 1.0 / self.n_categories)
        return counts / counts.sum()


class RecentPopularityRecommender(_PopularityRecommender):
    """Recent item-popularity baseline over a sliding interaction window."""

    def __init__(
        self,
        n_items: int,
        item_categories: Iterable[int],
        window_size: int = 600,
        name: str = "recent_popularity",
    ) -> None:
        """Initialise sliding-window counters for recent popularity."""

        super().__init__(n_items, item_categories, name)
        self.window_size = window_size
        self.events: deque[tuple[int, int, int]] = deque()

    def update(self, event: InteractionEvent, weight: float = 1.0) -> None:
        """Insert the event into the sliding window and evict old counts."""

        del weight
        record = (event.timestamp, event.item_id, event.category_id)
        self.events.append(record)
        self.item_counts[event.item_id] += 1
        self.category_counts[event.category_id] += 1
        while len(self.events) > self.window_size:
            _, item_id, category_id = self.events.popleft()
            self.item_counts[item_id] -= 1
            self.category_counts[category_id] -= 1
            if self.item_counts[item_id] <= 0:
                del self.item_counts[item_id]
            if self.category_counts[category_id] <= 0:
                del self.category_counts[category_id]


class GlobalPopularityRecommender(_PopularityRecommender):
    """Cumulative item-popularity baseline over all past interactions."""

    def __init__(
        self,
        n_items: int,
        item_categories: Iterable[int],
        name: str = "global_popularity",
    ) -> None:
        """Initialise cumulative item/category counters."""

        super().__init__(n_items, item_categories, name)

    def update(self, event: InteractionEvent, weight: float = 1.0) -> None:
        """Insert the event into cumulative popularity counters."""

        del weight
        self.item_counts[event.item_id] += 1
        self.category_counts[event.category_id] += 1


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the K largest scores in descending order."""

    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    partial = np.argpartition(scores, -k)[-k:]
    return partial[np.argsort(scores[partial])[::-1]]


def _candidate_array(
    candidate_items: np.ndarray | None,
    n_items: int,
) -> np.ndarray:
    """Return explicit candidate item IDs for scoring."""

    if candidate_items is None:
        return np.arange(n_items)
    return np.asarray(candidate_items, dtype=int)
