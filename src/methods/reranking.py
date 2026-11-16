"""Drift-aware reranking based on short-term user patterns.

The base recommender still produces the candidate list. This module temporarily
reorders that list after a user-level drift signal, biasing it toward categories
that dominate the user's recent behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.methods.pattern_extraction import PatternFeatures
from src.methods.recommender import Recommendation


@dataclass(frozen=True)
class RerankConfig:
    """Parameters controlling reranker duration and score mixing."""

    active_window: int = 350
    rerank_alpha: float = 0.65
    preference_weight: float = 0.65
    transition_weight: float = 0.35


class DriftAwareReranker:
    """Temporarily bias recommendations after a detected user drift."""

    def __init__(
        self,
        item_categories: list[int] | np.ndarray,
        config: RerankConfig,
        detector_name: str,
        always_active: bool = False,
    ) -> None:
        """Store item categories and initialise per-user activation state."""

        self.item_categories = np.asarray(item_categories, dtype=int)
        self.config = config
        self.detector_name = detector_name
        self.always_active = always_active
        self.active_until: dict[int, int] = {}
        self.activation_count = 0

    def activate(self, user_id: int, timestamp: int) -> None:
        """Activate drift-aware reranking for a user for a fixed time window."""

        self.active_until[user_id] = max(
            self.active_until.get(user_id, -1),
            timestamp + self.config.active_window,
        )
        self.activation_count += 1

    def is_active(self, user_id: int, timestamp: int) -> bool:
        """Return whether reranking should currently affect this user."""

        if self.always_active:
            return True
        return timestamp <= self.active_until.get(user_id, -1)

    def rerank(
        self,
        recommendation: Recommendation,
        user_id: int,
        timestamp: int,
        features: PatternFeatures,
        k: int,
        force_active: bool = False,
        active_reason: str | None = None,
    ) -> tuple[Recommendation, dict[str, float | int | str]]:
        """Combine base scores with recent-pattern scores and return a new top-K."""

        if not recommendation.item_ids:
            return recommendation, {"rerank_active": 0}

        locally_active = self.is_active(user_id, timestamp)
        if not locally_active and not force_active:
            return (
                Recommendation(
                    item_ids=recommendation.item_ids[:k],
                    scores=recommendation.scores[:k],
                ),
                {"rerank_active": 0},
            )

        item_ids = np.asarray(recommendation.item_ids, dtype=int)
        base_scores = np.asarray(recommendation.scores, dtype=float)
        base_norm = _minmax(base_scores)
        categories = self.item_categories[item_ids]
        adaptive_scores = (
            self.config.preference_weight * features.preference[categories]
            + self.config.transition_weight * features.transition[categories]
        )
        combined = (1.0 - self.config.rerank_alpha) * base_norm + (
            self.config.rerank_alpha * adaptive_scores
        )
        order = np.argsort(combined)[::-1][:k]
        return (
            Recommendation(
                item_ids=[int(x) for x in item_ids[order]],
                scores=[float(x) for x in combined[order]],
            ),
            {
                "rerank_active": 1,
                "adaptive_mean": float(adaptive_scores.mean()),
                "base_mean": float(base_scores.mean()),
                "detector": self.detector_name,
                "rerank_reason": active_reason
                or ("local" if locally_active else "shared"),
            },
        )


def _minmax(values: np.ndarray) -> np.ndarray:
    """Normalise scores to [0, 1] so they can be mixed with pattern features."""

    low = float(values.min())
    high = float(values.max())
    if high - low <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)
