"""Recent interaction pattern extraction for drift-aware adaptation.

After a drift signal, the reranker needs short-term behavioral evidence. This
file keeps each user's recent category history and converts it into simple,
interpretable features: recent preference and first-order transitions.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from src.core.events import InteractionEvent


@dataclass(frozen=True)
class PatternConfig:
    """Thresholds controlling which short-term patterns are trusted."""

    history_length: int = 45
    min_support: int = 2
    min_confidence: float = 0.12


@dataclass(frozen=True)
class PatternFeatures:
    """Feature bundle passed from the extractor to the reranker."""

    preference: np.ndarray
    transition: np.ndarray
    last_category: int | None
    n_interactions: int
    n_transitions: int


class RecentPatternExtractor:
    """Extract short-term category preferences and Markov transitions."""

    def __init__(self, n_categories: int, config: PatternConfig) -> None:
        """Create bounded recent category histories for all users."""

        self.n_categories = n_categories
        self.config = config
        self.histories: defaultdict[int, deque[int]] = defaultdict(
            lambda: deque(maxlen=config.history_length)
        )

    def update(self, event: InteractionEvent) -> None:
        """Append the latest consumed category to the user's short history."""

        self.histories[event.user_id].append(event.category_id)

    def features(self, user_id: int) -> PatternFeatures:
        """Build current recent-preference and transition features for a user."""

        history = list(self.histories[user_id])
        preference = self._preference_vector(history)
        transition = self._transition_vector(history)
        n_transitions = max(0, len(history) - 1)
        return PatternFeatures(
            preference=preference,
            transition=transition,
            last_category=history[-1] if history else None,
            n_interactions=len(history),
            n_transitions=n_transitions,
        )

    def _preference_vector(self, history: list[int]) -> np.ndarray:
        """Estimate recent category preference with mild recency weighting."""

        if not history:
            return np.full(self.n_categories, 1.0 / self.n_categories)
        counts = np.bincount(history, minlength=self.n_categories).astype(float)
        # Recency weighting inside the short window emphasizes the emerging regime.
        for idx, category in enumerate(history):
            counts[category] += (idx + 1) / len(history)
        return counts / counts.sum()

    def _transition_vector(self, history: list[int]) -> np.ndarray:
        """Estimate likely next categories from first-order transitions."""

        if len(history) < 2:
            return np.zeros(self.n_categories, dtype=float)
        last_category = history[-1]
        outgoing = np.zeros(self.n_categories, dtype=float)
        support = 0
        for source, target in zip(history[:-1], history[1:]):
            if source == last_category:
                outgoing[target] += 1.0
                support += 1
        if support < self.config.min_support:
            return np.zeros(self.n_categories, dtype=float)
        confidence = outgoing / max(1, support)
        confidence[confidence < self.config.min_confidence] = 0.0
        total = confidence.sum()
        if total <= 0.0:
            return np.zeros(self.n_categories, dtype=float)
        return confidence / total
