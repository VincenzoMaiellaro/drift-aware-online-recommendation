"""Online behavioral clustering for sharing temporary adaptation rules.

The clusterer is intentionally dataset-agnostic: it only sees category-level
features produced by the recent pattern extractor. Each user is assigned to an
online centroid, and drift-triggered adaptation rules are stored in a bounded
cluster-level rule bank for a fixed time window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.methods.pattern_extraction import PatternFeatures


@dataclass(frozen=True)
class OnlineClusterConfig:
    """Hyperparameters for online user clustering and rule sharing."""

    n_clusters: int = 8
    min_interactions: int = 5
    centroid_learning_rate: float = 0.08
    new_cluster_similarity: float = 0.85
    rule_learning_rate: float = 0.5
    rule_ttl: int = 1500
    shared_rule_weight: float = 0.45
    min_similarity: float = 0.10
    rule_merge_similarity: float = 0.80
    max_rules_per_cluster: int = 3
    preference_weight: float = 0.85
    transition_weight: float = 0.15


@dataclass
class ClusterRule:
    """A short-lived adaptation rule learned from drifted users in a cluster."""

    preference: np.ndarray
    transition: np.ndarray
    active_until: int
    source_user_id: int
    rule_id: int
    created_at: int
    updated_at: int
    support: int
    detector: str


class OnlineUserClusterer:
    """Maintain streaming user clusters and cluster-level adaptation rules."""

    def __init__(self, n_categories: int, config: OnlineClusterConfig) -> None:
        """Create fixed-size centroid storage and empty rule state."""

        self.n_categories = int(n_categories)
        self.config = config
        n_clusters = max(1, int(config.n_clusters))
        self.centroids = np.zeros((n_clusters, 2 * self.n_categories), dtype=float)
        self.initialized = np.zeros(n_clusters, dtype=bool)
        self.user_clusters: dict[int, int] = {}
        self.last_user_update: dict[int, int] = {}
        self.rules: dict[int, list[ClusterRule]] = {}
        self._next_rule_id = 0

    def update_user(
        self,
        user_id: int,
        timestamp: int,
        features: PatternFeatures,
        force: bool = False,
    ) -> int | None:
        """Assign or refresh one user's cluster from current recent behavior."""

        if not force and features.n_interactions < self.config.min_interactions:
            return self.user_clusters.get(user_id)

        vector = self._feature_vector(features)
        cluster_id = self._choose_cluster(user_id, vector)
        self._update_centroid(cluster_id, vector)
        self.user_clusters[user_id] = cluster_id
        self.last_user_update[user_id] = int(timestamp)
        return cluster_id

    def publish_rule(
        self,
        user_id: int,
        timestamp: int,
        features: PatternFeatures,
        detector: str,
    ) -> dict[str, float | int | str]:
        """Store a temporary adaptation rule for the user's current cluster."""

        cluster_id = self.update_user(user_id, timestamp, features, force=True)
        if cluster_id is None:
            return {
                "cluster_id": -1,
                "cluster_rule_published": 0,
            }

        preference = _probability_vector(features.preference)
        transition = _sparse_probability_vector(features.transition)
        active_until = int(timestamp) + int(self.config.rule_ttl)
        self.rules.setdefault(cluster_id, [])
        self._prune_expired_rules(cluster_id, timestamp)
        existing, similarity = self._nearest_rule(
            self.rules.get(cluster_id, []),
            preference,
            transition,
        )
        if existing is None or similarity < self.config.rule_merge_similarity:
            rule = ClusterRule(
                preference=preference,
                transition=transition,
                active_until=active_until,
                source_user_id=int(user_id),
                rule_id=self._next_rule_id,
                created_at=int(timestamp),
                updated_at=int(timestamp),
                support=1,
                detector=str(detector),
            )
            self._next_rule_id += 1
            self.rules.setdefault(cluster_id, []).append(rule)
            self._trim_rules(cluster_id)
            support = 1
            rule_id = rule.rule_id
            created = 1
        else:
            learning_rate = float(np.clip(self.config.rule_learning_rate, 0.0, 1.0))
            existing.preference = _probability_vector(
                (1.0 - learning_rate) * existing.preference
                + learning_rate * preference
            )
            existing.transition = _sparse_probability_vector(
                (1.0 - learning_rate) * existing.transition
                + learning_rate * transition
            )
            existing.active_until = max(existing.active_until, active_until)
            existing.source_user_id = int(user_id)
            existing.updated_at = int(timestamp)
            existing.support += 1
            existing.detector = str(detector)
            support = existing.support
            rule_id = existing.rule_id
            created = 0

        return {
            "cluster_id": int(cluster_id),
            "cluster_rule_published": 1,
            "cluster_rule_created": int(created),
            "cluster_rule_id": int(rule_id),
            "cluster_rule_support": int(support),
            "cluster_rule_similarity": float(similarity),
        }

    def features_for_user(
        self,
        user_id: int,
        timestamp: int,
        base_features: PatternFeatures,
    ) -> tuple[PatternFeatures, dict[str, float | int | str]]:
        """Blend a user's features with an active rule from their cluster."""

        cluster_id = self.user_clusters.get(user_id)
        details: dict[str, float | int | str] = {
            "cluster_id": int(cluster_id) if cluster_id is not None else -1,
            "cluster_rule_active": 0,
            "cluster_rule_weight": 0.0,
        }
        if cluster_id is None:
            return base_features, details

        self._prune_expired_rules(cluster_id, timestamp)
        rule, similarity = self._best_rule_for_user(
            self.rules.get(cluster_id, []),
            base_features,
        )
        if rule is None:
            return base_features, details
        age = max(0, int(timestamp) - int(rule.updated_at))
        details.update(
            {
                "cluster_rule_id": int(rule.rule_id),
                "cluster_rule_age": age,
                "cluster_rule_source_user": int(rule.source_user_id),
                "cluster_rule_support": int(rule.support),
                "cluster_rule_detector": str(rule.detector),
            }
        )
        if int(timestamp) > rule.active_until:
            return base_features, details

        details["cluster_similarity"] = float(similarity)
        if similarity < self.config.min_similarity:
            return base_features, details

        weight = float(np.clip(self.config.shared_rule_weight, 0.0, 1.0))
        preference = _probability_vector(
            (1.0 - weight) * base_features.preference + weight * rule.preference
        )
        transition = _sparse_probability_vector(
            (1.0 - weight) * base_features.transition + weight * rule.transition
        )
        details.update(
            {
                "cluster_rule_active": 1,
                "cluster_rule_weight": weight,
            }
        )
        return (
            PatternFeatures(
                preference=preference,
                transition=transition,
                last_category=base_features.last_category,
                n_interactions=base_features.n_interactions,
                n_transitions=base_features.n_transitions,
            ),
            details,
        )

    def _prune_expired_rules(self, cluster_id: int, timestamp: int) -> None:
        """Remove expired rules for a cluster before matching or publishing."""

        rules = self.rules.get(cluster_id, [])
        if not rules:
            return
        self.rules[cluster_id] = [
            rule for rule in rules if int(timestamp) <= int(rule.active_until)
        ]

    def _trim_rules(self, cluster_id: int) -> None:
        """Keep only the strongest active rules for bounded memory use."""

        max_rules = max(1, int(self.config.max_rules_per_cluster))
        rules = self.rules.get(cluster_id, [])
        if len(rules) <= max_rules:
            return
        rules.sort(key=lambda rule: (rule.support, rule.updated_at), reverse=True)
        self.rules[cluster_id] = rules[:max_rules]

    def _nearest_rule(
        self,
        rules: list[ClusterRule],
        preference: np.ndarray,
        transition: np.ndarray,
    ) -> tuple[ClusterRule | None, float]:
        """Find the most similar active rule to a new candidate rule."""

        candidate = self._rule_vector(preference, transition)
        return self._most_similar_rule(rules, candidate)

    def _best_rule_for_user(
        self,
        rules: list[ClusterRule],
        features: PatternFeatures,
    ) -> tuple[ClusterRule | None, float]:
        """Choose the active cluster rule most compatible with a user."""

        user_vector = self._feature_vector(features)
        return self._most_similar_rule(rules, user_vector)

    def _most_similar_rule(
        self,
        rules: list[ClusterRule],
        target: np.ndarray,
    ) -> tuple[ClusterRule | None, float]:
        """Return the rule with maximum cosine similarity to a target vector."""

        if not rules:
            return None, 0.0
        similarities = [
            self._rule_vector(rule.preference, rule.transition) @ target
            for rule in rules
        ]
        best_idx = int(np.argmax(similarities))
        return rules[best_idx], float(similarities[best_idx])

    def _choose_cluster(self, user_id: int, vector: np.ndarray) -> int:
        """Choose an existing centroid or initialise a new one when needed."""

        existing = self.user_clusters.get(user_id)
        if existing is not None and not self.initialized.any():
            self.initialized[existing] = True
            return existing

        if not self.initialized.any():
            return self._initialise_next_cluster(vector)

        nearest, similarity = self._nearest_cluster(vector)
        has_capacity = not bool(self.initialized.all())
        should_create_cluster = (
            has_capacity and similarity < self.config.new_cluster_similarity
        )
        if should_create_cluster:
            return self._initialise_next_cluster(vector)
        return nearest

    def _initialise_next_cluster(self, vector: np.ndarray) -> int:
        """Initialise the first free centroid slot with a normalized vector."""

        free = np.flatnonzero(~self.initialized)
        cluster_id = int(free[0]) if len(free) else 0
        self.centroids[cluster_id] = vector
        self.initialized[cluster_id] = True
        return cluster_id

    def _nearest_cluster(self, vector: np.ndarray) -> tuple[int, float]:
        """Return the nearest initialized centroid by cosine similarity."""

        initialized_ids = np.flatnonzero(self.initialized)
        if len(initialized_ids) == 0:
            return 0, 0.0
        similarities = self.centroids[initialized_ids] @ vector
        best_idx = int(np.argmax(similarities))
        return int(initialized_ids[best_idx]), float(similarities[best_idx])

    def _update_centroid(self, cluster_id: int, vector: np.ndarray) -> None:
        """Apply an online centroid update for one assigned user vector."""

        learning_rate = float(np.clip(self.config.centroid_learning_rate, 0.0, 1.0))
        if not self.initialized[cluster_id]:
            self.centroids[cluster_id] = vector
            self.initialized[cluster_id] = True
            return
        updated = (1.0 - learning_rate) * self.centroids[cluster_id]
        updated += learning_rate * vector
        self.centroids[cluster_id] = _unit_vector(updated)

    def _feature_vector(self, features: PatternFeatures) -> np.ndarray:
        """Convert recent pattern features to a normalized clustering vector."""

        return self._rule_vector(features.preference, features.transition)

    def _rule_vector(self, preference: np.ndarray, transition: np.ndarray) -> np.ndarray:
        """Convert rule preference and transition scores to matchable vectors."""

        return _unit_vector(
            np.concatenate(
                [
                    self.config.preference_weight * _probability_vector(preference),
                    self.config.transition_weight
                    * _sparse_probability_vector(transition),
                ]
            )
        )


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return a unit-length vector, preserving all-zero vectors."""

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def _probability_vector(values: np.ndarray) -> np.ndarray:
    """Normalise non-negative values to a probability vector."""

    cleaned = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(cleaned.sum())
    if total <= 1e-12:
        return np.full(len(cleaned), 1.0 / max(1, len(cleaned)))
    return cleaned / total


def _sparse_probability_vector(values: np.ndarray) -> np.ndarray:
    """Normalise sparse transition scores while preserving no-evidence zeros."""

    cleaned = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(cleaned.sum())
    if total <= 1e-12:
        return np.zeros_like(cleaned)
    return cleaned / total
