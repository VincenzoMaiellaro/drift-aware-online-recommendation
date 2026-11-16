"""User-level concept drift detectors for recommendation streams.

The experiment uses detectors after each online model update. Detectors observe
only a user's consumed categories and emit a `DriftSignal`; the adaptive reranker
is activated only when such a signal reports drift.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.core.events import InteractionEvent


@dataclass(frozen=True)
class DriftSignal:
    """Standard detector output consumed by the experiment runner."""

    user_id: int
    timestamp: int
    detector: str
    drift_detected: bool
    score: float
    threshold: float
    details: dict[str, float | int | str]


@dataclass(frozen=True)
class KLConfig:
    """Hyperparameters for the KL-divergence window detector."""

    reference_window: int = 70
    current_window: int = 35
    threshold: float = 0.55
    min_events: int = 105
    cooldown: int = 120
    smoothing: float = 0.01


@dataclass(frozen=True)
class JSDConfig:
    """Hyperparameters for the Jensen-Shannon divergence window detector."""

    reference_window: int = 45
    current_window: int = 25
    threshold: float = 0.12
    min_events: int = 70
    cooldown: int = 12
    smoothing: float = 0.01


class _WindowDivergenceDetector:
    """Shared reference/current window detector for categorical divergence."""

    def __init__(
        self,
        n_categories: int,
        config: KLConfig | JSDConfig,
        name: str,
        divergence: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        """Initialise per-user category histories and cooldown state."""

        self.n_categories = n_categories
        self.config = config
        self.name = name
        self.divergence = divergence
        max_len = config.reference_window + config.current_window
        if config.min_events > max_len:
            raise ValueError(
                f"{name} detector cannot satisfy min_events={config.min_events} "
                f"with reference_window + current_window = {max_len}."
            )
        self.histories: defaultdict[int, deque[int]] = defaultdict(
            lambda: deque(maxlen=max_len)
        )
        self.last_detection_event_index: defaultdict[int, int] = defaultdict(
            lambda: -10**12
        )

    def update(self, event: InteractionEvent) -> DriftSignal:
        """Update one user's windows and return whether drift was detected."""

        history = self.histories[event.user_id]
        history.append(event.category_id)
        score = float("nan")
        drift_detected = False
        last_detection = self.last_detection_event_index[event.user_id]
        user_event_index, events_since_last_detection = _detection_timing(
            event,
            last_detection,
        )
        required_events = max(
            self.config.min_events,
            self.config.reference_window + self.config.current_window,
        )

        if len(history) >= required_events:
            values = list(history)
            reference = values[: self.config.reference_window]
            current = values[
                self.config.reference_window : self.config.reference_window
                + self.config.current_window
            ]
            p_ref = _category_distribution(
                reference, self.n_categories, self.config.smoothing
            )
            p_cur = _category_distribution(
                current, self.n_categories, self.config.smoothing
            )
            score = self.divergence(p_cur, p_ref)
            enough_cooldown = _enough_user_cooldown(
                user_event_index,
                last_detection,
                self.config.cooldown,
            )
            drift_detected = bool(score >= self.config.threshold and enough_cooldown)
            if drift_detected:
                self.last_detection_event_index[event.user_id] = user_event_index
                recent = values[-self.config.current_window :]
                history.clear()
                history.extend(recent)

        return DriftSignal(
            user_id=event.user_id,
            timestamp=event.timestamp,
            detector=self.name,
            drift_detected=drift_detected,
            score=score,
            threshold=self.config.threshold,
            details={
                "history_size": len(history),
                "cooldown_unit": "user_interactions",
                "user_events_since_last_detection": events_since_last_detection,
            },
        )


class KLDriftDetector(_WindowDivergenceDetector):
    """KL-divergence detector over reference/current category windows."""

    def __init__(self, n_categories: int, config: KLConfig) -> None:
        """Initialise the KL detector."""

        super().__init__(n_categories, config, "kl", _symmetric_kl)


class JSDDriftDetector(_WindowDivergenceDetector):
    """Jensen-Shannon detector over reference/current category windows."""

    def __init__(self, n_categories: int, config: JSDConfig) -> None:
        """Initialise the JSD detector."""

        super().__init__(n_categories, config, "jsd", _jensen_shannon_divergence)


class _HistoryDivergenceDetector:
    """Compare a dataset history baseline to recent clicked categories."""

    def __init__(
        self,
        n_categories: int,
        config: KLConfig | JSDConfig,
        name: str,
        divergence: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        """Initialise recent-click windows for MIND-style user histories."""

        self.n_categories = n_categories
        self.config = config
        self.name = name
        self.divergence = divergence
        if config.min_events > config.current_window:
            raise ValueError(
                f"{name} detector cannot satisfy min_events={config.min_events} "
                f"with current_window={config.current_window}."
            )
        self.histories: defaultdict[int, deque[int]] = defaultdict(
            lambda: deque(maxlen=config.current_window)
        )
        self.last_detection: defaultdict[int, int] = defaultdict(lambda: -10**12)

    def update(self, event: InteractionEvent) -> DriftSignal:
        """Compare the event history baseline with the user's recent clicks."""

        history = self.histories[event.user_id]
        history.append(event.category_id)
        baseline = event.reference_user_distribution
        score = float("nan")
        drift_detected = False
        last_detection = self.last_detection[event.user_id]
        user_event_index, events_since_last_detection = _detection_timing(
            event,
            last_detection,
        )
        details: dict[str, float | int | str] = {
            "recent_window_size": len(history),
            "source_history_size": int(event.source_history_size),
            "cooldown_unit": "user_interactions",
            "user_events_since_last_detection": events_since_last_detection,
        }

        if baseline is not None and len(history) >= self.config.min_events:
            p_ref = np.asarray(baseline, dtype=float)
            p_cur = _category_distribution(
                history, self.n_categories, self.config.smoothing
            )
            score = self.divergence(p_cur, p_ref)
            baseline_top = int(np.argmax(p_ref))
            recent_top = int(np.argmax(p_cur))
            details.update(
                {
                    "baseline_top_category": baseline_top,
                    "recent_top_category": recent_top,
                    "baseline_top_probability": float(p_ref[baseline_top]),
                    "recent_top_probability": float(p_cur[recent_top]),
                    "rule": (
                        f"IF recent behavior shifts from label {baseline_top} "
                        f"to {recent_top} THEN boost label {recent_top}"
                    ),
                }
            )
            enough_cooldown = _enough_user_cooldown(
                user_event_index,
                last_detection,
                self.config.cooldown,
            )
            changed_category = baseline_top != recent_top
            drift_detected = bool(
                changed_category
                and score >= self.config.threshold
                and enough_cooldown
            )
            if drift_detected:
                self.last_detection[event.user_id] = user_event_index

        return DriftSignal(
            user_id=event.user_id,
            timestamp=event.timestamp,
            detector=self.name,
            drift_detected=drift_detected,
            score=score,
            threshold=self.config.threshold,
            details=details,
        )


class HistoryKLDriftDetector(_HistoryDivergenceDetector):
    """KL detector that compares a dataset history baseline to recent clicks."""

    def __init__(self, n_categories: int, config: KLConfig) -> None:
        """Initialise the MIND history-vs-recent KL detector."""

        super().__init__(n_categories, config, "history_kl", _symmetric_kl)


class HistoryJSDDriftDetector(_HistoryDivergenceDetector):
    """JSD detector that compares a dataset history baseline to recent clicks."""

    def __init__(self, n_categories: int, config: JSDConfig) -> None:
        """Initialise the MIND history-vs-recent JSD detector."""

        super().__init__(
            n_categories,
            config,
            "history_jsd",
            _jensen_shannon_divergence,
        )


class OracleDriftDetector:
    """Emit one synthetic oracle drift signal per user at the hidden drift point."""

    def __init__(self) -> None:
        """Track which users have already received their oracle signal."""

        self.detected_users: set[int] = set()
        self.name = "oracle"

    def update(self, event: InteractionEvent) -> DriftSignal:
        """Use hidden simulation metadata to mark the first drift-onset event."""

        drift_start = int(getattr(event, "drift_start", event.user_drift_point))
        if drift_start < 0:
            return DriftSignal(
                user_id=event.user_id,
                timestamp=event.timestamp,
                detector=self.name,
                drift_detected=False,
                score=float("nan"),
                threshold=0.0,
                details={"oracle_delay": -1, "oracle_reference": "no_drift"},
            )
        drift_detected = (
            event.user_id not in self.detected_users
            and event.timestamp >= drift_start
        )
        if drift_detected:
            self.detected_users.add(event.user_id)
        delay = event.timestamp - drift_start
        return DriftSignal(
            user_id=event.user_id,
            timestamp=event.timestamp,
            detector=self.name,
            drift_detected=drift_detected,
            score=float(delay),
            threshold=0.0,
            details={"oracle_delay": int(delay), "oracle_reference": "drift_start"},
        )


def _category_distribution(
    categories: list[int] | deque[int], n_categories: int, smoothing: float
) -> np.ndarray:
    """Estimate a smoothed categorical distribution from a short history."""

    counts = np.full(n_categories, smoothing, dtype=float)
    for category in categories:
        counts[int(category)] += 1.0
    return counts / counts.sum()


def _symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Compute symmetric KL divergence between two probability vectors."""

    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))))


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two probability vectors."""

    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / midpoint))
        + 0.5 * np.sum(q * np.log(q / midpoint))
    )


def _enough_user_cooldown(
    current_user_event_index: int,
    previous_user_event_index: int,
    cooldown: int,
) -> bool:
    """Return whether enough user-level interactions elapsed since detection."""

    return current_user_event_index - previous_user_event_index >= max(0, cooldown)


def _detection_timing(
    event: InteractionEvent,
    last_detection: int,
) -> tuple[int, int]:
    """Return the user-local event index and interactions since detection."""

    user_event_index = int(getattr(event, "user_event_index", event.timestamp))
    events_since_last_detection = (
        -1 if last_detection < 0 else user_event_index - last_detection
    )
    return user_event_index, events_since_last_detection
