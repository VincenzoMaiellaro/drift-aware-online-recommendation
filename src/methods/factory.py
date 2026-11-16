"""Method construction for online recommendation experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from src.data.stream_factory import is_mind_source
from src.methods.drift_detection import (
    HistoryJSDDriftDetector,
    HistoryKLDriftDetector,
    JSDConfig,
    JSDDriftDetector,
    KLConfig,
    KLDriftDetector,
    OracleDriftDetector,
)
from src.methods.pattern_extraction import PatternConfig, RecentPatternExtractor
from src.methods.recommender import (
    GlobalPopularityRecommender,
    MFConfig,
    OnlineMatrixFactorization,
    RecentPopularityRecommender,
    TimeDecayMatrixFactorization,
)
from src.methods.reranking import DriftAwareReranker, RerankConfig
from src.methods.user_clustering import OnlineClusterConfig, OnlineUserClusterer

SYNTHETIC_PREFIX_METHODS = ("pure_mf", "base_mf")
SYNTHETIC_SUFFIX_METHODS = (
    "recent_popularity",
    "global_popularity",
    "mf_kl_detection",
    "mf_jsd_detection",
    "mf_rerank_always",
    "mf_rerank_always_preference_only",
    "mf_rerank_always_transition_only",
    "mf_rerank_oracle",
    "drift_aware_kl",
    "drift_aware_kl_preference_only",
    "drift_aware_kl_transition_only",
    "drift_aware_jsd",
)
SYNTHETIC_CLUSTER_METHODS = ("drift_aware_kl_clustered", "drift_aware_jsd_clustered")
SYNTHETIC_CLUSTER_ABLATION_METHODS = (
    "drift_aware_kl_clustered_no_force",
    "drift_aware_jsd_clustered_no_force",
)
MIND_PREFIX_METHODS = ("base_mf",)
MIND_SUFFIX_METHODS = (
    "recent_popularity",
    "global_popularity",
    "mf_kl_history_detection",
    "mf_jsd_history_detection",
    "mf_rerank_always",
    "drift_aware_kl",
    "drift_aware_jsd",
)
MIND_CLUSTER_METHODS = ("drift_aware_kl_clustered", "drift_aware_jsd_clustered")


@dataclass
class MethodState:
    """Container for one experimental method and its optional adaptive modules."""

    name: str
    recommender: Any
    detector: Any | None = None
    pattern_extractor: RecentPatternExtractor | None = None
    reranker: DriftAwareReranker | None = None
    clusterer: OnlineUserClusterer | None = None
    cluster_force_activation: bool = True


@dataclass(frozen=True)
class MethodFactory:
    """Build fresh stateful components for one method definition."""

    common: dict[str, Any]
    item_categories: Any
    n_categories: int
    seed: int
    rec_cfg: MFConfig
    popularity_window: int
    pattern_cfg: PatternConfig
    rerank_cfg: RerankConfig
    kl_cfg: KLConfig
    jsd_cfg: JSDConfig | None = None
    cluster_cfg: OnlineClusterConfig | None = None

    def mf(
        self,
        name: str,
        rec_cfg: MFConfig | None = None,
        detector: Any | None = None,
        pattern: bool = False,
        reranker: DriftAwareReranker | None = None,
        clusterer: OnlineUserClusterer | None = None,
        cluster_force_activation: bool = True,
    ) -> MethodState:
        """Create an online-MF method with optional adaptive modules."""

        return MethodState(
            name,
            OnlineMatrixFactorization(
                **self.common,
                config=deepcopy(rec_cfg or self.rec_cfg),
                seed=self.seed,
            ),
            detector=detector,
            pattern_extractor=self.pattern_extractor() if pattern else None,
            reranker=reranker,
            clusterer=clusterer,
            cluster_force_activation=cluster_force_activation,
        )

    def _adaptive_mf(
        self,
        name: str,
        reranker_label: str,
        detector: Any | None = None,
        rerank_cfg: RerankConfig | None = None,
        always_active: bool = False,
        clustered: bool = False,
        cluster_force_activation: bool = True,
    ) -> MethodState:
        """Create an MF method wired to recent patterns and adaptive reranking."""

        return self.mf(
            name,
            detector=detector,
            pattern=True,
            reranker=self.reranker(
                reranker_label,
                rerank_cfg,
                always_active=always_active,
            ),
            clusterer=self.clusterer() if clustered else None,
            cluster_force_activation=cluster_force_activation,
        )

    def recent_popularity(self) -> MethodState:
        """Create the recent-popularity baseline."""

        return MethodState(
            "recent_popularity",
            RecentPopularityRecommender(
                self.common["n_items"],
                self.item_categories,
                window_size=self.popularity_window,
            ),
        )

    def global_popularity(self) -> MethodState:
        """Create the cumulative-popularity baseline."""

        return MethodState(
            "global_popularity",
            GlobalPopularityRecommender(self.common["n_items"], self.item_categories),
        )

    def kl_detector(self) -> KLDriftDetector:
        """Create a fresh synthetic KL detector."""

        return KLDriftDetector(self.n_categories, deepcopy(self.kl_cfg))

    def history_kl_detector(self) -> HistoryKLDriftDetector:
        """Create a fresh MIND history-vs-recent KL detector."""

        return HistoryKLDriftDetector(self.n_categories, deepcopy(self.kl_cfg))

    def history_jsd_detector(self) -> HistoryJSDDriftDetector:
        """Create a fresh MIND history-vs-recent JSD detector."""

        if self.jsd_cfg is None:
            raise ValueError("JSD config is required for JSD methods.")
        return HistoryJSDDriftDetector(self.n_categories, deepcopy(self.jsd_cfg))

    def jsd_detector(self) -> JSDDriftDetector:
        """Create a fresh synthetic JSD detector."""

        if self.jsd_cfg is None:
            raise ValueError("JSD config is required for JSD methods.")
        return JSDDriftDetector(self.n_categories, deepcopy(self.jsd_cfg))

    def pattern_extractor(self) -> RecentPatternExtractor:
        """Create a fresh recent-pattern extractor."""

        return RecentPatternExtractor(self.n_categories, deepcopy(self.pattern_cfg))

    def reranker(
        self,
        label: str,
        cfg: RerankConfig | None = None,
        always_active: bool = False,
    ) -> DriftAwareReranker:
        """Create a fresh drift-aware reranker."""

        return DriftAwareReranker(
            self.item_categories,
            deepcopy(cfg or self.rerank_cfg),
            label,
            always_active=always_active,
        )

    def clusterer(self) -> OnlineUserClusterer:
        """Create a fresh online user clusterer."""

        if self.cluster_cfg is None:
            raise ValueError("Cluster config is required for clustered methods.")
        return OnlineUserClusterer(self.n_categories, deepcopy(self.cluster_cfg))


def _build_method_factory(
    config: dict[str, Any],
    stream: Any,
    seed: int,
    rec_cfg: MFConfig,
    rerank_cfg: RerankConfig | None = None,
) -> MethodFactory:
    """Build the shared stateful-component factory for one experiment stream."""

    return MethodFactory(
        common=recommender_kwargs(stream),
        item_categories=stream.item_categories,
        n_categories=stream.config.n_categories,
        seed=seed,
        rec_cfg=rec_cfg,
        popularity_window=int(config["recommendation"]["popularity_window"]),
        pattern_cfg=build_pattern_config(config),
        rerank_cfg=(
            rerank_cfg if rerank_cfg is not None else build_rerank_config(config)
        ),
        kl_cfg=KLConfig(**config["drift_detection"]["kl"]),
        jsd_cfg=JSDConfig(**config["drift_detection"]["jsd"]),
        cluster_cfg=build_cluster_config(config),
    )


def build_methods(
    config: dict[str, Any],
    stream: Any,
    seed: int,
) -> list[MethodState]:
    """Create the baselines and ablations for the configured experiment."""

    if is_mind_source(config):
        return build_mind_methods(config, stream, seed)

    rec_cfg = build_mf_config(config, category_weight_default=0.18)
    pure_rec_cfg = replace(rec_cfg, category_weight=0.0)
    common = recommender_kwargs(stream)
    rerank_cfg = build_rerank_config(config)
    preference_rerank_cfg = replace(
        rerank_cfg,
        preference_weight=1.0,
        transition_weight=0.0,
    )
    transition_rerank_cfg = replace(
        rerank_cfg,
        preference_weight=0.0,
        transition_weight=1.0,
    )
    cluster_enabled = bool(config.get("clustering", {}).get("enabled", False))
    factory = _build_method_factory(config, stream, seed, rec_cfg, rerank_cfg)

    methods = [
        factory.mf("pure_mf", rec_cfg=pure_rec_cfg),
        factory.mf("base_mf"),
        *build_time_decay_methods(config, common, rec_cfg, seed),
        factory.recent_popularity(),
        factory.global_popularity(),
        factory.mf("mf_kl_detection", detector=factory.kl_detector()),
        factory.mf("mf_jsd_detection", detector=factory.jsd_detector()),
        factory._adaptive_mf(
            "mf_rerank_always",
            "always",
            always_active=True,
        ),
        factory._adaptive_mf(
            "mf_rerank_always_preference_only",
            "always_preference_only",
            rerank_cfg=preference_rerank_cfg,
            always_active=True,
        ),
        factory._adaptive_mf(
            "mf_rerank_always_transition_only",
            "always_transition_only",
            rerank_cfg=transition_rerank_cfg,
            always_active=True,
        ),
        factory._adaptive_mf(
            "mf_rerank_oracle",
            "oracle",
            detector=OracleDriftDetector(),
        ),
        factory._adaptive_mf(
            "drift_aware_kl",
            "kl",
            detector=factory.kl_detector(),
        ),
        factory._adaptive_mf(
            "drift_aware_kl_preference_only",
            "kl_preference_only",
            detector=factory.kl_detector(),
            rerank_cfg=preference_rerank_cfg,
        ),
        factory._adaptive_mf(
            "drift_aware_kl_transition_only",
            "kl_transition_only",
            detector=factory.kl_detector(),
            rerank_cfg=transition_rerank_cfg,
        ),
        factory._adaptive_mf(
            "drift_aware_jsd",
            "jsd",
            detector=factory.jsd_detector(),
        ),
    ]
    if cluster_enabled:
        methods.extend(
            [
                factory._adaptive_mf(
                    "drift_aware_kl_clustered",
                    "kl_clustered",
                    detector=factory.kl_detector(),
                    clustered=True,
                ),
                factory._adaptive_mf(
                    "drift_aware_jsd_clustered",
                    "jsd_clustered",
                    detector=factory.jsd_detector(),
                    clustered=True,
                ),
            ]
        )
        if bool(config.get("clustering", {}).get("ablation_no_force_activation", False)):
            methods.extend(
                [
                    factory._adaptive_mf(
                        "drift_aware_kl_clustered_no_force",
                        "kl_clustered_no_force",
                        detector=factory.kl_detector(),
                        clustered=True,
                        cluster_force_activation=False,
                    ),
                    factory._adaptive_mf(
                        "drift_aware_jsd_clustered_no_force",
                        "jsd_clustered_no_force",
                        detector=factory.jsd_detector(),
                        clustered=True,
                        cluster_force_activation=False,
                    ),
                ]
            )
    return methods


def build_mind_methods(
    config: dict[str, Any],
    stream: Any,
    seed: int,
) -> list[MethodState]:
    """Create the real-data MIND method set with history KL/JSD detectors."""

    rec_cfg = build_mf_config(config, category_weight_default=0.12)
    common = recommender_kwargs(stream)
    cluster_enabled = bool(config.get("clustering", {}).get("enabled", False))
    factory = _build_method_factory(config, stream, seed, rec_cfg)
    methods = [
        factory.mf("base_mf"),
        *build_time_decay_methods(config, common, rec_cfg, seed),
        factory.recent_popularity(),
        factory.global_popularity(),
        factory.mf(
            "mf_kl_history_detection",
            detector=factory.history_kl_detector(),
        ),
        factory.mf(
            "mf_jsd_history_detection",
            detector=factory.history_jsd_detector(),
        ),
        factory._adaptive_mf(
            "mf_rerank_always",
            "always",
            always_active=True,
        ),
        factory._adaptive_mf(
            "drift_aware_kl",
            "kl",
            detector=factory.history_kl_detector(),
        ),
        factory._adaptive_mf(
            "drift_aware_jsd",
            "jsd",
            detector=factory.history_jsd_detector(),
        ),
    ]
    if cluster_enabled:
        methods.extend(
            [
                factory._adaptive_mf(
                    "drift_aware_kl_clustered",
                    "history_kl_clustered",
                    detector=factory.history_kl_detector(),
                    clustered=True,
                ),
                factory._adaptive_mf(
                    "drift_aware_jsd_clustered",
                    "history_jsd_clustered",
                    detector=factory.history_jsd_detector(),
                    clustered=True,
                ),
            ]
        )
    return methods


def synthetic_method_names(config: dict[str, Any]) -> list[str]:
    """Return expected synthetic method names for a config."""

    names = [
        *SYNTHETIC_PREFIX_METHODS,
        *build_time_decay_method_names(config),
        *SYNTHETIC_SUFFIX_METHODS,
    ]
    if bool(config.get("clustering", {}).get("enabled", False)):
        names.extend(SYNTHETIC_CLUSTER_METHODS)
        if bool(config.get("clustering", {}).get("ablation_no_force_activation", False)):
            names.extend(SYNTHETIC_CLUSTER_ABLATION_METHODS)
    return names


def mind_method_names(config: dict[str, Any]) -> list[str]:
    """Return expected MIND method names for a config."""

    names = [
        *MIND_PREFIX_METHODS,
        *build_time_decay_method_names(config),
        *MIND_SUFFIX_METHODS,
    ]
    if bool(config.get("clustering", {}).get("enabled", False)):
        names.extend(MIND_CLUSTER_METHODS)
    return names


def build_time_decay_method_names(config: dict[str, Any]) -> list[str]:
    """Return configured time-decay method names in build order."""

    default_decay = float(config["time_decay"]["decay_rate"])
    return [
        "time_decay_mf"
        if decay_rate == default_decay
        else f"time_decay_mf_decay_{format_decay_rate(decay_rate)}"
        for decay_rate in unique_decay_rates(config)
    ]


def unique_decay_rates(config: dict[str, Any]) -> list[float]:
    """Return unique configured decay rates, preserving configured order."""

    default_decay = float(config["time_decay"]["decay_rate"])
    raw_rates = config["time_decay"].get("decay_rates", [default_decay])
    decay_rates = []
    for raw_rate in raw_rates:
        rate = float(raw_rate)
        if rate not in decay_rates:
            decay_rates.append(rate)
    if default_decay not in decay_rates:
        decay_rates.append(default_decay)
    return decay_rates


def build_cluster_config(config: dict[str, Any]) -> OnlineClusterConfig:
    """Create the optional online clustering configuration with safe defaults."""

    raw = dict(config.get("clustering", {}))
    return OnlineClusterConfig(
        n_clusters=int(raw.get("n_clusters", 8)),
        min_interactions=int(raw.get("min_interactions", 5)),
        centroid_learning_rate=float(raw.get("centroid_learning_rate", 0.08)),
        new_cluster_similarity=float(raw.get("new_cluster_similarity", 0.85)),
        rule_learning_rate=float(raw.get("rule_learning_rate", 0.5)),
        rule_ttl=int(raw.get("rule_ttl", config["adaptation"]["active_window"])),
        shared_rule_weight=float(raw.get("shared_rule_weight", 0.45)),
        min_similarity=float(raw.get("min_similarity", 0.10)),
        rule_merge_similarity=float(raw.get("rule_merge_similarity", 0.80)),
        max_rules_per_cluster=int(raw.get("max_rules_per_cluster", 3)),
        preference_weight=float(raw.get("preference_weight", 0.85)),
        transition_weight=float(raw.get("transition_weight", 0.15)),
    )


def build_mf_config(
    config: dict[str, Any],
    category_weight_default: float,
) -> MFConfig:
    """Create MF hyperparameters from the recommendation config block."""

    recommendation = config["recommendation"]
    return MFConfig(
        embedding_dim=int(recommendation["embedding_dim"]),
        learning_rate=float(recommendation["learning_rate"]),
        regularization=float(recommendation["regularization"]),
        negative_samples=int(recommendation["negative_samples"]),
        init_std=float(recommendation["init_std"]),
        category_weight=float(
            recommendation.get("category_weight", category_weight_default)
        ),
    )


def build_pattern_config(config: dict[str, Any]) -> PatternConfig:
    """Create recent-pattern extraction parameters from the config."""

    adaptation = config["adaptation"]
    return PatternConfig(
        history_length=int(adaptation["history_length"]),
        min_support=int(adaptation["min_support"]),
        min_confidence=float(adaptation["min_confidence"]),
    )


def build_rerank_config(config: dict[str, Any]) -> RerankConfig:
    """Create adaptive reranking parameters from the config."""

    adaptation = config["adaptation"]
    return RerankConfig(
        active_window=int(adaptation["active_window"]),
        rerank_alpha=float(adaptation["rerank_alpha"]),
        preference_weight=float(adaptation["preference_weight"]),
        transition_weight=float(adaptation["transition_weight"]),
    )


def recommender_kwargs(stream: Any) -> dict[str, Any]:
    """Return the shared constructor arguments for recommenders on a stream."""

    return {
        "n_users": stream.config.n_users,
        "n_items": stream.n_items,
        "item_categories": stream.item_categories,
    }


def build_time_decay_methods(
    config: dict[str, Any],
    common: dict[str, Any],
    rec_cfg: MFConfig,
    seed: int,
) -> list[MethodState]:
    """Create the configured time-decay MF sweep for the recency ablation."""

    methods = []
    for decay_rate in unique_decay_rates(config):
        name = build_time_decay_method_name(config, decay_rate)
        methods.append(
            MethodState(
                name,
                TimeDecayMatrixFactorization(
                    **common,
                    config=deepcopy(rec_cfg),
                    decay_rate=decay_rate,
                    drift_boost=float(config["time_decay"]["drift_boost"]),
                    max_elapsed=int(config["time_decay"].get("max_elapsed", 200)),
                    user_factor_decay_floor=float(
                        config["time_decay"].get("user_factor_decay_floor", 0.985)
                    ),
                    seed=seed,
                ),
            )
        )
    return methods


def build_time_decay_method_name(config: dict[str, Any], decay_rate: float) -> str:
    """Return the method name for one time-decay rate."""

    default_decay = float(config["time_decay"]["decay_rate"])
    if decay_rate == default_decay:
        return "time_decay_mf"
    return f"time_decay_mf_decay_{format_decay_rate(decay_rate)}"


def format_decay_rate(decay_rate: float) -> str:
    """Return a stable method-name suffix for a decay-rate value."""

    return f"{decay_rate:g}".replace(".", "p").replace("-", "m")
