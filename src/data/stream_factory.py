"""Stream construction helpers for experiment runners."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from src.data.generator import SyntheticDriftStream, SyntheticStreamConfig
from src.data.mind_dataset import MINDBehaviorStream, MINDStreamConfig


Stream = SyntheticDriftStream | MINDBehaviorStream
_MIND_SOURCES = frozenset({"mind", "mindsmall", "mindsmall_train"})


def build_stream(config: dict[str, Any], seed: int) -> Stream:
    """Create either the controlled synthetic stream or the MIND stream."""

    data_config = dict(config["data"])
    source = str(data_config.get("source", "synthetic")).strip().lower()
    if source == "synthetic":
        allowed = dataclass_field_names(SyntheticStreamConfig) - {"seed"}
        kwargs = {key: value for key, value in data_config.items() if key in allowed}
        return SyntheticDriftStream(SyntheticStreamConfig(seed=seed, **kwargs))
    if source in _MIND_SOURCES:
        return MINDBehaviorStream(
            MINDStreamConfig(
                path=str(data_config.get("path", "MINDsmall_train")),
                n_events=int(data_config.get("n_events", 0)),
                category_level=str(data_config.get("category_level", "category")),
                smoothing=float(data_config.get("smoothing", 0.05)),
            )
        )
    raise ValueError(
        f"Unknown data.source={source!r}; expected 'synthetic' or 'mind'."
    )


def is_mind_source(config: dict[str, Any]) -> bool:
    """Return whether the configured experiment uses MINDsmall."""

    source = str(config["data"].get("source", "synthetic")).strip().lower()
    return source in _MIND_SOURCES


def dataclass_field_names(dataclass_type: type[Any]) -> set[str]:
    """Return dataclass field names without relying on private attributes."""

    return {field.name for field in fields(dataclass_type)}
