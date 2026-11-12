"""Model factory and registry, keeping config names decoupled from classes."""

from __future__ import annotations

from typing import Any

from ..config import ConfigError, ModelConfig
from .base import Classifier
from .linear import LogisticRegression


def _logistic(cfg: ModelConfig, seed: int) -> LogisticRegression:
    return LogisticRegression(
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
        l2=cfg.l2,
        seed=seed,
        class_weight=False,
    )


def _logistic_balanced(cfg: ModelConfig, seed: int) -> LogisticRegression:
    return LogisticRegression(
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
        l2=cfg.l2,
        seed=seed,
        class_weight=True,
    )


MODEL_REGISTRY: dict[str, Any] = {
    "logistic": _logistic,
    "logistic_balanced": _logistic_balanced,
}


def create_model(cfg: ModelConfig, *, seed: int = 42) -> Classifier:
    factory = MODEL_REGISTRY.get(cfg.name)
    if factory is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ConfigError(f"unknown model '{cfg.name}' (available: {available})")
    return factory(cfg, seed)


def model_from_dict(payload: dict[str, object]) -> Classifier:
    kind = str(payload.get("kind", ""))
    if kind != "logistic":
        raise ConfigError(f"cannot restore unsupported model kind: {kind!r}")
    return LogisticRegression.from_dict(payload)


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)
