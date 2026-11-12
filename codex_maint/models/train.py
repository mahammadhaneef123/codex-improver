"""Training orchestration: fit, tune the decision threshold, report."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from ..config import ModelConfig
from ..evaluation import metrics
from ..utils.timing import Stopwatch
from . import registry


@dataclass(slots=True)
class TrainingResult:
    model: Any
    threshold: float
    history: list[dict[str, float]]
    val_metrics: dict[str, float | int]
    train_seconds: float


def choose_threshold(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    default: float = 0.5,
) -> float:
    """Pick the threshold that maximises F1 on validation data."""
    if not y_true:
        return default

    best_threshold = default
    best_f1 = -1.0
    best_recall = -1.0

    for step in range(5, 96):
        candidate = step / 100.0
        predictions = [1 if p >= candidate else 0 for p in probabilities]
        matrix = metrics.confusion_matrix(y_true, predictions)
        current_f1 = metrics.f1_score(matrix)
        current_recall = metrics.recall(matrix)

        if current_f1 > best_f1 + 1e-9 or (
            abs(current_f1 - best_f1) <= 1e-9 and current_recall > best_recall
        ):
            best_threshold = candidate
            best_f1 = current_f1
            best_recall = current_recall

    return best_threshold


def train_model(
    X_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    X_val: Sequence[Sequence[float]],
    y_val: Sequence[int],
    cfg: ModelConfig,
    *,
    batch_size: int = 64,
    patience: int = 20,
    seed: int = 42,
) -> TrainingResult:
    model = registry.create_model(cfg, seed=seed)
    model.batch_size = batch_size  # type: ignore[attr-defined]

    with Stopwatch() as watch:
        model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            patience=patience,
        )

    val_probabilities = model.predict_proba(X_val)
    threshold = choose_threshold(y_val, val_probabilities, default=cfg.threshold)
    val_metrics = metrics.classification_report(y_val, val_probabilities, threshold)

    return TrainingResult(
        model=model,
        threshold=threshold,
        history=list(getattr(model, "history", [])),
        val_metrics=val_metrics,
        train_seconds=round(watch.elapsed, 4),
    )
