"""Classifier interface implemented by every model in the registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Classifier(ABC):
    """Minimal binary-classification contract used by the pipeline."""

    @abstractmethod
    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        X_val: Sequence[Sequence[float]] | None = None,
        y_val: Sequence[int] | None = None,
        patience: int = 20,
    ) -> "Classifier":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        raise NotImplementedError

    def predict(
        self,
        X: Sequence[Sequence[float]],
        threshold: float = 0.5,
    ) -> list[int]:
        return [1 if probability >= threshold else 0 for probability in self.predict_proba(X)]

    @abstractmethod
    def to_dict(self) -> dict[str, object]:
        raise NotImplementedError


class ModelError(RuntimeError):
    """Raised for invalid training data or model state."""
