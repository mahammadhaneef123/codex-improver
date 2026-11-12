"""Logistic regression trained with mini-batch gradient descent.

Deliberately dependency-free so the whole workspace runs on a bare Python
install; the interface mirrors what a scikit-learn estimator would expose.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from collections.abc import Sequence

from .base import Classifier, ModelError

KIND = "logistic"


@dataclass(slots=True)
class LogisticRegression(Classifier):
    learning_rate: float = 0.1
    epochs: int = 300
    l2: float = 0.001
    batch_size: int = 64
    class_weight: bool = True
    seed: int = 42
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        X_val: Sequence[Sequence[float]] | None = None,
        y_val: Sequence[int] | None = None,
        patience: int = 20,
    ) -> "LogisticRegression":
        self._validate(X, y, X_val, y_val)

        dimensions = len(X[0])
        self.weights = [0.0] * dimensions
        self.bias = 0.0
        self.history = []

        positives = sum(1 for label in y if label == 1)
        negatives = len(y) - positives
        positive_weight = (
            negatives / positives if self.class_weight and positives > 0 else 1.0
        )

        rng = random.Random(self.seed)
        indices = list(range(len(X)))
        order = [positive_weight if label == 1 else 1.0 for label in y]

        best_loss = math.inf
        best_weights = list(self.weights)
        best_bias = self.bias
        stale_epochs = 0
        use_validation = X_val is not None and y_val is not None

        for epoch in range(self.epochs):
            rng.shuffle(indices)
            decay = 1.0 / (1.0 + 0.01 * epoch)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                gradients = [0.0] * dimensions
                bias_gradient = 0.0

                for index in batch:
                    row = X[index]
                    error = (self._probability(row) - y[index]) * order[index]
                    for feature in range(dimensions):
                        gradients[feature] += error * row[feature]
                    bias_gradient += error

                count = len(batch)
                rate = self.learning_rate * decay
                for feature in range(dimensions):
                    gradient = gradients[feature] / count + self.l2 * self.weights[feature]
                    self.weights[feature] -= rate * gradient
                self.bias -= rate * (bias_gradient / count)

            train_loss = self._log_loss(X, y, order)
            record = {"epoch": float(epoch + 1), "loss": train_loss}
            monitored = train_loss

            if use_validation:
                val_loss = self._log_loss(X_val, y_val, [1.0] * len(y_val))
                record["val_loss"] = val_loss
                monitored = val_loss

            self.history.append(record)

            if monitored < best_loss - 1e-6:
                best_loss = monitored
                best_weights = list(self.weights)
                best_bias = self.bias
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        self.weights = best_weights
        self.bias = best_bias
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        if not self.weights:
            raise ModelError("model is not fitted")
        return [self._probability(row) for row in X]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": KIND,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "batch_size": self.batch_size,
            "class_weight": self.class_weight,
            "seed": self.seed,
            "weights": list(self.weights),
            "bias": self.bias,
            "history_size": len(self.history),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LogisticRegression":
        model = cls(
            learning_rate=float(payload["learning_rate"]),
            epochs=int(payload["epochs"]),
            l2=float(payload["l2"]),
            batch_size=int(payload["batch_size"]),
            class_weight=bool(payload["class_weight"]),
            seed=int(payload["seed"]),
        )
        model.weights = [float(v) for v in payload["weights"]]  # type: ignore[union-attr]
        model.bias = float(payload["bias"])
        return model

    def _probability(self, row: Sequence[float]) -> float:
        score = self.bias + sum(w * x for w, x in zip(self.weights, row))
        if score < -35:
            return 0.0
        if score > 35:
            return 1.0
        return 1.0 / (1.0 + math.exp(-score))

    def _log_loss(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        weights: Sequence[float],
    ) -> float:
        total = 0.0
        weight_sum = 0.0
        for row, label, weight in zip(X, y, weights):
            probability = min(max(self._probability(row), 1e-12), 1 - 1e-12)
            if label == 1:
                total -= weight * math.log(probability)
            else:
                total -= weight * math.log(1 - probability)
            weight_sum += weight
        return total / weight_sum if weight_sum else 0.0

    def _validate(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        X_val: Sequence[Sequence[float]] | None,
        y_val: Sequence[int] | None,
    ) -> None:
        if not X:
            raise ModelError("cannot fit on an empty feature matrix")
        if len(X) != len(y):
            raise ModelError("X and y must have the same number of rows")

        width = len(X[0])
        if width == 0 or any(len(row) != width for row in X):
            raise ModelError("feature matrix must be rectangular")

        if any(label not in (0, 1) for label in y):
            raise ModelError("labels must be 0 or 1")

        if (X_val is None) != (y_val is None):
            raise ModelError("X_val and y_val must be provided together")
        if X_val is not None and y_val is not None and len(X_val) != len(y_val):
            raise ModelError("X_val and y_val must have the same number of rows")
