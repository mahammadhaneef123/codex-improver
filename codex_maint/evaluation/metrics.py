"""Binary classification metrics implemented without third-party libraries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class ConfusionMatrix:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    def to_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn}


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> ConfusionMatrix:
    _validate(y_true, y_pred)
    matrix = ConfusionMatrix()
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            matrix.tp += 1
        elif actual == 0 and predicted == 0:
            matrix.tn += 1
        elif actual == 0 and predicted == 1:
            matrix.fp += 1
        else:
            matrix.fn += 1
    return matrix


def accuracy(matrix: ConfusionMatrix) -> float:
    return (matrix.tp + matrix.tn) / matrix.total if matrix.total else 0.0


def precision(matrix: ConfusionMatrix) -> float:
    denominator = matrix.tp + matrix.fp
    return matrix.tp / denominator if denominator else 0.0


def recall(matrix: ConfusionMatrix) -> float:
    denominator = matrix.tp + matrix.fn
    return matrix.tp / denominator if denominator else 0.0


def f1_score(matrix: ConfusionMatrix) -> float:
    p = precision(matrix)
    r = recall(matrix)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def brier_score(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    _validate(y_true, probabilities)
    if not y_true:
        return 0.0
    return sum((p - y) ** 2 for y, p in zip(y_true, probabilities)) / len(y_true)


def roc_auc(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    """Rank-based AUC (equivalent to the Mann-Whitney U statistic)."""
    _validate(y_true, probabilities)
    pairs = sorted(zip(probabilities, y_true))
    count = len(pairs)
    if count == 0:
        return 0.5

    ranks = [0.0] * count
    start = 0
    while start < count:
        end = start
        while end < count and pairs[end][0] == pairs[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for index in range(start, end):
            ranks[index] = average_rank
        start = end

    positives = sum(1 for _, label in pairs if label == 1)
    negatives = count - positives
    if positives == 0 or negatives == 0:
        return 0.5

    rank_sum = sum(
        rank for rank, (_, label) in zip(ranks, pairs) if label == 1
    )
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def classification_report(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    _validate(y_true, probabilities)
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    matrix = confusion_matrix(y_true, predictions)
    positives = sum(y_true)

    return {
        "n": len(y_true),
        "positives": positives,
        "negatives": len(y_true) - positives,
        "positive_rate": round(positives / len(y_true), 4) if y_true else 0.0,
        "threshold": round(threshold, 4),
        **matrix.to_dict(),
        "accuracy": round(accuracy(matrix), 4),
        "precision": round(precision(matrix), 4),
        "recall": round(recall(matrix), 4),
        "f1": round(f1_score(matrix), 4),
        "roc_auc": round(roc_auc(y_true, probabilities), 4),
        "brier": round(brier_score(y_true, probabilities), 4),
    }


def _validate(y_true: Sequence[int], other: Sequence[float]) -> None:
    if len(y_true) != len(other):
        raise ValueError("y_true and predictions must have the same length")
