"""Distribution drift detection (PSI + Kolmogorov-Smirnov)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..utils.stats import bin_edges, bin_proportions

PSI_EPS = 1e-4


def psi(
    expected_proportions: Sequence[float],
    actual_proportions: Sequence[float],
    *,
    eps: float = PSI_EPS,
) -> float:
    """Population Stability Index between two binned distributions."""
    if len(expected_proportions) != len(actual_proportions):
        raise ValueError("expected and actual proportions must align")

    total = 0.0
    for expected, actual in zip(expected_proportions, actual_proportions):
        expected = max(float(expected), eps)
        actual = max(float(actual), eps)
        total += (actual - expected) * math.log(actual / expected)
    return total


def psi_from_values(
    expected: Sequence[float],
    actual: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    edges = bin_edges(expected, bins)
    return psi(
        bin_proportions(expected, edges),
        bin_proportions(actual, edges),
    )


def ks_statistic(reference: Sequence[float], live: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov distance."""
    if not reference or not live:
        return 0.0

    left = sorted(reference)
    right = sorted(live)
    n_left, n_right = len(left), len(right)
    i = j = 0
    distance = 0.0

    while i < n_left and j < n_right:
        if left[i] == right[j]:
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
        distance = max(distance, abs(i / n_left - j / n_right))

    distance = max(distance, abs(1.0 - j / n_right), abs(i / n_left - 1.0))
    return distance


def _status(value: float, warn: float, alert: float) -> str:
    if value >= alert:
        return "alert"
    if value >= warn:
        return "warning"
    return "stable"


def compare_matrix(
    live_matrix: Sequence[Sequence[float]],
    reference_edges: Sequence[Sequence[float]],
    reference_props: Sequence[Sequence[float]],
    feature_names: Sequence[str],
    *,
    warn: float = 0.1,
    alert: float = 0.25,
) -> dict[str, object]:
    """Compare a live feature matrix against the training reference bins."""
    features: list[dict[str, object]] = []

    for index, name in enumerate(feature_names):
        column = [row[index] for row in live_matrix]
        actual = bin_proportions(column, reference_edges[index])
        value = psi(reference_props[index], actual)
        features.append(
            {
                "name": name,
                "psi": round(value, 4),
                "status": _status(value, warn, alert),
            }
        )

    maximum = max((float(item["psi"]) for item in features), default=0.0)
    overall = "stable"
    if any(item["status"] == "alert" for item in features):
        overall = "alert"
    elif any(item["status"] == "warning" for item in features):
        overall = "warning"

    return {
        "status": overall,
        "max_psi": round(maximum, 4),
        "warn_threshold": warn,
        "alert_threshold": alert,
        "features": features,
    }
