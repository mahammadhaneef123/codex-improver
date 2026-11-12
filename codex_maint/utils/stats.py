"""Small statistics helpers (pure Python, no NumPy)."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence

NAN = float("nan")


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float], ddof: int = 0) -> float:
    n = len(values)
    if n - ddof <= 0:
        return 0.0
    mu = mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (n - ddof)
    return math.sqrt(variance)


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile. Returns 0.0 for empty input."""
    if not values:
        return 0.0

    q = min(1.0, max(0.0, q))
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def median(values: Sequence[float]) -> float:
    return quantile(values, 0.5)


def safe_div(numerator: float, denominator: float, default: float = NAN) -> float:
    if denominator == 0 or math.isnan(denominator) or math.isnan(numerator):
        return default
    return numerator / denominator


def finite(values: Sequence[float]) -> list[float]:
    """Drop NaN/inf values."""
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def bin_edges(values: Sequence[float], bins: int = 10) -> list[float]:
    """Quantile edges for equal-frequency binning."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    return [quantile(values, k / bins) for k in range(1, bins)]


def bin_proportions(
    values: Sequence[float],
    edges: Sequence[float],
    eps: float = 0.0,
) -> list[float]:
    """Share of values falling into each bucket defined by `edges`."""
    counts = [0] * (len(edges) + 1)
    for value in values:
        counts[bisect_right(edges, value)] += 1

    total = len(values)
    if total == 0:
        return [0.0] * (len(edges) + 1)

    proportions = [count / total for count in counts]
    if eps > 0:
        proportions = [max(p, eps) for p in proportions]
        mass = sum(proportions)
        proportions = [p / mass for p in proportions]
    return proportions
