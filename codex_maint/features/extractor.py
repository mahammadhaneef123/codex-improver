"""Feature extraction: raw sensors plus derived wear and rolling signals."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

from ..data.schema import MaintenanceRecord
from ..utils.stats import safe_div

BASE_FEATURES: tuple[str, ...] = (
    "vibration",
    "temperature",
    "pressure",
    "runtime_hours",
    "cycles_since_service",
)

DERIVED_FEATURES: tuple[str, ...] = (
    "temp_vibration_ratio",
    "pressure_deficit",
    "wear_index",
    "vibration_roll_mean",
)

FEATURE_NAMES: tuple[str, ...] = BASE_FEATURES + DERIVED_FEATURES

NAN = float("nan")


def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    """Rolling mean that ignores NaN and starts as soon as data exists."""
    if window < 1:
        raise ValueError("window must be >= 1")

    result: list[float] = []
    history: deque[float] = deque(maxlen=window)
    for value in values:
        if value is not None and not math.isnan(value):
            history.append(value)
        result.append(sum(history) / len(history) if history else NAN)
    return result


def _as_float(value: float | None) -> float:
    return NAN if value is None else float(value)


def raw_feature_matrix(
    records: Sequence[MaintenanceRecord],
    rolling_window: int = 5,
) -> list[list[float]]:
    """Build the un-imputed feature matrix, one row per record."""
    if not records:
        return []

    runtimes = [r.runtime_hours for r in records if r.runtime_hours is not None]
    max_runtime = max(runtimes) if runtimes else 1.0
    max_runtime = max_runtime if max_runtime > 0 else 1.0

    roll_values = [NAN] * len(records)
    order = sorted(
        range(len(records)),
        key=lambda i: (records[i].asset_id, records[i].timestamp, i),
    )

    cursor = 0
    while cursor < len(order):
        asset = records[order[cursor]].asset_id
        end = cursor
        while end < len(order) and records[order[end]].asset_id == asset:
            end += 1

        indices = order[cursor:end]
        series = [_as_float(records[i].vibration) for i in indices]
        for index, value in zip(indices, rolling_mean(series, rolling_window)):
            roll_values[index] = value
        cursor = end

    matrix: list[list[float]] = []
    for index, record in enumerate(records):
        vibration = _as_float(record.vibration)
        temperature = _as_float(record.temperature)
        pressure = _as_float(record.pressure)
        runtime = _as_float(record.runtime_hours)
        cycles = _as_float(record.cycles_since_service)

        matrix.append(
            [
                vibration,
                temperature,
                pressure,
                runtime,
                cycles,
                safe_div(temperature, vibration),
                (100.0 - pressure) if not math.isnan(pressure) else NAN,
                safe_div(runtime, max_runtime),
                roll_values[index],
            ]
        )

    return matrix
