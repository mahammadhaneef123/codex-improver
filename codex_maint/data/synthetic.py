"""Deterministic synthetic telemetry for developing the maintenance model."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from .schema import MaintenanceRecord

DEFAULT_ASSETS = 25
BASE_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _sigmoid(value: float) -> float:
    if value < -35:
        return 0.0
    if value > 35:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def generate_records(
    n_samples: int = 2000,
    *,
    seed: int = 42,
    missing_rate: float = 0.0,
    shift: float = 0.0,
    assets: int = DEFAULT_ASSETS,
) -> list[MaintenanceRecord]:
    """Generate a labelled dataset with a realistic wear signal.

    `shift` displaces the sensor distributions and is used to simulate
    production drift during monitoring.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if assets < 1:
        raise ValueError("assets must be >= 1")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1)")

    rng = random.Random(seed)
    records: list[MaintenanceRecord] = []

    for index in range(n_samples):
        asset = f"AST-{rng.randrange(assets):03d}"
        runtime = rng.uniform(0.0, 8000.0)
        wear = runtime / 8000.0

        vibration = rng.gauss(2.0 + 6.0 * wear + shift, 0.5)
        temperature = rng.gauss(60.0 + 25.0 * wear + 4.0 * shift, 3.0)
        pressure = rng.gauss(100.0 - 20.0 * wear - 2.0 * shift, 4.0)
        cycles = rng.uniform(0.0, 5000.0)

        logit = (
            -6.5
            + 5.0 * wear
            + 0.9 * (vibration - 4.0)
            - 0.02 * (pressure - 90.0)
            + 0.0004 * cycles
        )
        failure = 1 if rng.random() < _sigmoid(logit) else 0

        timestamp = BASE_TIMESTAMP + timedelta(hours=index)

        record = MaintenanceRecord(
            asset_id=asset,
            timestamp=timestamp.isoformat(),
            vibration=_maybe_missing(rng, vibration, missing_rate),
            temperature=_maybe_missing(rng, temperature, missing_rate),
            pressure=_maybe_missing(rng, pressure, missing_rate),
            runtime_hours=_maybe_missing(rng, runtime, missing_rate),
            cycles_since_service=_maybe_missing(rng, cycles, missing_rate),
            failure=failure,
        )
        records.append(record)

    return records


def _maybe_missing(rng: random.Random, value: float, rate: float) -> float | None:
    return None if rng.random() < rate else value
