"""The record schema exchanged between ingestion, features and models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SENSOR_FIELDS: tuple[str, ...] = (
    "vibration",
    "temperature",
    "pressure",
    "runtime_hours",
    "cycles_since_service",
)

CSV_HEADERS: tuple[str, ...] = (
    "asset_id",
    "timestamp",
    *SENSOR_FIELDS,
    "failure",
)


@dataclass(slots=True)
class MaintenanceRecord:
    """One telemetry snapshot for one asset."""

    asset_id: str
    timestamp: str
    vibration: float | None = None
    temperature: float | None = None
    pressure: float | None = None
    runtime_hours: float | None = None
    cycles_since_service: float | None = None
    failure: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "MaintenanceRecord":
        def number(key: str) -> float | None:
            value = row.get(key)
            if value is None or value == "" or value == "nan":
                return None
            return float(value)

        return cls(
            asset_id=str(row["asset_id"]),
            timestamp=str(row.get("timestamp", "")),
            vibration=number("vibration"),
            temperature=number("temperature"),
            pressure=number("pressure"),
            runtime_hours=number("runtime_hours"),
            cycles_since_service=number("cycles_since_service"),
            failure=int(float(row.get("failure", 0) or 0)),
        )

    def sensors_present(self) -> int:
        return sum(getattr(self, field) is not None for field in SENSOR_FIELDS)
