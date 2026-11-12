"""Risk-prioritised maintenance queue and retraining cadence."""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ..data.schema import MaintenanceRecord


@dataclass(slots=True)
class MaintenanceTask:
    asset_id: str
    risk: float
    due_in_days: int
    action: str

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "risk": round(self.risk, 4),
            "due_in_days": self.due_in_days,
            "action": self.action,
        }


def aggregate_asset_risk(
    records: Sequence[MaintenanceRecord],
    probabilities: Sequence[float],
) -> dict[str, float]:
    """Highest predicted risk observed per asset."""
    if len(records) != len(probabilities):
        raise ValueError("records and probabilities must align")

    risks: dict[str, float] = {}
    for record, probability in zip(records, probabilities):
        current = risks.get(record.asset_id, 0.0)
        risks[record.asset_id] = max(current, float(probability))
    return risks


def plan_maintenance(
    asset_risks: dict[str, float],
    *,
    threshold: float = 0.5,
    horizon_days: int = 14,
) -> list[MaintenanceTask]:
    """Build a priority queue ordered by due date, then risk."""
    if horizon_days < 0:
        raise ValueError("horizon_days must be >= 0")

    queue: list[tuple[int, float, str, str]] = []
    for asset_id, risk in asset_risks.items():
        if risk < threshold:
            continue
        due = int(round((1.0 - risk) * horizon_days))
        due = max(0, min(horizon_days, due))
        action = _action_for(risk)
        heapq.heappush(queue, (due, -risk, asset_id, action))

    tasks: list[MaintenanceTask] = []
    while queue:
        due, negative_risk, asset_id, action = heapq.heappop(queue)
        tasks.append(
            MaintenanceTask(
                asset_id=asset_id,
                risk=round(-negative_risk, 4),
                due_in_days=due,
                action=action,
            )
        )
    return tasks


def _action_for(risk: float) -> str:
    if risk >= 0.8:
        return "immediate service"
    if risk >= 0.65:
        return "schedule service"
    return "inspect"


def next_retrain_date(last_run: str, interval_days: int) -> str:
    """ISO date when the next retrain is due."""
    if interval_days < 1:
        raise ValueError("interval_days must be >= 1")
    last = datetime.fromisoformat(last_run)
    return (last + timedelta(days=interval_days)).date().isoformat()


def retrain_due(last_run: str, interval_days: int, today: date | None = None) -> bool:
    current = today or date.today()
    due = date.fromisoformat(next_retrain_date(last_run, interval_days))
    return due <= current
