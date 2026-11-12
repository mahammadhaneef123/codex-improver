"""Release governance: gates, promotion, rollback and an append-only audit log."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import GovernanceConfig
from ..utils import io


class GovernanceError(RuntimeError):
    """Raised when a release action breaks policy."""


@dataclass(slots=True)
class GateCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(slots=True)
class GateResult:
    approved: bool
    checks: list[GateCheck]
    approvals: int
    required_approvals: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "approvals": self.approvals,
            "required_approvals": self.required_approvals,
            "checks": [check.to_dict() for check in self.checks],
        }

    def failures(self) -> list[str]:
        return [check.name for check in self.checks if not check.passed]


def evaluate_gate(
    metrics: dict[str, Any],
    drift: dict[str, Any],
    cfg: GovernanceConfig,
    *,
    approvals: int = 0,
    retrain_due: bool = False,
) -> GateResult:
    """Apply the release policy to a scored artifact."""
    recall = float(metrics.get("recall", 0.0))
    f1 = float(metrics.get("f1", 0.0))
    max_psi = float(drift.get("max_psi", 0.0))

    checks = [
        GateCheck(
            name="recall",
            passed=recall >= cfg.min_recall,
            detail=f"recall {recall:.3f} vs minimum {cfg.min_recall:.3f}",
        ),
        GateCheck(
            name="f1",
            passed=f1 >= cfg.min_f1,
            detail=f"f1 {f1:.3f} vs minimum {cfg.min_f1:.3f}",
        ),
        GateCheck(
            name="drift",
            passed=max_psi <= cfg.max_psi,
            detail=f"max PSI {max_psi:.3f} vs limit {cfg.max_psi:.3f}",
        ),
        GateCheck(
            name="retrain_window",
            passed=not retrain_due,
            detail="retrain overdue" if retrain_due else "retrain window open",
        ),
        GateCheck(
            name="approvals",
            passed=approvals >= cfg.required_approvals,
            detail=(
                f"{approvals} approval(s) vs required {cfg.required_approvals}"
            ),
        ),
    ]

    return GateResult(
        approved=all(check.passed for check in checks),
        checks=checks,
        approvals=approvals,
        required_approvals=cfg.required_approvals,
    )


def promote(
    source: str | Path,
    target: str | Path,
    audit_log: str | Path,
    *,
    gate: dict[str, Any],
    approvals: int,
    actor: str,
) -> dict[str, Any]:
    """Copy an approved bundle to the live path and record the event."""
    if not gate.get("approved"):
        raise GovernanceError("release gate has not passed")
    if approvals < int(gate.get("required_approvals", 0)):
        raise GovernanceError("not enough recorded approvals")
    if not actor.strip():
        raise GovernanceError("promotion requires an acting maintainer")

    source_path = Path(source)
    if not source_path.exists():
        raise GovernanceError(f"source artifact not found: {source_path}")

    target_path = Path(target)
    io.ensure_dir(target_path.parent)
    _atomic_copy(source_path, target_path)

    event = {
        "event": "promote",
        "timestamp": io.utc_now_iso(),
        "actor": actor,
        "source": str(source_path),
        "target": str(target_path),
        "approvals": approvals,
        "gate": gate,
    }
    append_event(audit_log, event)
    return {"status": "ok", "event": event}


def rollback(
    source: str | Path,
    target: str | Path,
    audit_log: str | Path,
    *,
    actor: str,
    reason: str,
) -> dict[str, Any]:
    """Restore a previously promoted bundle and record why."""
    source_path = Path(source)
    target_path = Path(target)

    if not source_path.exists():
        raise GovernanceError(f"rollback source not found: {source_path}")
    if source_path.resolve() == target_path.resolve():
        raise GovernanceError("rollback source is already the live artifact")
    if not reason.strip():
        raise GovernanceError("rollback requires a reason")
    if not actor.strip():
        raise GovernanceError("rollback requires an acting maintainer")

    io.ensure_dir(target_path.parent)
    _atomic_copy(source_path, target_path)

    event = {
        "event": "rollback",
        "timestamp": io.utc_now_iso(),
        "actor": actor,
        "source": str(source_path),
        "target": str(target_path),
        "reason": reason,
    }
    append_event(audit_log, event)
    return {"status": "ok", "event": event}


def append_event(path: str | Path, event: dict[str, Any]) -> Path:
    target = Path(path)
    io.ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    return target


def read_events(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []

    events: list[dict[str, Any]] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise GovernanceError(
                f"corrupt audit log at line {number}: {exc}"
            ) from exc

    if limit is not None and limit >= 0:
        return events[-limit:]
    return events


def tail_events(path: str | Path, limit: int = 5) -> list[dict[str, Any]]:
    return read_events(path, limit=limit)


def _atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
