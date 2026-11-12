"""Maintainer ownership registry and on-call rotation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

DEFAULT_ESCALATION_MINUTES = 60


class OwnershipError(RuntimeError):
    """Raised when the maintainer registry is missing or invalid."""


@dataclass(slots=True)
class Maintainer:
    handle: str
    role: str = "maintainer"
    areas: tuple[str, ...] = ()
    escalation_minutes: int = DEFAULT_ESCALATION_MINUTES

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "role": self.role,
            "areas": list(self.areas),
            "escalation_minutes": self.escalation_minutes,
        }

    def owns(self, area: str) -> bool:
        return area in self.areas


@dataclass(slots=True)
class Ownership:
    maintainers: list[Maintainer]
    rotation_days: int = 7

    def __post_init__(self) -> None:
        if not self.maintainers:
            raise OwnershipError("at least one maintainer is required")
        if self.rotation_days < 1:
            raise OwnershipError("rotation_days must be >= 1")

    @property
    def primary(self) -> Maintainer:
        return self.maintainers[0]

    def on_call(self, today: date | None = None) -> Maintainer:
        """Deterministic rotation across the maintainer list."""
        if len(self.maintainers) == 1:
            return self.maintainers[0]

        current = today or date.today()
        slot = current.toordinal() // self.rotation_days
        return self.maintainers[slot % len(self.maintainers)]

    def owners_for(self, area: str) -> list[Maintainer]:
        return [maintainer for maintainer in self.maintainers if maintainer.owns(area)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.to_dict(),
            "rotation_days": self.rotation_days,
            "maintainers": [maintainer.to_dict() for maintainer in self.maintainers],
        }


def load(path: str | Path) -> Ownership:
    source = Path(path)
    if not source.exists():
        raise OwnershipError(f"maintainer registry not found: {source}")

    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise OwnershipError(f"invalid maintainer registry: {exc}") from exc

    entries = raw.get("maintainer", [])
    if not isinstance(entries, list) or not entries:
        raise OwnershipError("registry needs at least one [[maintainer]] entry")

    maintainers: list[Maintainer] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("handle"):
            raise OwnershipError("each maintainer needs a handle")

        areas = entry.get("areas", [])
        if not isinstance(areas, list):
            raise OwnershipError("maintainer.areas must be a list of strings")

        maintainers.append(
            Maintainer(
                handle=str(entry["handle"]),
                role=str(entry.get("role", "maintainer")),
                areas=tuple(str(area) for area in areas),
                escalation_minutes=int(
                    entry.get("escalation_minutes", DEFAULT_ESCALATION_MINUTES)
                ),
            )
        )

    return Ownership(
        maintainers=maintainers,
        rotation_days=int(raw.get("rotation_days", 7)),
    )
