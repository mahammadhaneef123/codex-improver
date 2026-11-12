"""Filesystem and JSON helpers anchored to the project root."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the repository root (the folder containing `codex_maint/`)."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(value: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a possibly-relative path against the project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    base = Path(root) if root is not None else project_root()
    return (base / path).resolve()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
