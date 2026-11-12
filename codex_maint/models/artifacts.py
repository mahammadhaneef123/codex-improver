"""Model bundle persistence (weights + preprocessing + metadata)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..features.pipeline import FeaturePipeline
from ..utils import io
from . import registry

BUNDLE_VERSION = 1


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing or malformed."""


@dataclass(slots=True)
class Artifact:
    model: Any
    pipeline: FeaturePipeline
    metadata: dict[str, Any]


def save_bundle(path: str | Path, artifact: Artifact) -> Path:
    payload = {
        "version": BUNDLE_VERSION,
        "model": artifact.model.to_dict(),
        "pipeline": artifact.pipeline.to_dict(),
        "metadata": artifact.metadata,
    }
    return io.write_json(path, payload)


def load_bundle(path: str | Path) -> Artifact:
    source = Path(path)
    if not source.exists():
        raise ArtifactError(f"artifact not found: {source}")

    payload = io.read_json(source)
    if int(payload.get("version", 0)) != BUNDLE_VERSION:
        raise ArtifactError("unsupported artifact version")

    model = registry.model_from_dict(payload["model"])
    pipeline = FeaturePipeline.from_dict(payload["pipeline"])
    metadata = payload.get("metadata", {})

    return Artifact(model=model, pipeline=pipeline, metadata=metadata)
