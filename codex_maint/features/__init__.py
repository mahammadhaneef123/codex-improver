"""Feature engineering for telemetry-driven failure prediction."""

from __future__ import annotations

from .extractor import FEATURE_NAMES, raw_feature_matrix, rolling_mean
from .pipeline import FeaturePipeline, PipelineError

__all__ = [
    "FEATURE_NAMES",
    "FeaturePipeline",
    "PipelineError",
    "raw_feature_matrix",
    "rolling_mean",
]
