"""Fitted preprocessing pipeline: impute -> clip -> scale -> reference bins."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Sequence

from ..data.schema import MaintenanceRecord
from ..utils.stats import (
    bin_edges,
    bin_proportions,
    finite,
    mean,
    median,
    quantile,
    stdev,
)
from .extractor import FEATURE_NAMES, raw_feature_matrix

PIPELINE_VERSION = 1


class PipelineError(RuntimeError):
    """Raised when a pipeline is used before fitting or given bad state."""


@dataclass(slots=True)
class FeaturePipeline:
    rolling_window: int = 5
    clip_quantile: float = 0.99
    scale: bool = True
    reference_bins: int = 10
    feature_names: list[str] = field(default_factory=list)
    medians: list[float] = field(default_factory=list)
    lows: list[float] = field(default_factory=list)
    highs: list[float] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    ref_edges: list[list[float]] = field(default_factory=list)
    ref_props: list[list[float]] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def fit(self, records: Sequence[MaintenanceRecord]) -> "FeaturePipeline":
        raw = raw_feature_matrix(records, self.rolling_window)
        if not raw:
            raise PipelineError("cannot fit a pipeline on an empty dataset")

        self.feature_names = list(FEATURE_NAMES)
        columns = [list(column) for column in zip(*raw)]

        self.medians = []
        for column in columns:
            valid = finite(column)
            self.medians.append(median(valid) if valid else 0.0)

        filled = [self._fill(row) for row in raw]
        filled_columns = [list(column) for column in zip(*filled)]

        low_q = (1.0 - self.clip_quantile) / 2.0
        high_q = 1.0 - low_q
        self.lows = [quantile(column, low_q) for column in filled_columns]
        self.highs = [quantile(column, high_q) for column in filled_columns]

        clipped = [self._clip(row) for row in filled]
        clipped_columns = [list(column) for column in zip(*clipped)]

        if self.scale:
            self.means = [mean(column) for column in clipped_columns]
            stds: list[float] = []
            for column in clipped_columns:
                deviation = stdev(column)
                stds.append(deviation if deviation > 0 else 1.0)
            self.stds = stds
        else:
            self.means = [0.0] * len(self.feature_names)
            self.stds = [1.0] * len(self.feature_names)

        scaled = [self._apply_scale(row) for row in clipped]
        scaled_columns = [list(column) for column in zip(*scaled)]

        self.ref_edges = [
            bin_edges(column, self.reference_bins) for column in scaled_columns
        ]
        self.ref_props = [
            bin_proportions(column, edges)
            for column, edges in zip(scaled_columns, self.ref_edges)
        ]

        return self

    def transform(self, records: Sequence[MaintenanceRecord]) -> list[list[float]]:
        if not self.feature_names:
            raise PipelineError("pipeline is not fitted; call fit() first")

        raw = raw_feature_matrix(records, self.rolling_window)
        if raw and len(raw[0]) != self.n_features:
            raise PipelineError(
                f"expected {self.n_features} features, got {len(raw[0])}"
            )
        return [self._apply_scale(self._clip(self._fill(row))) for row in raw]

    def fit_transform(self, records: Sequence[MaintenanceRecord]) -> list[list[float]]:
        return self.fit(records).transform(records)

    def _fill(self, row: Sequence[float]) -> list[float]:
        return [
            self.medians[index] if math.isnan(value) else value
            for index, value in enumerate(row)
        ]

    def _clip(self, row: Sequence[float]) -> list[float]:
        return [
            min(max(value, self.lows[index]), self.highs[index])
            for index, value in enumerate(row)
        ]

    def _apply_scale(self, row: Sequence[float]) -> list[float]:
        return [
            (value - self.means[index]) / self.stds[index]
            for index, value in enumerate(row)
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": PIPELINE_VERSION,
            "rolling_window": self.rolling_window,
            "clip_quantile": self.clip_quantile,
            "scale": self.scale,
            "reference_bins": self.reference_bins,
            "feature_names": list(self.feature_names),
            "medians": list(self.medians),
            "lows": list(self.lows),
            "highs": list(self.highs),
            "means": list(self.means),
            "stds": list(self.stds),
            "ref_edges": [list(edge) for edge in self.ref_edges],
            "ref_props": [list(props) for props in self.ref_props],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FeaturePipeline":
        if int(payload.get("version", 0)) != PIPELINE_VERSION:
            raise PipelineError("unsupported pipeline version")

        pipeline = cls(
            rolling_window=int(payload["rolling_window"]),
            clip_quantile=float(payload["clip_quantile"]),
            scale=bool(payload["scale"]),
            reference_bins=int(payload["reference_bins"]),
        )
        pipeline.feature_names = [str(name) for name in payload["feature_names"]]
        pipeline.medians = [float(v) for v in payload["medians"]]
        pipeline.lows = [float(v) for v in payload["lows"]]
        pipeline.highs = [float(v) for v in payload["highs"]]
        pipeline.means = [float(v) for v in payload["means"]]
        pipeline.stds = [float(v) for v in payload["stds"]]
        pipeline.ref_edges = [
            [float(v) for v in edge] for edge in payload["ref_edges"]
        ]
        pipeline.ref_props = [
            [float(v) for v in props] for props in payload["ref_props"]
        ]

        if len(pipeline.medians) != pipeline.n_features:
            raise PipelineError("corrupt pipeline: imputer length mismatch")
        return pipeline
