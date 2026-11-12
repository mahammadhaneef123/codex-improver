"""Typed application configuration loaded from TOML profiles."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing or contains bad keys."""


@dataclass(slots=True)
class DataConfig:
    n_samples: int = 2000
    seed: int = 42
    train_split: float = 0.7
    val_split: float = 0.15
    missing_rate: float = 0.0


@dataclass(slots=True)
class FeatureConfig:
    rolling_window: int = 5
    clip_quantile: float = 0.99
    scale: bool = True


@dataclass(slots=True)
class ModelConfig:
    name: str = "logistic"
    learning_rate: float = 0.1
    epochs: int = 400
    l2: float = 0.001
    threshold: float = 0.5


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 64
    early_stopping_patience: int = 20
    artifact_dir: str = "artifacts"


@dataclass(slots=True)
class MaintenanceConfig:
    psi_warn: float = 0.1
    psi_alert: float = 0.25
    recall_warn: float = 0.75
    retrain_interval_days: int = 30


@dataclass(slots=True)
class AppConfig:
    profile: str = "dev"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)
    paths: dict[str, str] = field(
        default_factory=lambda: {"data_dir": "data", "report_dir": "reports"}
    )

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        """Load and validate a TOML profile."""
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigError(f"config file not found: {config_path}")

        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        profile = raw.pop("profile", config_path.stem)
        paths = raw.pop("paths", {})

        if not isinstance(paths, dict):
            raise ConfigError("[paths] must be a table of string values")

        config = cls(
            profile=str(profile),
            data=_section(DataConfig, raw.pop("data", {})),
            features=_section(FeatureConfig, raw.pop("features", {})),
            model=_section(ModelConfig, raw.pop("model", {})),
            training=_section(TrainingConfig, raw.pop("training", {})),
            maintenance=_section(MaintenanceConfig, raw.pop("maintenance", {})),
            paths={str(k): str(v) for k, v in paths.items()},
        )

        if raw:
            raise ConfigError(f"unknown config sections: {', '.join(sorted(raw))}")

        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 < self.data.train_split < 1.0:
            raise ConfigError("data.train_split must be between 0 and 1")
        if not 0.0 <= self.data.val_split < 1.0:
            raise ConfigError("data.val_split must be between 0 and 1")
        if self.data.train_split + self.data.val_split >= 1.0:
            raise ConfigError("train_split + val_split must leave room for a test split")
        if self.data.n_samples < 20:
            raise ConfigError("data.n_samples must be at least 20")
        if self.features.rolling_window < 1:
            raise ConfigError("features.rolling_window must be >= 1")
        if not 0.5 <= self.features.clip_quantile < 1.0:
            raise ConfigError("features.clip_quantile must be in [0.5, 1.0)")
        if self.model.epochs < 1:
            raise ConfigError("model.epochs must be >= 1")
        if self.model.learning_rate <= 0:
            raise ConfigError("model.learning_rate must be positive")
        if self.training.batch_size < 1:
            raise ConfigError("training.batch_size must be >= 1")
        if not 0.0 < self.maintenance.psi_warn <= self.maintenance.psi_alert:
            raise ConfigError("maintenance PSI thresholds must satisfy 0 < warn <= alert")
        if self.maintenance.retrain_interval_days < 1:
            raise ConfigError("maintenance.retrain_interval_days must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """Persist the resolved configuration as JSON for reproducibility."""
        from .utils import io

        target = io.ensure_dir(Path(path).parent) / Path(path).name
        io.write_json(target, {"config": self.to_dict()})
        return target


def _section(section_cls: type, raw: dict[str, Any]) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"[{section_cls.__name__}] must be a table")

    allowed = {f.name for f in fields(section_cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(
            f"unknown key(s) in [{section_cls.__name__}]: {', '.join(sorted(unknown))}"
        )

    try:
        return section_cls(**raw)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ConfigError(str(exc)) from exc
