"""Failure-risk models and training utilities."""

from __future__ import annotations

from .artifacts import Artifact, ArtifactError, load_bundle, save_bundle
from .base import Classifier, ModelError
from .linear import LogisticRegression
from .registry import available_models, create_model, model_from_dict
from .train import TrainingResult, choose_threshold, train_model

__all__ = [
    "Artifact",
    "ArtifactError",
    "Classifier",
    "LogisticRegression",
    "ModelError",
    "TrainingResult",
    "available_models",
    "choose_threshold",
    "create_model",
    "load_bundle",
    "model_from_dict",
    "save_bundle",
    "train_model",
]
