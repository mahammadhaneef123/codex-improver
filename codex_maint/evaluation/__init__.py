"""Evaluation metrics and report rendering."""

from __future__ import annotations

from . import metrics, report
from .metrics import (
    ConfusionMatrix,
    brier_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc,
)
from .report import render_report, write_markdown

__all__ = [
    "ConfusionMatrix",
    "brier_score",
    "classification_report",
    "confusion_matrix",
    "f1_score",
    "metrics",
    "render_report",
    "report",
    "roc_auc",
    "write_markdown",
]
