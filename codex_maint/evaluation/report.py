"""Markdown reporting for training and monitoring runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import io


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _metric_rows(metrics: dict[str, Any]) -> list[list[object]]:
    keys = (
        "n",
        "positives",
        "positive_rate",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier",
        "tp",
        "tn",
        "fp",
        "fn",
    )
    return [[key, metrics.get(key, "-")] for key in keys]


def _drift_rows(drift: dict[str, Any]) -> list[list[object]]:
    return [
        [feature["name"], feature["psi"], feature["status"]]
        for feature in drift.get("features", [])
    ]


def _health_lines(health: dict[str, Any]) -> list[str]:
    lines = [
        f"**Health score:** {health.get('score', '-')} / 100 "
        f"({health.get('status', 'unknown')})",
        "",
    ]
    for reason in health.get("reasons", []):
        lines.append(f"- {reason}")
    return lines


def _maintenance_lines(tasks: list[dict[str, Any]], limit: int = 12) -> list[str]:
    if not tasks:
        return ["No assets currently exceed the risk threshold."]

    rows = [
        [task["asset_id"], task["risk"], task["due_in_days"], task["action"]]
        for task in tasks[:limit]
    ]
    lines = [_table(["Asset", "Risk", "Due (days)", "Action"], rows)]
    if len(tasks) > limit:
        lines.append("")
        lines.append(f"_{len(tasks) - limit} additional tasks omitted._")
    return lines


def _recommendations(summary: dict[str, Any]) -> list[str]:
    drift = summary.get("drift", {})
    health = summary.get("health", {})
    recommendations: list[str] = []

    if drift.get("status") == "alert":
        recommendations.append(
            "Drift alert: retrain on recent production data before the next release."
        )
    elif drift.get("status") == "warning":
        recommendations.append(
            "Drift warning: widen validation, inspect shifted sensors, and plan a retrain."
        )

    if health.get("status") in ("critical", "degraded"):
        recommendations.append(
            "Model health is below target: review recall, threshold and feature quality."
        )

    retrain = summary.get("retrain", {})
    if retrain.get("due"):
        recommendations.append(
            f"Retraining is overdue (next due {retrain.get('next_due')})."
        )

    if not recommendations:
        recommendations.append(
            "Model is within operating bounds; continue the scheduled monitoring cadence."
        )
    return recommendations


def render_report(summary: dict[str, Any], *, title: str) -> str:
    dataset = summary.get("dataset", {})
    model = summary.get("model", {})
    metrics = summary.get("metrics", {})
    drift = summary.get("drift", {})
    health = summary.get("health", {})

    dataset_rows: list[list[object]] = [
        ["train", dataset.get("train", "-")],
        ["validation", dataset.get("val", "-")],
        ["test", dataset.get("test", "-")],
    ]
    if "live" in dataset:
        dataset_rows.append(["live", dataset["live"]])
    if "shift" in dataset:
        dataset_rows.append(["sensor shift", dataset["shift"]])
    dataset_rows.append(["positive rate", dataset.get("positive_rate", "-")])

    lines = [
        f"# {title}",
        "",
        f"- **Profile:** {summary.get('profile', 'unknown')}",
        f"- **Generated:** {summary.get('created_at', 'unknown')}",
        f"- **Artifact:** `{summary.get('artifact', 'n/a')}`",
        f"- **Model:** {model.get('name', 'n/a')} "
        f"(threshold {model.get('threshold', 'n/a')})",
        "",
        "## Dataset",
        "",
        _table(
            ["Split", "Rows"],
            dataset_rows,
        ),
        "",
        "## Metrics",
        "",
        _table(["Metric", "Value"], _metric_rows(metrics)),
        "",
        "## Drift vs Training Reference",
        "",
        f"Overall status: **{drift.get('status', 'unknown')}** "
        f"(max PSI {drift.get('max_psi', '-')})",
        "",
    ]

    drift_rows = _drift_rows(drift)
    if drift_rows:
        lines.extend([_table(["Feature", "PSI", "Status"], drift_rows), ""])

    lines.extend(
        [
            "## Model Health",
            "",
            *_health_lines(health),
            "",
            "## Maintenance Queue",
            "",
            *_maintenance_lines(summary.get("maintenance", [])),
            "",
            "## Recommendations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in _recommendations(summary))
    lines.append("")
    return "\n".join(lines)


def write_markdown(path: str | Path, text: str) -> Path:
    target = Path(path)
    io.ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
    return target
