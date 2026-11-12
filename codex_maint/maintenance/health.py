"""Model health scoring that blends performance with drift pressure."""

from __future__ import annotations

from typing import Any

from ..config import MaintenanceConfig

STATUS_ORDER = ("healthy", "watch", "degraded", "critical")


def assess(
    metrics: dict[str, float | int],
    drift: dict[str, Any],
    cfg: MaintenanceConfig,
) -> dict[str, Any]:
    """Return a 0-100 health score with status and human-readable reasons."""
    f1 = float(metrics.get("f1", 0.0))
    recall = float(metrics.get("recall", 0.0))
    auc = float(metrics.get("roc_auc", 0.0))
    max_psi = float(drift.get("max_psi", 0.0))

    performance = 0.6 * f1 + 0.4 * recall

    # Drift only starts to penalise once PSI passes the warning band, then
    # ramps linearly to a 50% penalty at the alert threshold.
    if max_psi <= cfg.psi_warn:
        drift_ratio = 0.0
    else:
        span = max(cfg.psi_alert - cfg.psi_warn, 1e-6)
        drift_ratio = min(1.0, (max_psi - cfg.psi_warn) / span)

    score = round(100.0 * performance * (1.0 - 0.5 * drift_ratio), 2)

    reasons: list[str] = []
    if recall < cfg.recall_warn:
        reasons.append(
            f"recall {recall:.3f} is below the {cfg.recall_warn:.2f} floor"
        )
    if auc < 0.7:
        reasons.append(f"ROC AUC {auc:.3f} indicates weak ranking quality")
    if max_psi >= cfg.psi_alert:
        reasons.append(
            f"max PSI {max_psi:.3f} exceeds alert threshold {cfg.psi_alert:.2f}"
        )
    elif max_psi >= cfg.psi_warn:
        reasons.append(
            f"max PSI {max_psi:.3f} exceeds warning threshold {cfg.psi_warn:.2f}"
        )

    status = _status_for(score, max_psi, cfg)
    if not reasons:
        if status == "healthy":
            reasons.append("performance and drift are within expected bounds")
        else:
            reasons.append(
                f"composite health score {score:.2f} is below the 85 healthy target"
            )

    return {
        "score": score,
        "status": status,
        "components": {
            "performance": round(performance, 4),
            "drift_ratio": round(drift_ratio, 4),
            "f1": f1,
            "recall": recall,
            "roc_auc": auc,
            "max_psi": max_psi,
        },
        "reasons": reasons,
    }


def _status_for(score: float, max_psi: float, cfg: MaintenanceConfig) -> str:
    if max_psi >= cfg.psi_alert or score < 50:
        return "critical"
    if max_psi >= cfg.psi_warn or score < 70:
        return "degraded"
    if score < 85:
        return "watch"
    return "healthy"
