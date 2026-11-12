"""High-level workflows wiring data, model, evaluation and maintenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig
from .data import loader, synthetic
from .evaluation import metrics as metrics_lib
from .evaluation import report as report_lib
from .features.pipeline import FeaturePipeline
from .maintenance import drift as drift_lib
from .maintenance import health as health_lib
from .maintenance import scheduler as scheduler_lib
from .models import artifacts, train as train_lib
from .utils import io
from .utils.timing import Stopwatch


def build_split(cfg: AppConfig) -> loader.DatasetSplit:
    records = synthetic.generate_records(
        cfg.data.n_samples,
        seed=cfg.data.seed,
        missing_rate=cfg.data.missing_rate,
    )
    return loader.split_records(
        records,
        cfg.data.train_split,
        cfg.data.val_split,
        seed=cfg.data.seed,
    )


def _labels(records: list[Any]) -> list[int]:
    return [record.failure for record in records]


def _positive_rate(records: list[Any]) -> float:
    return round(sum(_labels(records)) / len(records), 4) if records else 0.0


def run_training(
    cfg: AppConfig,
    *,
    root: str | Path | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """Fit the model, evaluate on holdout data and persist an artifact."""
    base = Path(root) if root is not None else io.project_root()
    run_tag = tag or io.timestamp_slug()
    created_at = io.utc_now_iso()

    with Stopwatch() as total:
        split = build_split(cfg)

        pipeline = FeaturePipeline(
            rolling_window=cfg.features.rolling_window,
            clip_quantile=cfg.features.clip_quantile,
            scale=cfg.features.scale,
        )
        X_train = pipeline.fit_transform(split.train)
        X_val = pipeline.transform(split.val)
        X_test = pipeline.transform(split.test)
        y_train = _labels(split.train)
        y_val = _labels(split.val)
        y_test = _labels(split.test)

        result = train_lib.train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            cfg.model,
            batch_size=cfg.training.batch_size,
            patience=cfg.training.early_stopping_patience,
            seed=cfg.data.seed,
        )

        test_probabilities = result.model.predict_proba(X_test)
        test_metrics = metrics_lib.classification_report(
            y_test, test_probabilities, result.threshold
        )

        drift = drift_lib.compare_matrix(
            X_test,
            pipeline.ref_edges,
            pipeline.ref_props,
            pipeline.feature_names,
            warn=cfg.maintenance.psi_warn,
            alert=cfg.maintenance.psi_alert,
        )
        health = health_lib.assess(test_metrics, drift, cfg.maintenance)

        asset_risks = scheduler_lib.aggregate_asset_risk(split.test, test_probabilities)
        tasks = scheduler_lib.plan_maintenance(
            asset_risks,
            threshold=result.threshold,
            horizon_days=cfg.maintenance.retrain_interval_days // 2 or 7,
        )

        metadata = {
            "created_at": created_at,
            "profile": cfg.profile,
            "tag": run_tag,
            "model_name": cfg.model.name,
            "threshold": result.threshold,
            "feature_names": list(pipeline.feature_names),
            "epochs_run": len(result.history),
            "train_seconds": result.train_seconds,
            "val_metrics": result.val_metrics,
            "test_metrics": test_metrics,
            "config": cfg.to_dict(),
        }

        artifact = artifacts.Artifact(
            model=result.model,
            pipeline=pipeline,
            metadata=metadata,
        )

        artifact_dir = io.ensure_dir(io.resolve_path(cfg.training.artifact_dir, base))
        artifact_path = artifacts.save_bundle(
            artifact_dir / f"model_{run_tag}.json", artifact
        )
        artifacts.save_bundle(artifact_dir / "latest.json", artifact)
        cfg.save(artifact_dir / f"config_{run_tag}.json")

    summary: dict[str, Any] = {
        "status": "ok",
        "stage": "training",
        "profile": cfg.profile,
        "tag": run_tag,
        "created_at": created_at,
        "artifact": str(artifact_path),
        "dataset": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
            "positive_rate": _positive_rate(split.test),
        },
        "model": {
            "name": cfg.model.name,
            "threshold": result.threshold,
            "epochs_run": len(result.history),
            "train_seconds": result.train_seconds,
        },
        "metrics": test_metrics,
        "val_metrics": result.val_metrics,
        "drift": drift,
        "health": health,
        "maintenance": [task.to_dict() for task in tasks],
        "retrain": {
            "last_trained_at": created_at,
            "next_due": scheduler_lib.next_retrain_date(
                created_at, cfg.maintenance.retrain_interval_days
            ),
            "due": False,
            "interval_days": cfg.maintenance.retrain_interval_days,
        },
        "total_seconds": round(total.elapsed, 4),
    }

    report_path = report_lib.write_markdown(
        io.resolve_path(cfg.paths.get("report_dir", "reports"), base)
        / f"training_{run_tag}.md",
        report_lib.render_report(summary, title="Codex Maintenance Model - Training Report"),
    )
    summary["report"] = str(report_path)
    return summary


def run_monitoring(
    cfg: AppConfig,
    *,
    artifact_path: str | Path | None = None,
    root: str | Path | None = None,
    tag: str | None = None,
    shift: float = 0.35,
    n_samples: int | None = None,
) -> dict[str, Any]:
    """Score a drifted 'live' batch and assess model health."""
    base = Path(root) if root is not None else io.project_root()
    run_tag = tag or io.timestamp_slug()
    created_at = io.utc_now_iso()

    bundle_path = (
        Path(artifact_path)
        if artifact_path is not None
        else base / cfg.training.artifact_dir / "latest.json"
    )
    bundle = artifacts.load_bundle(bundle_path)

    pipeline = bundle.pipeline
    model = bundle.model
    metadata = bundle.metadata
    threshold = float(metadata.get("threshold", cfg.model.threshold))

    sample_size = n_samples or max(200, cfg.data.n_samples // 3)
    live = synthetic.generate_records(
        sample_size,
        seed=cfg.data.seed + 101,
        missing_rate=cfg.data.missing_rate,
        shift=shift,
    )

    with Stopwatch() as total:
        X_live = pipeline.transform(live)
        probabilities = model.predict_proba(X_live)
        live_metrics = metrics_lib.classification_report(
            _labels(live), probabilities, threshold
        )

        drift = drift_lib.compare_matrix(
            X_live,
            pipeline.ref_edges,
            pipeline.ref_props,
            pipeline.feature_names,
            warn=cfg.maintenance.psi_warn,
            alert=cfg.maintenance.psi_alert,
        )
        health = health_lib.assess(live_metrics, drift, cfg.maintenance)

        asset_risks = scheduler_lib.aggregate_asset_risk(live, probabilities)
        tasks = scheduler_lib.plan_maintenance(
            asset_risks,
            threshold=threshold,
            horizon_days=cfg.maintenance.retrain_interval_days // 2 or 7,
        )

    last_trained_at = str(metadata.get("created_at", created_at))
    retrain = {
        "last_trained_at": last_trained_at,
        "next_due": scheduler_lib.next_retrain_date(
            last_trained_at, cfg.maintenance.retrain_interval_days
        ),
        "due": scheduler_lib.retrain_due(
            last_trained_at, cfg.maintenance.retrain_interval_days
        ),
        "interval_days": cfg.maintenance.retrain_interval_days,
    }

    summary: dict[str, Any] = {
        "status": "ok",
        "stage": "monitoring",
        "profile": cfg.profile,
        "tag": run_tag,
        "created_at": created_at,
        "artifact": str(bundle_path),
        "dataset": {
            "live": len(live),
            "shift": shift,
            "positive_rate": _positive_rate(live),
        },
        "model": {
            "name": str(metadata.get("model_name", cfg.model.name)),
            "threshold": threshold,
            "epochs_run": metadata.get("epochs_run", "-"),
            "train_seconds": metadata.get("train_seconds", "-"),
        },
        "metrics": live_metrics,
        "drift": drift,
        "health": health,
        "maintenance": [task.to_dict() for task in tasks],
        "retrain": retrain,
        "total_seconds": round(total.elapsed, 4),
    }

    report_path = report_lib.write_markdown(
        io.resolve_path(cfg.paths.get("report_dir", "reports"), base)
        / f"monitoring_{run_tag}.md",
        report_lib.render_report(summary, title="Codex Maintenance Model - Monitoring Report"),
    )
    summary["report"] = str(report_path)
    return summary


def run_all(
    cfg: AppConfig,
    *,
    root: str | Path | None = None,
    shift: float = 0.35,
) -> dict[str, Any]:
    training = run_training(cfg, root=root)
    monitoring = run_monitoring(
        cfg,
        artifact_path=training["artifact"],
        root=root,
        shift=shift,
    )
    return {"status": "ok", "stage": "run-all", "training": training, "monitoring": monitoring}


def generate_dataset(
    cfg: AppConfig,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    records = synthetic.generate_records(
        cfg.data.n_samples,
        seed=cfg.data.seed,
        missing_rate=cfg.data.missing_rate,
    )
    path = loader.write_csv(
        records,
        io.resolve_path(cfg.paths.get("data_dir", "data"), root)
        / f"maintenance_{cfg.data.seed}.csv",
    )
    return {
        "status": "ok",
        "stage": "generate-data",
        "rows": len(records),
        "path": str(path),
        "positive_rate": _positive_rate(records),
    }
