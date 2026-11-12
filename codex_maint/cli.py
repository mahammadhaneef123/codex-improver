"""Command-line entry point for the Codex maintenance workspace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import AppConfig, ConfigError
from .evaluation import report as report_lib
from .features.pipeline import PipelineError
from .logging_utils import configure, get_logger
from .maintenance.governance import GovernanceError
from .maintenance.ownership import OwnershipError
from .models import available_models
from .models.artifacts import ArtifactError
from .models.base import ModelError
from .utils import io
from . import workflow

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex-maint",
        description="Develop, evaluate and maintain the Codex failure-risk model.",
    )
    parser.add_argument(
        "--config",
        default="config/dev.toml",
        help="path to a TOML profile (default: config/dev.toml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--version", action="version", version=__version__)

    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-data", help="write a synthetic dataset to CSV")
    generate.add_argument("--out", default=None, help="optional output directory")

    train = subparsers.add_parser("train", help="train and persist a model artifact")
    train.add_argument("--tag", default=None, help="artifact/report tag")

    monitor = subparsers.add_parser("monitor", help="score a drifted live batch")
    monitor.add_argument("--artifact", default=None, help="artifact path override")
    monitor.add_argument("--shift", type=float, default=0.35, help="simulated sensor shift")
    monitor.add_argument("--samples", type=int, default=None, help="live batch size")
    monitor.add_argument("--tag", default=None)

    subparsers.add_parser("run-all", help="train, then monitor the fresh artifact")
    subparsers.add_parser("report", help="locate the latest generated report")
    subparsers.add_parser("models", help="list registered model factories")

    subparsers.add_parser(
        "maintain",
        help="show maintainer ownership, release gate and recent releases",
    )

    promote = subparsers.add_parser(
        "promote",
        help="promote an artifact to latest.json once the gate passes",
    )
    promote.add_argument("--artifact", required=True, help="artifact bundle path")
    promote.add_argument("--approvals", type=int, default=0)
    promote.add_argument("--actor", default="maintainer")

    rollback = subparsers.add_parser(
        "rollback",
        help="restore a previous artifact to latest.json",
    )
    rollback.add_argument("--source", required=True, help="previous bundle path")
    rollback.add_argument("--actor", default="maintainer")
    rollback.add_argument("--reason", required=True)

    audit = subparsers.add_parser("audit", help="show the release audit log")
    audit.add_argument("--limit", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure(args.log_level)

    if args.command == "models":
        _emit({"models": available_models()})
        return 0

    try:
        cfg = _load_config(args.config)
        result = _dispatch(args, cfg)
    except (
        ConfigError,
        PipelineError,
        ModelError,
        ArtifactError,
        GovernanceError,
        OwnershipError,
        ValueError,
        OSError,
    ) as exc:
        logger.error("%s", exc)
        return 1

    _emit(result)
    return 0


def _load_config(value: str) -> AppConfig:
    path = Path(value)
    if not path.is_absolute() and not path.exists():
        path = io.project_root() / value
    return AppConfig.load(path)


def _dispatch(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    root = io.project_root()

    if args.command == "generate-data":
        if args.out:
            cfg = _with_data_dir(cfg, args.out)
        return workflow.generate_dataset(cfg, root=root)

    if args.command == "train":
        return workflow.run_training(cfg, root=root, tag=args.tag)

    if args.command == "monitor":
        return workflow.run_monitoring(
            cfg,
            artifact_path=args.artifact,
            root=root,
            tag=args.tag,
            shift=args.shift,
            n_samples=args.samples,
        )

    if args.command == "run-all":
        return workflow.run_all(cfg, root=root)

    if args.command == "report":
        return _latest_report(cfg, root)

    if args.command == "maintain":
        return workflow.run_maintain(cfg, root=root)

    if args.command == "promote":
        return workflow.run_promote(
            cfg,
            artifact=args.artifact,
            approvals=args.approvals,
            actor=args.actor,
            root=root,
        )

    if args.command == "rollback":
        return workflow.run_rollback(
            cfg,
            source=args.source,
            actor=args.actor,
            reason=args.reason,
            root=root,
        )

    if args.command == "audit":
        return workflow.run_audit(cfg, limit=args.limit, root=root)

    raise ConfigError(f"unsupported command: {args.command}")


def _with_data_dir(cfg: AppConfig, value: str) -> AppConfig:
    cfg.paths["data_dir"] = value
    return cfg


def _latest_report(cfg: AppConfig, root: Path) -> dict[str, Any]:
    directory = io.resolve_path(cfg.paths.get("report_dir", "reports"), root)
    if not directory.exists():
        raise ConfigError(f"no reports found in {directory}")

    reports = sorted(directory.glob("*.md"))
    if not reports:
        raise ConfigError(f"no reports found in {directory}")

    return {
        "status": "ok",
        "stage": "report",
        "latest": str(reports[-1]),
        "reports": [str(path) for path in reports],
    }


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
