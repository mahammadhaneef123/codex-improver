from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from codex_maint.config import AppConfig
from codex_maint.models.artifacts import load_bundle
from codex_maint.workflow import run_monitoring, run_training

ROOT = Path(__file__).resolve().parents[1]


def small_config(root: Path) -> AppConfig:
    config = AppConfig.load(ROOT / "config" / "dev.toml")
    return replace(
        config,
        data=replace(config.data, n_samples=420, seed=3, missing_rate=0.02),
        model=replace(config.model, epochs=60),
        training=replace(
            config.training,
            artifact_dir=str(root / "artifacts"),
            early_stopping_patience=8,
            batch_size=32,
        ),
        paths={
            "data_dir": str(root / "data"),
            "report_dir": str(root / "reports"),
        },
    )


class WorkflowTests(unittest.TestCase):
    def test_training_then_monitoring_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = small_config(Path(tmp))

            training = run_training(config, root=Path(tmp), tag="unit")
            self.assertEqual(training["stage"], "training")
            self.assertTrue(Path(training["artifact"]).exists())
            self.assertTrue(Path(training["report"]).exists())

            self.assertIn("f1", training["metrics"])
            self.assertIn("roc_auc", training["metrics"])
            self.assertIn(training["health"]["status"], {"healthy", "watch", "degraded", "critical"})
            self.assertGreaterEqual(training["model"]["epochs_run"], 1)
            self.assertEqual(training["dataset"]["test"], 63)

            bundle = load_bundle(training["artifact"])
            self.assertEqual(bundle.metadata["threshold"], training["model"]["threshold"])
            self.assertEqual(len(bundle.pipeline.feature_names), 9)

            monitoring = run_monitoring(
                config,
                artifact_path=training["artifact"],
                root=Path(tmp),
                tag="unit",
                shift=2.5,
                n_samples=240,
            )
            self.assertEqual(monitoring["stage"], "monitoring")
            self.assertEqual(monitoring["dataset"]["live"], 240)
            self.assertTrue(Path(monitoring["report"]).exists())
            self.assertIn(monitoring["drift"]["status"], {"warning", "alert"})
            self.assertGreater(monitoring["drift"]["max_psi"], 0.1)
            self.assertIsInstance(monitoring["maintenance"], list)

            report = Path(monitoring["report"]).read_text(encoding="utf-8")
            self.assertIn("Monitoring Report", report)
            self.assertIn("Recommendations", report)


if __name__ == "__main__":
    unittest.main()
