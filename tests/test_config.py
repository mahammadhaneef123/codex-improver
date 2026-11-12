from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from codex_maint.config import AppConfig, ConfigError

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_dev_profile_loads(self) -> None:
        config = AppConfig.load(ROOT / "config" / "dev.toml")
        self.assertEqual(config.profile, "dev")
        self.assertEqual(config.data.n_samples, 2500)
        self.assertEqual(config.model.name, "logistic_balanced")
        self.assertIn("report_dir", config.paths)

    def test_unknown_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.toml"
            path.write_text("[model]\nunknown_option = 1\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                AppConfig.load(path)

    def test_invalid_split_is_rejected(self) -> None:
        config = AppConfig.load(ROOT / "config" / "dev.toml")
        broken = replace(config, data=replace(config.data, train_split=0.95, val_split=0.1))
        with self.assertRaises(ConfigError):
            broken.validate()

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(ConfigError):
            AppConfig.load(ROOT / "config" / "does-not-exist.toml")


if __name__ == "__main__":
    unittest.main()
