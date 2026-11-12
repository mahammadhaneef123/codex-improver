from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from codex_maint.maintenance import ownership

ROOT = Path(__file__).resolve().parents[1]


class OwnershipTests(unittest.TestCase):
    def test_registry_loads_primary_maintainer(self) -> None:
        registry = ownership.load(ROOT / "config" / "maintainers.toml")
        self.assertEqual(registry.primary.handle, "Haneef Syed")
        self.assertIn("release", registry.primary.areas)
        self.assertEqual(registry.primary.role, "core maintainer")

    def test_on_call_rotates_between_maintainers(self) -> None:
        registry = ownership.Ownership(
            maintainers=[
                ownership.Maintainer(handle="A", areas=("model",)),
                ownership.Maintainer(handle="B", areas=("data",)),
            ],
            rotation_days=1,
        )
        first = registry.on_call(date(2026, 1, 1))
        second = registry.on_call(date(2026, 1, 2))
        self.assertNotEqual(first.handle, second.handle)

    def test_owners_for_area(self) -> None:
        registry = ownership.Ownership(
            maintainers=[
                ownership.Maintainer(handle="A", areas=("model", "release")),
                ownership.Maintainer(handle="B", areas=("data",)),
            ]
        )
        self.assertEqual([m.handle for m in registry.owners_for("model")], ["A"])
        self.assertEqual([m.handle for m in registry.owners_for("docs")], [])

    def test_single_maintainer_is_always_on_call(self) -> None:
        registry = ownership.Ownership(
            maintainers=[ownership.Maintainer(handle="A")],
            rotation_days=3,
        )
        self.assertEqual(registry.on_call(date(2026, 5, 1)).handle, "A")

    def test_missing_registry_raises(self) -> None:
        with self.assertRaises(ownership.OwnershipError):
            ownership.load(ROOT / "config" / "missing-maintainers.toml")

    def test_empty_registry_is_invalid(self) -> None:
        with self.assertRaises(ownership.OwnershipError):
            ownership.Ownership(maintainers=[])

    def test_bad_registry_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "maintainers.toml"
            path.write_text("rotation_days = 7\n", encoding="utf-8")
            with self.assertRaises(ownership.OwnershipError):
                ownership.load(path)


if __name__ == "__main__":
    unittest.main()
