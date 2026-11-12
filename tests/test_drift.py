from __future__ import annotations

import unittest

from codex_maint.maintenance import drift


class DriftTests(unittest.TestCase):
    def test_identical_distributions_have_near_zero_psi(self) -> None:
        values = [float(i) for i in range(100)]
        self.assertLess(drift.psi_from_values(values, values), 1e-6)

    def test_shifted_distribution_has_higher_psi(self) -> None:
        reference = [float(i) for i in range(200)]
        live = [value + 50.0 for value in reference]
        value = drift.psi_from_values(reference, live)
        self.assertGreater(value, 0.25)

    def test_ks_statistic_bounds(self) -> None:
        reference = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(drift.ks_statistic(reference, reference), 0.0)
        self.assertEqual(drift.ks_statistic(reference, [10.0, 11.0, 12.0, 13.0]), 1.0)

    def test_compare_matrix_reports_every_feature(self) -> None:
        reference = [[float(i), float(i % 5)] for i in range(100)]
        live = [[float(i) + 10.0, float(i % 5)] for i in range(100)]
        report = drift.compare_matrix(
            live,
            [[20.0, 40.0], [1.0, 2.0]],
            [[0.25] * 3, [0.25] * 3],
            ["a", "b"],
            warn=0.1,
            alert=0.25,
        )

        self.assertEqual(len(report["features"]), 2)
        self.assertEqual(report["features"][0]["status"], "alert")
        self.assertEqual(report["status"], "alert")
        self.assertGreater(float(report["max_psi"]), 0.25)

    def test_compare_matrix_flags_stable_data(self) -> None:
        reference = [[float(i)] for i in range(0, 100)]
        live = [[float(i)] for i in range(0, 100)]
        edges = [[float(i) for i in range(10, 100, 10)]]
        props = [[0.1] * 10]
        report = drift.compare_matrix(live, edges, props, ["a"], warn=0.1, alert=0.25)
        self.assertEqual(report["status"], "stable")

    def test_psi_requires_aligned_bins(self) -> None:
        with self.assertRaises(ValueError):
            drift.psi([0.5, 0.5], [1.0])


if __name__ == "__main__":
    unittest.main()
