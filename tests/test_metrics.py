from __future__ import annotations

import unittest

from codex_maint.evaluation import metrics


class MetricsTests(unittest.TestCase):
    def test_confusion_matrix_counts(self) -> None:
        matrix = metrics.confusion_matrix(
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0],
        )
        self.assertEqual(
            (matrix.tp, matrix.tn, matrix.fp, matrix.fn),
            (2, 1, 1, 1),
        )
        self.assertEqual(matrix.total, 5)

    def test_perfect_ranking_auc(self) -> None:
        self.assertAlmostEqual(
            metrics.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
            1.0,
        )

    def test_auc_handles_ties_and_single_class(self) -> None:
        self.assertAlmostEqual(metrics.roc_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertAlmostEqual(metrics.roc_auc([0, 0], [0.2, 0.4]), 0.5)

    def test_report_contains_core_metrics(self) -> None:
        report = metrics.classification_report(
            [0, 1, 1, 0],
            [0.2, 0.7, 0.8, 0.4],
            threshold=0.5,
        )
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["positives"], 2)
        self.assertAlmostEqual(float(report["accuracy"]), 1.0)
        self.assertAlmostEqual(float(report["precision"]), 1.0)
        self.assertAlmostEqual(float(report["recall"]), 1.0)
        self.assertAlmostEqual(float(report["f1"]), 1.0)
        self.assertIn("roc_auc", report)
        self.assertIn("brier", report)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            metrics.confusion_matrix([1], [1, 0])


if __name__ == "__main__":
    unittest.main()
