from __future__ import annotations

import math
import unittest

from codex_maint.data.synthetic import generate_records
from codex_maint.features.extractor import FEATURE_NAMES, raw_feature_matrix, rolling_mean
from codex_maint.features.pipeline import FeaturePipeline, PipelineError


class ExtractorTests(unittest.TestCase):
    def test_rolling_mean_basic(self) -> None:
        self.assertEqual(rolling_mean([1.0, 2.0, 3.0, 4.0], 2), [1.0, 1.5, 2.5, 3.5])

    def test_rolling_mean_ignores_nan(self) -> None:
        values = [1.0, float("nan"), 3.0]
        self.assertEqual(rolling_mean(values, 2), [1.0, 1.0, 2.0])

    def test_matrix_shape_matches_feature_names(self) -> None:
        records = generate_records(30, seed=1)
        matrix = raw_feature_matrix(records, rolling_window=3)
        self.assertEqual(len(matrix), 30)
        self.assertTrue(all(len(row) == len(FEATURE_NAMES) for row in matrix))


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = generate_records(120, seed=3, missing_rate=0.05)

    def test_fit_transform_is_nan_free_and_stable(self) -> None:
        pipeline = FeaturePipeline(rolling_window=4)
        matrix = pipeline.fit_transform(self.records)

        self.assertEqual(len(matrix), len(self.records))
        self.assertEqual(len(matrix[0]), len(FEATURE_NAMES))
        self.assertTrue(
            all(math.isfinite(value) for row in matrix for value in row)
        )

        again = pipeline.transform(self.records)
        self.assertEqual(matrix, again)

    def test_unfitted_pipeline_raises(self) -> None:
        with self.assertRaises(PipelineError):
            FeaturePipeline().transform(self.records)

    def test_serialisation_roundtrip(self) -> None:
        pipeline = FeaturePipeline(rolling_window=2).fit(self.records)
        payload = pipeline.to_dict()
        restored = FeaturePipeline.from_dict(payload)

        self.assertEqual(pipeline.feature_names, restored.feature_names)
        self.assertEqual(pipeline.medians, restored.medians)
        self.assertEqual(
            pipeline.transform(self.records),
            restored.transform(self.records),
        )

    def test_reference_bins_are_present(self) -> None:
        pipeline = FeaturePipeline().fit(self.records)
        self.assertEqual(len(pipeline.ref_edges), len(FEATURE_NAMES))
        self.assertEqual(len(pipeline.ref_props), len(FEATURE_NAMES))
        self.assertAlmostEqual(sum(pipeline.ref_props[0]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
