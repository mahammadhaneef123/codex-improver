from __future__ import annotations

import unittest
from datetime import date

from codex_maint.data.synthetic import generate_records
from codex_maint.maintenance import scheduler


class SchedulerTests(unittest.TestCase):
    def test_aggregate_uses_max_risk_per_asset(self) -> None:
        records = generate_records(6, seed=5, assets=2)
        risks = scheduler.aggregate_asset_risk(records, [0.1, 0.9, 0.2, 0.7, 0.3, 0.4])
        for record, probability in zip(
            records, [0.1, 0.9, 0.2, 0.7, 0.3, 0.4]
        ):
            self.assertGreaterEqual(risks[record.asset_id], probability)

    def test_plan_filters_and_orders_tasks(self) -> None:
        tasks = scheduler.plan_maintenance(
            {"A": 0.9, "B": 0.55, "C": 0.3, "D": 0.75},
            threshold=0.5,
            horizon_days=10,
        )
        self.assertEqual([task.asset_id for task in tasks], ["A", "D", "B"])
        self.assertEqual(tasks[0].action, "immediate service")
        self.assertEqual(tasks[-1].due_in_days, 4)
        self.assertTrue(all(0 <= task.due_in_days <= 10 for task in tasks))

    def test_retrain_schedule(self) -> None:
        self.assertEqual(
            scheduler.next_retrain_date("2026-01-01T00:00:00+00:00", 30),
            "2026-01-31",
        )
        self.assertTrue(
            scheduler.retrain_due(
                "2026-01-01T00:00:00+00:00",
                30,
                today=date(2026, 2, 10),
            )
        )
        self.assertFalse(
            scheduler.retrain_due(
                "2026-03-01T00:00:00+00:00",
                30,
                today=date(2026, 3, 10),
            )
        )


if __name__ == "__main__":
    unittest.main()
