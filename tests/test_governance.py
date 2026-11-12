from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codex_maint.config import GovernanceConfig
from codex_maint.maintenance import governance


def policy(**overrides: object) -> GovernanceConfig:
    values: dict[str, object] = {
        "min_recall": 0.7,
        "min_f1": 0.7,
        "max_psi": 0.2,
        "required_approvals": 1,
    }
    values.update(overrides)
    return GovernanceConfig(**values)  # type: ignore[arg-type]


GOOD_METRICS = {"recall": 0.8, "f1": 0.8}
GOOD_DRIFT = {"max_psi": 0.05}


class GateTests(unittest.TestCase):
    def test_gate_passes_when_policy_is_met(self) -> None:
        result = governance.evaluate_gate(
            GOOD_METRICS,
            GOOD_DRIFT,
            policy(),
            approvals=1,
        )
        self.assertTrue(result.approved)
        self.assertEqual(result.failures(), [])

    def test_gate_fails_low_recall(self) -> None:
        result = governance.evaluate_gate(
            {"recall": 0.4, "f1": 0.9},
            GOOD_DRIFT,
            policy(),
            approvals=1,
        )
        self.assertFalse(result.approved)
        self.assertIn("recall", result.failures())

    def test_gate_fails_drift_and_retrain_window(self) -> None:
        result = governance.evaluate_gate(
            GOOD_METRICS,
            {"max_psi": 0.4},
            policy(),
            approvals=2,
            retrain_due=True,
        )
        self.assertFalse(result.approved)
        self.assertIn("drift", result.failures())
        self.assertIn("retrain_window", result.failures())

    def test_gate_requires_approvals(self) -> None:
        result = governance.evaluate_gate(
            GOOD_METRICS,
            GOOD_DRIFT,
            policy(required_approvals=2),
            approvals=1,
        )
        self.assertFalse(result.approved)
        self.assertIn("approvals", result.failures())


class ReleaseTests(unittest.TestCase):
    def test_promote_copies_bundle_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model_a.json"
            target = root / "latest.json"
            audit = root / "audit.jsonl"
            source.write_text('{"version": 1}', encoding="utf-8")

            gate = governance.evaluate_gate(
                GOOD_METRICS, GOOD_DRIFT, policy(), approvals=1
            ).to_dict()
            result = governance.promote(
                source,
                target,
                audit,
                gate=gate,
                approvals=1,
                actor="Haneef Syed",
            )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(target.read_text(encoding="utf-8"), '{"version": 1}')

            events = governance.read_events(audit)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event"], "promote")
            self.assertEqual(events[0]["actor"], "Haneef Syed")

    def test_promote_blocks_failed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.json"
            source.write_text("{}", encoding="utf-8")
            gate = {"approved": False, "required_approvals": 1}
            with self.assertRaises(governance.GovernanceError):
                governance.promote(
                    source,
                    Path(tmp) / "latest.json",
                    Path(tmp) / "audit.jsonl",
                    gate=gate,
                    approvals=1,
                    actor="Haneef Syed",
                )

    def test_promote_requires_actor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.json"
            source.write_text("{}", encoding="utf-8")
            gate = {"approved": True, "required_approvals": 0}
            with self.assertRaises(governance.GovernanceError):
                governance.promote(
                    source,
                    Path(tmp) / "latest.json",
                    Path(tmp) / "audit.jsonl",
                    gate=gate,
                    approvals=0,
                    actor="  ",
                )

    def test_rollback_restores_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root / "model_old.json"
            target = root / "latest.json"
            audit = root / "audit.jsonl"
            previous.write_text('{"old": true}', encoding="utf-8")
            target.write_text('{"new": true}', encoding="utf-8")

            result = governance.rollback(
                previous,
                target,
                audit,
                actor="Haneef Syed",
                reason="drift regression",
            )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(target.read_text(encoding="utf-8"), '{"old": true}')
            events = governance.read_events(audit)
            self.assertEqual(events[-1]["event"], "rollback")
            self.assertEqual(events[-1]["reason"], "drift regression")

    def test_rollback_requires_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.json"
            source.write_text("{}", encoding="utf-8")
            with self.assertRaises(governance.GovernanceError):
                governance.rollback(
                    source,
                    Path(tmp) / "latest.json",
                    Path(tmp) / "audit.jsonl",
                    actor="Haneef Syed",
                    reason="",
                )

    def test_audit_tail_and_missing_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.jsonl"
            self.assertEqual(governance.read_events(audit), [])

            for index in range(3):
                governance.append_event(
                    audit,
                    {"event": f"e{index}", "timestamp": "2026-01-01"},
                )

            self.assertEqual(len(governance.tail_events(audit, limit=2)), 2)
            self.assertEqual(governance.tail_events(audit, limit=2)[-1]["event"], "e2")


if __name__ == "__main__":
    unittest.main()
