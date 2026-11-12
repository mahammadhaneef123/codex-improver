# Maintainers

This file covers **this repository** (`opencode` / Codex Maintenance Model
workspace). It does not describe a role in OpenAI's Codex project or any
other upstream repository.

## Primary maintainer (core)

| Area | Owner | Responsibilities |
| --- | --- | --- |
| Model | Haneef Syed | Model design, threshold policy, evaluation quality |
| Data | Haneef Syed | Telemetry schema, splits, missing-data policy |
| Release | Haneef Syed | Release gate approvals, promotion, rollback |
| Monitoring | Haneef Syed | Drift review, health triage, retrain schedule |

The machine-readable source of truth is `config/maintainers.toml`; the CLI
reads it with `python main.py maintain`.

## Reviewers

No second reviewer is configured yet. Production profiles require two
approvals, so adding a reviewer to `config/maintainers.toml` is the first
step before promoting with `--config config/prod.toml`.

## Decision rights

- Only a maintainer listed in `config/maintainers.toml` may run `promote`
  or `rollback`.
- Promotions must pass the release gate in `[governance]` and record the
  required number of approvals plus the acting maintainer in the audit log.
- Any rollback is recorded with actor and reason; see
  `artifacts/audit.jsonl`.
- Retraining cadence is owned by the primary maintainer and reviewed monthly.

## Escalation

1. Primary maintainer (30 minute response target per `maintainers.toml`).
2. If unavailable, the next maintainer in the rotation (`on_call`).
3. If no maintainer is reachable, freeze promotion and keep the last known
   good bundle live rather than shipping an unapproved model.

## Becoming a maintainer

Add a `[[maintainer]]` entry with an explicit area list, then update this
document. Governance changes should be reviewed like code.
