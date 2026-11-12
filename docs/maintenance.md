# Maintaining the Model

This is the operational runbook for the Codex failure-risk model.

## Cadence

| Trigger | Action |
| --- | --- |
| Each scheduled run | `python main.py monitor` |
| `retrain.due = true` | `python main.py train` and promote the new artifact |
| Drift status `warning` | Inspect the shifted sensors, widen validation, plan a retrain |
| Drift status `alert` | Retrain before the next release; consider rolling back |
| Health `degraded` / `critical` | Review recall and threshold, then retrain |

The retraining interval is `maintenance.retrain_interval_days` in the active
profile (30 days in `dev.toml`, 14 in `prod.toml`).

## Reading a report

`reports/*.md` always contains:

1. **Dataset** - row counts and the positive (failure) rate.
2. **Metrics** - threshold, accuracy, precision, recall, F1, ROC AUC, Brier,
   and the confusion matrix.
3. **Drift vs Training Reference** - per-feature PSI and a status.
4. **Model Health** - 0-100 score, status and the reasons behind it.
5. **Maintenance Queue** - assets above threshold, ordered by due date.
6. **Recommendations** - generated next steps.

### PSI thresholds

| PSI | Status | Meaning |
| --- | --- | --- |
| `< 0.10` | stable | Live data matches training |
| `0.10 - 0.25` | warning | Investigate the shifted feature |
| `>= 0.25` | alert | Retrain before promoting anything |

Thresholds are configurable per profile and are also stored in the artifact
metadata, so an old report can be read with the thresholds it was judged by.

### Health score

`score = 100 * (0.6 * F1 + 0.4 * recall) * (1 - 0.5 * drift_ratio)`

`drift_ratio` stays at `0` while max PSI is at or below `psi_warn`, then ramps
linearly to `1` at `psi_alert`. Drift can therefore remove at most half the
score; performance is the other half. Status cut-offs:

| Score | Status | Action |
| --- | --- | --- |
| `>= 85` | healthy | Keep monitoring |
| `70 - 85` | watch | Watch the next two runs |
| `50 - 70` | degraded | Plan a retrain this cycle |
| `< 50` | critical | Retrain or roll back now |

Any PSI at/above `psi_alert` forces `critical`, and a recall below
`recall_warn` is always listed as a reason.

## Retraining checklist

1. Confirm the drift alert is real (sensor outage vs genuine behaviour change).
2. `python main.py generate-data` if you need a refreshed development set, or
   point `data.loader.read_csv` at the production extract.
3. `python main.py train` with the chosen profile.
4. Compare the new report against the previous one: recall and PSI first.
5. Promote by copying the new bundle over `artifacts/latest.json`.
6. Keep the previous bundle for rollback.
7. Archive the report next to the bundle.

## Rollback

Bundles are immutable JSON files named `model_<tag>.json`. To roll back,
point monitoring at the previous file:

```powershell
python main.py monitor --artifact artifacts/model_<previous-tag>.json
```

Then overwrite `latest.json` with that bundle once verified.

## Common failures

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `pipeline is not fitted` | Calling transform before fit | Fit on train first |
| `artifact not found` | Wrong profile or tag | Check `artifacts/` and `--config` |
| Recall collapses after deploy | Threshold too high for live prevalence | Retune threshold on fresh validation |
| PSI alert on one feature only | Sensor calibration or outage | Inspect telemetry before retraining |
| `config` unknown key error | Typo in TOML | Match keys in `config.py` dataclasses |

## Ownership

- **Model quality:** whoever owns the failure-prediction service.
- **Drift alerts:** on-call data engineer.
- **Sensor data quality:** maintenance/telemetry team.
- **Retraining schedule:** model owner, reviewed monthly.
