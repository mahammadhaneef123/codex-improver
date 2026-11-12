# Architecture

## Runtime flow

```
config/*.toml
     |
     v
 AppConfig (validated, typed)
     |
     v
 synthetic.generate_records ----> data/loader.split_records
     |                                     |
     |                                     v
     |                            train / val / test
     |                                     |
     +----------> FeaturePipeline.fit_transform(train) -----> transform(val, test)
                                                                    |
                                                                    v
                                                        models.train.train_model
                                                       (fit + threshold tuning)
                                                                    |
                    +-----------------------------------------------+
                    |                        |                      |
                    v                        v                      v
             artifacts.save_bundle    metrics.report        drift.compare_matrix
                    |                        |                      |
                    |                        v                      v
                    |                 health.assess  <----  maintenance queue
                    |                        |
                    +--------> evaluation.report.render_report -> reports/*.md
                    |
                    v
          governance.evaluate_gate (policy + approvals)
                    |
          +---------+----------+
          |                    |
      promote              rollback
          |                    |
          +----> artifacts/latest.json + audit.jsonl
```

## Layers

**Configuration (`config.py`)**
TOML is parsed into dataclasses, unknown keys are rejected, and semantic
checks (split ratios, PSI ordering, epochs, batch size) run before any work
starts. Fail fast beats a half-written artifact.

**Data (`data/`)**
`MaintenanceRecord` is the single row contract. `synthetic.py` produces a
seeded, labelled dataset whose failure probability rises with runtime, heat,
vibration and service cycles, and can inject a sensor `shift` for drift tests.
`loader.py` handles CSV round-trips and deterministic, shuffled splits.

**Features (`features/`)**
`extractor.raw_feature_matrix` builds nine columns:

| Column | Meaning |
| --- | --- |
| `vibration`, `temperature`, `pressure` | Raw sensor readings |
| `runtime_hours`, `cycles_since_service` | Usage counters |
| `temp_vibration_ratio` | Coupled thermal/mechanical stress |
| `pressure_deficit` | Drop from nominal 100 |
| `wear_index` | Runtime normalised by observed maximum |
| `vibration_roll_mean` | Per-asset rolling mean (timestamp ordered) |

`FeaturePipeline` is fitted only on training data and stores medians,
clip bounds, scaling statistics and reference histograms. Storing reference
bins is what makes monitoring possible without keeping the full training set.

**Models (`models/`)**
`Classifier` defines the contract. `LogisticRegression` uses mini-batch
gradient descent with L2, optional class weighting for imbalanced failures,
learning-rate decay and early stopping on validation loss. `registry.py` maps
config names to factories so a new algorithm needs no changes to the workflow.

`train.py` tunes the decision threshold on validation F1 (recall breaks ties),
which matters because the operating point, not just the weights, is a model
artifact.

`artifacts.py` serialises the model, the fitted pipeline and metadata into one
JSON bundle. The bundle is the deployable unit; monitoring never re-fits
preprocessing.

**Evaluation (`evaluation/`)**
Pure-Python confusion matrix, accuracy, precision, recall, F1, rank-based ROC
AUC and Brier score. `report.py` renders the same numbers plus drift, health
and the queue into Markdown.

**Maintenance (`maintenance/`)**
`drift.compare_matrix` compares live feature histograms against the stored
reference bins using PSI; `ks_statistic` is available for numeric comparisons.
`health.assess` blends F1/recall with drift pressure into a 0-100 score and a
status (`healthy` -> `watch` -> `degraded` -> `critical`).
`scheduler.py` turns per-asset risk into a priority queue and computes the next
retraining date.

**Governance (`maintenance/ownership.py`, `maintenance/governance.py`)**
`ownership.load` reads `config/maintainers.toml`, exposes the primary
maintainer, area ownership and a deterministic on-call rotation.
`governance.evaluate_gate` checks an artifact's stored metrics and drift
against the `[governance]` policy plus the approval count and retrain window.
`promote` and `rollback` perform atomic copies of the live bundle and append
every action to an append-only JSONL audit log. Promotion is blocked unless
the gate passed, the approvals are sufficient and an actor is named.

**Orchestration (`workflow.py`, `cli.py`)**
Workflows are plain functions returning JSON-serialisable summaries, so the
CLI, tests and future services share one implementation.

## Design decisions

- **Stdlib only.** No dependency drift while developing the model itself.
- **Train-only fitting.** Every imputation, clip and scaling constant comes
  from the training split; validation and test data are transformed with it.
- **Reference bins over raw rows.** Small artifacts, explicit drift contract.
- **Threshold as an artifact.** Changing the operating point requires no
  retraining, just a new bundle.
- **Governed releases.** A bundle does not become live by existing; it must
  pass a policy gate, carry recorded approvals and be promoted by a named
  actor. Every promotion and rollback is auditable.
- **Summaries as dicts.** Easy to print, assert on, log, or serve later.
