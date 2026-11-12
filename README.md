# Codex Maintenance Model

A dependency-free Python workspace for developing, evaluating and **keeping**
a predictive-maintenance model for Codex hardware healthy over time.

The model predicts the probability that an asset will fail, tunes its decision
threshold for recall, and ships with the parts a real model needs after launch:
drift detection, health scoring, a risk-prioritised maintenance queue and a
retraining cadence.

## Why this repo exists

A model is only useful while it stays accurate. This workspace treats
maintenance as a first-class feature:

| Concern | Module | What it does |
| --- | --- | --- |
| Data contracts | `codex_maint/data` | Schema, synthetic telemetry, CSV I/O, splits |
| Features | `codex_maint/features` | Rolling means, wear index, impute/clip/scale pipeline |
| Model | `codex_maint/models` | Logistic-regression classifier, registry, artifacts |
| Evaluation | `codex_maint/evaluation` | Confusion matrix, F1, ROC AUC, Brier, reports |
| Maintenance | `codex_maint/maintenance` | PSI/KS drift, health score, retrain schedule |
| Orchestration | `codex_maint/workflow.py`, `cli.py` | End-to-end runs and a CLI |

Everything is pure standard library (Python 3.11+) so the workspace runs
without installing a single package.

## Project layout

```
opencode/
├── main.py                     # `python main.py <command>` entry point
├── pyproject.toml              # packaging + console script metadata
├── requirements.txt            # no third-party runtime deps
├── config/
│   ├── dev.toml                # fast local profile
│   └── prod.toml               # stricter production profile
├── codex_maint/
│   ├── __init__.py
│   ├── cli.py                  # argparse commands
│   ├── config.py               # typed TOML config + validation
│   ├── logging_utils.py
│   ├── workflow.py             # train / monitor / run-all flows
│   ├── data/
│   │   ├── loader.py           # CSV read/write + deterministic split
│   │   ├── schema.py           # MaintenanceRecord dataclass
│   │   └── synthetic.py        # seeded telemetry generator
│   ├── features/
│   │   ├── extractor.py        # raw + derived + rolling features
│   │   └── pipeline.py         # fitted imputer/clipper/scaler + ref bins
│   ├── models/
│   │   ├── base.py             # Classifier interface
│   │   ├── linear.py           # logistic regression (mini-batch GD)
│   │   ├── registry.py         # config name -> factory
│   │   ├── train.py            # fit + threshold tuning
│   │   └── artifacts.py        # bundle save/load
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── report.py           # Markdown reports
│   ├── maintenance/
│   │   ├── drift.py            # PSI + KS
│   │   ├── health.py           # blended health score
│   │   └── scheduler.py        # maintenance queue + retrain dates
│   └── utils/
│       ├── io.py
│       ├── stats.py
│       └── timing.py
├── scripts/
│   ├── bootstrap_data.py
│   ├── run_training.py
│   └── run_tests.py
├── tests/
│   ├── test_config.py
│   ├── test_drift.py
│   ├── test_features.py
│   ├── test_metrics.py
│   ├── test_scheduler.py
│   └── test_workflow.py
└── docs/
    ├── architecture.md
    └── maintenance.md
```

Generated at runtime (git-ignored): `data/`, `artifacts/`, `reports/`.

## Quickstart

```powershell
# from the opencode folder
python main.py models            # list model factories
python main.py generate-data     # write data/maintenance_7.csv
python main.py train             # artifact + reports/training_*.md
python main.py monitor           # score drifted live data
python main.py run-all           # both, in order
python main.py report            # list generated reports
```

Use the production profile with `--config`:

```powershell
python main.py --config config/prod.toml train
```

Run the tests:

```powershell
python scripts/run_tests.py
# or
python -m unittest discover -s tests -t . -v
```

## Outputs

`train` writes three things:

1. `artifacts/model_<tag>.json` - model weights, preprocessing state, metadata.
2. `artifacts/latest.json` - the most recent bundle for monitoring.
3. `reports/training_<tag>.md` - metrics, drift table, health, queue.

`monitor` loads `latest.json`, scores a drifted batch and writes
`reports/monitoring_<tag>.md`.

## Extending the model

1. Add a factory in `codex_maint/models/registry.py`.
2. Implement `Classifier` (`fit`, `predict_proba`, `to_dict`).
3. Add a test under `tests/` and select it in TOML with `name = "..."`.

The feature contract lives in `codex_maint/features/extractor.py`; keep
`FEATURE_NAMES` and the matrix builder in sync when adding a signal.
