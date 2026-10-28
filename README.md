# vm_usage_forecasting

Azure VM workload analysis and forecasting.

This folder is the complete submission. It reads the **two original release files**; no merged CSV, notebook kernel, parent-project code or experiment folders are needed.

## Install

Python 3.11+ is recommended; the supplied run records its exact package versions in `src/results/run_manifest.json`.

```bash
cd vm_usage_forecasting
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the release files in any directory:

```text
/path/to/challenge_data/
  vm_cpu_readings.csv.gz
  vm_metadata.csv.gz
```

Uncompressed `.csv` inputs work too. Raw data and virtual environments are intentionally excluded from this submission.

## Generate predictions

From inside this folder, run:

```bash
python main.py --data-dir /path/to/challenge_data
```

Pass the directory containing the two raw files; it can be inside or outside the repository. This command writes `predictions.csv` and training-only analysis/validation under `src/results/`. The output defaults are relative to this submission folder, not the caller's working directory. An absolute path to `main.py` also works from another directory.

The required prediction file contains exactly:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

On the official release it has 24 rows at timestamps **33000, 33300, ..., 39900**, in seconds. Training uses the first 110 observations, through 32700 seconds. Each step is five minutes; the 24-step forecast is two hours.

To reproduce the retrospective holdout metrics and plots included with the report:

```bash
python main.py --data-dir /path/to/challenge_data --evaluate
```

Observed holdout CPU is scored/plotted only after the submission and run manifest have been saved. `--evaluate` never changes model fitting or the forecast recipe.

Optional paths and smoke-test controls:

```bash
python main.py --data-dir /path/to/challenge_data --output /tmp/predictions.csv --results-dir /tmp/workload_results
```

`--skip-analysis` skips figures without changing forecasts. `--horizon` and `--train-steps` support internal smoke data; the official horizon remains 24. Training defaults to all grid observations except the final horizon, yielding 110/24 for the supplied release. There is no hard-coded VM count, timestamp origin, fixed prediction table or requirement that all nine groups are populated.

## Fixed forecasting recipe

1. Read usage readings and metadata directly; construct the full five-minute targets before splitting.
2. Count distinct valid VM readings per timestamp. Average VM-level `avg_cpu` equally, in percent.
3. Calculate each subscription's VM count from the supplied metadata. Combine `vm_category` with subscription-size bins **1–3, 4–20, >20**, creating nine group slots.
4. Fit SARIMA `(1,0,1) × (2,1,0,12)` and `(1,0,1) × (2,1,0,24)` to each occupied group's training mean.
5. Compute `forecast12 - linear(forecast12) + linear(forecast24)` and clip to `[0,100]`. The lines are fitted to model forecasts, not observed holdout values.
6. Merge forecasts using the counts of valid represented VMs at the forecast origin. Predict VM count by persistence.

The original notebook did not save the SARIMA-12 fit definition. Its nonseasonal and seasonal component orders were explicitly assumed to match the saved SARIMA-24 specification, changing only the period. This assumption is preserved rather than silently claiming an exact reconstruction of a missing model.

The recipe is fixed by candidate choice. Validation does **not** automatically select a different method. Persistence and seasonal-naive baselines, and both standalone SARIMA variants, are evaluated on the same expanding origins 62 and 86, each forecasting 24 points inside the 110-point training window.

## Missing data and small inputs

Exact duplicate target readings are collapsed; conflicting `(vm_id,timestamp)` values and duplicate metadata IDs raise errors. Invalid/missing average CPU is excluded from the mean and represented count, without interpolation. Complete timestamp gaps and missing whole-timestamp training CPU cause a clear error after saving the target/audit files. Missing metadata for an observed VM also raises an error.

Missing subscription IDs are treated as singletons. Missing/unrecognized workload categories use `Unknown`. Empty group slots receive zero weight. When a group has fewer than 60 training observations, constant data, an incomplete CPU history, or an unsuccessful fit, a logged seasonal-naive/last-valid fallback is used. Warnings and actual optimizer outcomes appear in `src/results/fit_status.csv`. The official-data run should be checked for fallback counts; convergence alone does not establish model adequacy.

## Contents and interpretation

- `src/workload/data.py`: raw loading, target construction, groups and submission checks.
- `src/workload/model.py`: the fixed recipe and explicit numerical/small-data fallbacks.
- `src/workload/evaluation.py`: temporal evaluation and a validation-derived capacity buffer.
- `src/workload/analysis.py`: seven workload analyses and graphs.
- `analysis/README.md`: cleaned notebook ideas, interpretation and discarded leakage-prone steps.
- `reports/final_report.md`: findings, baseline comparison, limitations and recommendation.
- `src/results/`: small generated tables, figures, audit, optimizer log and reproduction manifest.
- `tests/`: data integrity, timestamp handling, leakage boundaries and standalone CLI checks.

`src/results/figures/groups_full_history.png` shows the nine groups from hour zero, with the training/holdout boundary and standalone SARIMA-12/24 forecasts. Reported fleet errors are not per-VM prediction errors. These forecasts do not infer exact physical core demand or host headroom from bucketed metadata.

## Tests

```bash
python -B -m unittest discover -s tests -v
```

The tests include an arbitrary-working-directory CLI run using only two tiny raw files, a nonzero timestamp origin, missing/empty group cases, and changed holdout targets/metadata summaries.

## Evaluation provenance

The public final 24 observations were inspected during iterative notebook/model exploration. Their final scores are therefore **retrospective**, not an untouched independent estimate of model-selection performance. Keeping the recipe fixed now and testing that runtime fitting is prefix-only does not undo that history. Training-only validation is reported alongside holdout results, including contrary evidence from alternatives; no claim is made that the chosen recipe wins validation.

Dataset source: [Microsoft Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md). The challenge's curated complete-trace population limits conclusions about churn and generalization.

## Cleaned notebooks

The four runnable notebooks are in [notebooks](notebooks/README.md). They retain jump, metadata, forecasting and failed-experiment analysis. Install `requirements-notebooks.txt` only for notebook/optional tree reproduction.

## Repository location and notebook data

The checkout may be named `vm_usage_forecasting` or any other name. Main derives source/output paths from its own file location; no parent project is required.

Notebook setup discovers the repository by its `main.py` and `src/workload/data.py` files. It works from the repository root, `notebooks/`, or a workspace containing exactly one matching checkout. For other launch locations, set `VM_PROJECT_ROOT` to the checkout path.

Notebooks default to raw files in `challenge_data/` inside the checkout. To use an external data directory, set an absolute path before launching Jupyter:

```bash
export VM_DATA_DIR=/path/to/challenge_data
```

`VM_PROJECT_ROOT` and `VM_DATA_DIR` configure locations; neither depends on the Git repository name. They are not needed by the main CLI, which takes `--data-dir` explicitly.
