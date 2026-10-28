# vm_usage_forecasting

Five-minute VM workload forecasting and analysis using the Microsoft Azure Public Dataset V2 challenge release.

## Installation

Python 3.11+ is supported; the recorded run uses Python 3.13. Exact runtime package versions are listed in `src/results/run_manifest.json`.

```bash
cd vm_usage_forecasting
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

Supply a directory containing:

```text
challenge_data/
  vm_cpu_readings.csv.gz
  vm_metadata.csv.gz
```

Uncompressed `.csv` files are also accepted. Raw data is excluded from version control. Inputs are loaded directly; no merged intermediate dataset is required.

## Run

```bash
python main.py --data-dir /path/to/challenge_data
```

The command writes `predictions.csv` and training-window analysis and validation results under `src/results/`. Default output paths are relative to the checkout; the command can also be invoked by an absolute path from another working directory.

The official release uses 110 training observations and a 24-step, two-hour forecast. Training timestamps are 0–32700 seconds; predictions cover 33000–39900 seconds at 300-second intervals. The prediction file has exactly 24 rows and these columns:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

To generate the retrospective holdout evaluation and plots:

```bash
python main.py --data-dir /path/to/challenge_data --evaluate
```

Predictions and their manifest are saved before holdout scoring. `--evaluate` does not change fitting or model selection.

Optional output locations:

```bash
python main.py --data-dir /path/to/challenge_data --output /tmp/predictions.csv --results-dir /tmp/workload_results
```

`--skip-analysis` omits figures. `--horizon` and `--train-steps` support smaller verification datasets; the official horizon is 24. Training defaults to all grid observations except the final horizon. VM counts and the timestamp origin are read from the data.

## Method

The full five-minute targets are constructed before the chronological split. Active count is the number of distinct VMs with valid average CPU readings, including idle VMs. Aggregate CPU is their unweighted mean percentage.

VMs are segmented by workload category and subscription size (1–3, 4–20, >20), producing nine group slots. Two SARIMA models with fixed orders `(1,0,1) × (2,1,0,12)` and `(1,0,1) × (2,1,0,24)` forecast each occupied group's mean. The combined forecast is:

```text
group_forecast = clip(forecast12 - linear(forecast12) + linear(forecast24), 0, 100)
```

The linear components are estimated from model forecasts. Group forecasts are merged using represented VM counts at the forecast origin. Active VM count uses persistence. Component orders and grouping boundaries are fixed design choices; they are not automatically optimized by validation.

Expanding-window validation uses origins 62 and 86 with 24-step horizons, entirely inside the first 110 observations. Persistence, seasonal-naive-12/24 and standalone SARIMA variants use the same validation windows.

## Data handling

Exact duplicate target readings are collapsed; conflicting VM/timestamp values and duplicate metadata IDs raise errors. Invalid average CPU readings are excluded from the count and mean without interpolation. Whole-timestamp source gaps, missing whole-timestamp training CPU, and missing metadata for an observed VM raise explicit errors.

Missing subscription IDs are treated as singletons; missing or unrecognized workload categories use `Unknown`. Empty groups receive zero weight. Short, constant, incomplete or unsuccessful model fits use logged seasonal-naive/last-valid fallbacks. Audit details and optimizer outcomes are recorded in `src/results/data_audit.json` and `src/results/fit_status.csv`.

## Analysis and results

- [Final report](reports/final_report.md): methodology, baseline comparisons, errors, operational recommendations and limitations.
- [Supporting analysis](reports/analysis_and_experiments.md): jump attribution, metadata relationships, subscription-size rationale, deletion-risk future work and unsuccessful tree experiments.
- [Notebooks](notebooks/README.md): exploration, metadata analysis, training and evaluation.
- [Full group histories](src/results/figures/groups_full_history.png): nine observed group series and final forecasts.
- [Forecast errors](src/results/figures/forecast_errors.png): validation and retrospective errors by forecast lead.

The final-window CPU MAE is 0.0579 percentage points, versus persistence at 0.1468. Training-only validation is less favorable (0.2097 versus 0.2150), and the fixed combination does not outperform every comparator. Detailed results are provided in the report.

The public holdout was inspected during development. Final-window results are retrospective and do not constitute an untouched model-selection test. Generalization requires evaluation on new chronological windows. Complete-trace sampling also limits conclusions about churn and lifetime distributions; equal-VM CPU percentages do not identify physical host capacity.

## Tests

```bash
python -B -m unittest discover -s tests -v
```

Tests cover raw-file loading, duplicate and missing-data policies, group boundaries, holdout independence, arbitrary working directories and renamed checkouts.

## Notebook configuration

```bash
python -m pip install -r requirements-notebooks.txt
export VM_DATA_DIR=/path/to/challenge_data
```

Notebooks default to `challenge_data/` inside the checkout. Repository discovery uses the source structure and is independent of the checkout name. Set `VM_PROJECT_ROOT` when launching from an unrelated directory. Notebook-only dependencies, including the optional tree benchmark, are separate from the prediction requirements.

Dataset source: [Microsoft Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md).
