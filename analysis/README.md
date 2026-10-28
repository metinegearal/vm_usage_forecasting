# Workload analysis

`src/workload/analysis.py` generates training-window analysis as part of `main.py`. The four notebooks provide the corresponding interactive exploration and experiment workflow.

| Notebook | Scope |
| --- | --- |
| [explore.ipynb](../notebooks/explore.ipynb) | Cadence, coverage, missingness, distributions, temporal patterns and jump attribution |
| [transform.ipynb](../notebooks/transform.ipynb) | Metadata relationships, subscription-size groups, resource interactions and recorded deletion timing |
| [train.ipynb](../notebooks/train.ipynb) | Raw-data forecasting, chronological validation, baseline comparisons and final predictions |
| [evaluate.ipynb](../notebooks/evaluate.ipynb) | Forecast errors and optional reproduction of RF/GBM/GAM benchmarks |

Reading-based summaries use training observations. Source metadata CPU statistics and deletion times are descriptive only, because their availability at a forecast origin is not guaranteed. They do not enter the forecasting model. Workload labels do not establish prices or service guarantees, and source lifecycle boundaries may be censored.

The generated findings cover seven areas: cadence and coverage; missingness, duplicates and irregularities; lifetimes and churn; CPU distributions; temporal patterns; workload segments; and peak/low periods. Compact CSV tables and PNG figures accompany the analysis.

The [final report](../reports/final_report.md) distinguishes training-only validation from retrospective public-window results. The [supporting report](../reports/analysis_and_experiments.md) documents metadata associations, grouping rationale, jump attribution, failed experiments and future work.
