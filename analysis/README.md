# Cleaned notebook ideas

The executable analysis is `src/workload/analysis.py`, called by `main.py`. Read `src/results/analysis_findings.md` and the final report alongside its figures and CSV summaries. Workload statistics use training readings; the full raw input is audited for integrity before splitting.

| Original work | Retained implementation and interpretation |
| --- | --- |
| `explore.ipynb`: source schema, time coverage, distributions | Full raw audit; 300-second grid; reading and per-VM CPU histograms; idle/high fractions |
| `explore.ipynb`: creation/deletion and workload labels | Observed membership changes; descriptive lifecycle summaries with trace-boundary/censoring caveats; category comparisons |
| `transform.ipynb`: shared subscription/deployment IDs | Subscription population size is calculated directly from metadata. Only inclusive bins 1–3, 4–20, >20 are needed by the final model. |
| `train.ipynb`: SARIMA and trend replacement | Explicit period-12 and period-24 specifications; fixed `p12 - line(p12) + line(p24)`; no hidden notebook globals |
| Later group comparisons | Fit the group's own mean, then count-weight its forecast. Do not assign one fleet forecast to every VM and mislabel the resulting row errors as fleet errors. |
| `evaluate.ipynb` | Empty in the working source; replaced with explicit temporal evaluation and report tables. |

## Corrections made for delivery

- The notebook's `109 * 3600` cutoff was incompatible with five-minute observations. This implementation slices the sorted target grid by position: the official training set is indices 0–109, timestamps 0–32700 seconds.
- No `vm_avg_cpu`, `distance`, `vm_p95_max_cpu`, source metadata CPU average/P95, or future deletion values enter the forecasting model. Reading-derived summaries used for interpretation are recomputed from training only.
- CPU terciles calculated on holdout data are not used. Segment assignment uses supplied static category and metadata subscription size.
- Exact duplicate target rows are collapsed; conflicting duplicate keys never enter a many-to-many join. Means count each VM once.
- Source deletion/creation statistics describe the trace and can be censored. They are not proof of observed churn or causal usage effects.
- Workload categories do not establish price ratios, latency guarantees or physical capacity. Extremely small buckets should not drive automated scaling decisions.
- One/two-hour periodicity is a hypothesis over a short window, not daily or weekly seasonality.
- Large intermediate merged datasets and VARMA experiment frameworks are omitted. An optional tree-experiment reproduction helper is retained with the notebooks, separate from the fixed final prediction path.

## Seven analysis sections

The generated findings cover cadence/coverage; missingness/duplicates/irregularities; lifetimes/churn; utilization distributions; temporal patterns; meaningful segments; and peak/low periods. The generated JSON/CSV files retain the numerical evidence. Add your own interpretation in `src/results/analysis_notes.md`; that file is preserved on reruns.

## Alternative-trend evidence

The previous training-only comparison favored damped Holt over the original SARIMA-24 trend (mean validation merged MAE 0.1491 versus 0.2097). It failed to maintain that advantage in the public final window (0.2140 versus 0.0579). This is contrary evidence to a universal superiority claim for either method. The final recipe remains fixed by candidate choice; the raw-data pipeline reproduces its baselines independently. A genuinely new future window is needed for an unbiased selection comparison.

The cleaned original notebook flow is now included in `../notebooks/`; source summaries remain descriptive, not forecasting inputs.
