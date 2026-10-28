# Jump analysis, metadata connections and failed model attempts

## Jump analysis

The original exploration proposed absolute five-minute CPU changes above `median(abs(change)) + 2 * median(abs(abs(change) - median(abs(change))))`. We retained that rule, using only the first 110 observations. The resulting threshold is **0.2233 percentage points** and there are **18 flags**. The threshold is an exploratory heuristic, not an anomaly probability or a guaranteed capacity violation.

The largest absolute fleet change occurs at **1500 seconds (25 minutes)**: **+0.4368 pp**. Its largest group contributions are:

| Group | Contribution to fleet change (pp) |
| --- | ---: |
| Interactive / >20 | 0.1402 |
| Delay-insensitive / >20 | 0.1266 |
| Delay-insensitive / 4–20 | 0.0507 |
| Interactive / 4–20 | 0.0339 |
| Unknown / 4–20 | 0.0305 |

All nine contributions sum to the observed fleet change. Because the challenge cohort is complete, this jump reflects changing readings rather than changing represented VM counts. The weighting is by VM count, not physical cores. Consecutive flags can refer to one burst and its reversal; jump-spacing counts are not independent events or proof of an hourly season. The broad transient visible in the large Delay-insensitive group is also different from a single large adjacent-timestamp change.

Operationally, both sustained elevated levels and abrupt ramps matter. A forecast that models a recurring waveform may still miss an unscheduled ramp. Inspect the contributing segments rather than scaling the entire fleet from one flag. [Notebook](../notebooks/explore.ipynb), [plot](../notebooks/outputs/jumps.png), [jump table](../notebooks/outputs/jumps.csv), [exact contribution table](../notebooks/outputs/jump_attribution.csv).

## Metadata relationships

### Subscription and deployment sharing

| Identifier | Singleton VMs | Shared VMs | Singleton training CPU | Shared training CPU |
| --- | ---: | ---: | ---: | ---: |
| Subscription | 1,204 | 8,796 | 6.3999% | 9.8267% |
| Deployment | 1,242 | 8,758 | 6.4070% | 9.8405% |

These differences are approximately **3.43 pp**. The exploratory Welch p-values are small (about 1.9e-23 and 5.2e-24), but shared organizations create dependence between VMs and category composition is a confounder. The notebook displays unadjusted p-values; it does not establish a causal mechanism. Earlier source-metadata averages (roughly 6.34%/9.75%) cover a different measurement definition/window, so they should not be silently substituted for training-reading means.

The model's three size buckets include the current VM: **1–3, 4–20, >20**. Exact singleton-versus-shared comparisons and these three forecasting buckets answer different questions. Neither represents the customer's complete estate outside the curated sample.

### Why sizes 1–3, 4–20 and >20?

The [restored notebook graph](../notebooks/outputs/subscription_size_rationale.png) shows training CPU mean/median and workload-category counts over the original finer bins (1, 2, 3, 4–5, 6–10, 11–20, 21–50, 51–100, 100+). The [tables](../notebooks/outputs/organization_size_profiles.csv) include VM and distinct-organization counts.

The cutoff at 3 preserves the notebook's small-group definition, fewer than three other VMs. The cutoff at 20 pools the finer mid-size/large cohorts into a simple three-level design. It was an explicit grouping choice, not a demonstrated optimal breakpoint. On the supplied population the size buckets contain 2,779, 2,667 and 4,554 VMs respectively. Crossing with workload type gives nine interpretable groups and fewer fitted series than fine bins or individual subscriptions. Category mix and rare groups remain visible in the graph.

This supports an interpretable exploratory segmentation, not a claim that the cuts maximize forecast accuracy. No systematic training-only cutoff search was performed, and the previous nine-group merge did not improve fleet MAE over the direct aggregate recipe. Such a search would need nested chronological validation. A group with many VMs still has only 110 training timestamps.

### Workload category, cores and memory

Training category means are **8.2573% Delay-insensitive**, **11.5051% Interactive**, and **10.3547% Unknown**. Unknown is retained. Category names do not imply price ratios, service guarantees, or a target label that should be imputed from CPU observations.

Resource interactions are not monotonic sizing laws. For example, the 2-core/32-GB bucket averages **24.3304%** CPU but has only **four** VMs. Four-core/32-GB has 913 VMs and averages 6.8200%; eight-core/32-GB has 466 VMs and averages 9.6197%. Category mix, workload placement and sparse buckets may all affect comparisons. Open-ended resource buckets stay categorical; they do not permit exact core-weighted capacity calculations.

### Burst gap and lifetime

Your proxy `abs(source P95 max - source mean) / sqrt(2)` is retained as a **descriptive source-statistic gap**, not a validated instability score. Delay-insensitive has the largest mean gap (20.4709), versus 17.1146 for Interactive and 16.0512 for Unknown, yet its mean within-training standard deviation is lower (2.7973 versus 3.2436 and 3.3092). The proxy and observed short-window variability are therefore not interchangeable.

The Spearman correlation between training CPU mean and source deletion day is **0.0907**, a weak positive association. The source-average relationship is 0.0978. Many deletion times coincide with the maximum source boundary, and complete-trace sampling is selective, so these are not unbiased lifetime estimates or evidence that higher utilization causes longer retention. Source metadata CPU summaries/deletion times are descriptive only and never used by the forecasting code.

[Cleaned metadata notebook](../notebooks/transform.ipynb), [sharing table](../notebooks/outputs/metadata_sharing.csv), [resource table](../notebooks/outputs/metadata_resources.csv), [lifetime associations](../notebooks/outputs/metadata_lifetime_associations.csv).

### Unknown category and earlier deletion: the specific finding

The original notebook's Unknown-category pattern is now explicitly retained and reproduced:

| Category | VMs | Median recorded deletion day | Mean training CPU |
| --- | ---: | ---: | ---: |
| Delay-insensitive | 6,180 | 29.9931 | 8.2573% |
| Interactive | 3,091 | 29.9931 | 11.5051% |
| Unknown | 729 | 3.6354 | 10.3547% |

Thus Unknown records have substantially earlier source deletion times. However, their mean training CPU is not lower than both known categories. All recorded creation timestamps are zero in this release, so the recorded deletion-day/lifetime rankings coincide, but boundary semantics remain unresolved.

Among records **before the maximum deletion boundary**, median training CPU is **3.9698%**, versus **5.7324%** at the boundary. That supports a narrower statement about typical utilization. The corresponding **means reverse direction**: **9.9006% versus 9.3082%**, reflecting skew and mixed populations. 'Early-deleted VMs had low usage' is therefore too broad without specifying the statistic and comparison group. The boundary group contains potentially censored records, not confirmed later deletion events.

Overall training-CPU/deletion-day Spearman correlation is 0.0907; by category it is 0.0952 for Delay-insensitive, 0.0006 for Interactive and -0.1767 for Unknown. These heterogeneous associations are not a universal utilization-to-deletion rule. [Category table](../notebooks/outputs/deletion_by_category.csv), [mean/median comparisons](../notebooks/outputs/deletion_usage_groups.csv), [within-category correlations](../notebooks/outputs/deletion_usage_correlations.csv), [plot](../notebooks/outputs/deletion_usage.png).

### Future work: usage-informed deletion-risk forecasting

A reasonable extension is to forecast **probability of deletion over the next day** or a remaining-lifetime distribution using VM category, age, prefix-only CPU mean/idle fraction/variability, recent changes and organization size. A category-and-age baseline should be evaluated before claiming that CPU adds information. Usage effects should be checked within categories, given the sign changes above.

This needs longer, less selectively sampled longitudinal data and verified creation/deletion event flags. Confirm whether maximum deletion timestamps encode actual events or right censoring before constructing labels. Use a time-to-event approach that can retain right-censored observations, with chronological landmark validation and controls for dependence within subscription/deployment. Evaluate risk calibration and event prediction prospectively; do not use source lifetime CPU summaries or actual future deletion timestamps as features.

The supplied complete-trace cohort has only about 11 hours of CPU readings and no observed internal churn. It does not establish deletion-prediction accuracy. The extension is future work, not part of the submitted CPU/count forecasting model or a justification for automatically deleting VMs.

## Unsuccessful group-level tree settings

The recorded experiment trained nine separate group models with lagged CPU, rolling means/variability, relative time and one/two-hour cyclic terms. Random Forest and gradient boosting predicted corrections to a 12-step seasonal anchor. Twenty-four-step forecasts fed their own previous predictions back into lag features. The comparable horizons were 62→86 and 86→110 inside training, then 110→134 retrospectively.

| Method | Mean validation fleet MAE | Retrospective fleet MAE | Retrospective RMSE |
| --- | ---: | ---: | ---: |
| Random Forest | 0.4109 | 0.1671 | 0.1977 |
| Gradient boosting | 0.4118 | 0.1821 | 0.2120 |
| Additive GAM comparator | 0.4205 | 0.2676 | 0.3102 |
| Persistence | 0.2150 | 0.1468 | 0.1848 |
| Seasonal-naive-12 | 0.1915 | 0.1400 | 0.1590 |

GAM is not a tree model. Random Forest used 150 trees, depth 4 and minimum leaf size 5; GBM used 150 stages, depth 2, learning rate 0.03 and minimum leaf size 5. Full settings and results are in [recorded evidence](../notebooks/evidence/tree_model_settings.json) and [metrics](../notebooks/evidence/tree_model_metrics.csv). Validation MAE is the mean over the two equal-size horizons; the notebook's displayed mean fold RMSE should not be confused with pooled RMSE in the final report.

Only **38/62 supervised time rows per group** were available in validation after a 24-step warmup; the final fit had **86**. Weakly stable short cycles, few samples, and compounding recursive corrections are plausible reasons for failure. These causes were not isolated in ablations. Increasing complexity or adding more categories is not guaranteed to help. These particular settings lost even to simple baselines and were excluded from main; they do not prove that all tree models fail on all VM workloads.

Earlier row-level LightGBM attempts addressed a different objective. Some original notebook cells used CPU summaries produced with the unsafe `109 * 3600` cutoff, or different rollout/split definitions. Those attractive scores are not presented as leakage-safe, like-for-like results. Clean group-level RF/GBM results are the evidence used here. Aggregate and per-VM errors must not be mixed.

The optional [experiment code](../notebooks/tree_experiments.py) reproduces the group-level comparison using the extra notebook requirements. It is not imported by main. The evaluation notebook defaults to recorded evidence and offers an explicit optional rerun. All final-window comparisons remain retrospective because the public holdout was inspected during development.
