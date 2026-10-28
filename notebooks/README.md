# Analysis notebooks

Recommended execution order:

1. `explore.ipynb`: training-window workload analysis and jumps.
2. `transform.ipynb`: metadata relationships and subscription-size segmentation.
3. `train.ipynb`: reproduce predictions and evaluation from the raw release files.
4. `evaluate.ipynb`: compare errors and inspect unsuccessful model benchmarks.

Install `requirements-notebooks.txt` for the interactive environment and optional tree-model reproduction. Code-cell outputs are cleared; generated figures and tables are retained under `outputs/` and `src/results/`.

Raw files default to `<checkout>/challenge_data`. Set `VM_DATA_DIR` for an external data directory and `VM_PROJECT_ROOT` when launching from an unrelated directory. Repository discovery does not depend on the checkout name.

`analysis_helpers.py` implements training-only jump attribution and descriptive metadata comparisons. `tree_experiments.py` contains the optional RF/GBM/GAM benchmark, isolated from the prediction entry point. `evidence/` contains recorded experiment settings and metrics; optional reruns write separately to `outputs/`.

The training notebook regenerates the submission and evaluation files before the evaluation notebook reads them. Each notebook starts with explicit data/result paths and does not require variables from another notebook kernel.
