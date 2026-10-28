# Cleaned notebooks

Open in this order: `explore.ipynb`, `transform.ipynb`, `train.ipynb`, `evaluate.ipynb`. Each starts from raw files or explicit saved results; no merged CSV or shared kernel globals are needed. Set `VM_DATA_DIR` or edit the data-directory cell. Install optional notebook/experiment tools with `python -m pip install -r requirements-notebooks.txt`.

Code cells have been cleaned and outputs cleared for delivery. Add explanations in the spaces after each analysis. Source notebooks remain in the working project; these are curated runnable versions. `train.ipynb` regenerates the submission and evaluation before `evaluate.ipynb` reads it.

`analysis_helpers.py` reproduces training-only jumps, exact group jump attribution, and descriptive metadata relationships. `tree_experiments.py` reproduces the optional RF/GBM/GAM comparison; it is not imported by main. `evidence/` retains the recorded experiment results/settings. Optional reruns save separately under `outputs/`, preserving that evidence.

Setup is independent of the checkout name. Raw files default to `<checkout>/challenge_data`; set `VM_DATA_DIR` for an external dataset and `VM_PROJECT_ROOT` when launching from an unrelated directory. No original parent-project directory is needed.
