# Football valuation — portfolio revision

**Status: code prepared; original player dataset required. No revised model scores are available.**

A reproducible companion to Adetola Adelase's 2021 MSc dissertation. This work
models historical FIFA/SoFIFA valuations, **not actual transfer fees**.
The original notebook is preserved unchanged at the repository root.

## Improvements

- Distinct player groups across approximately 60/20/20 train/validation/test splits.
- Training-only GroupKFold cross-validation; preprocessing inside each pipeline.
- Median baseline plus linear, ridge, lasso and polynomial-ridge models.
- Matched player-attribute and financial-proxy feature sets.
- Validation selects the workflow before a single final held-out evaluation.
- MAE/RMSE in EUR millions, R², residual plots and error analysis by value band/position.
- Input SHA-256, source provenance, package versions and split counts recorded.

## Run locally

Use Python 3.11 or 3.12. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name football-valuation-v2
python -m unittest discover -s tests -v
```

Restore `data/players_clean.csv` and `data/provenance.json` as described in
[data/README.md](data/README.md). Then execute from this directory:

```bash
python -m jupyter nbconvert --execute --to notebook \
  --ExecutePreprocessor.kernel_name=football-valuation-v2 \
  --ExecutePreprocessor.timeout=600 \
  --output football_valuation_v2_executed.ipynb \
  --output-dir notebooks notebooks/football_valuation_v2.ipynb
```

Inspect the executed notebook before publishing any findings. Computation
exports aggregate CSVs and a run manifest to ignored `outputs/`; individual
player records are not exported by default.

## Validation recorded during preparation

Three data-contract tests pass: deterministic identity-separated splits,
invalid target handling with predictor missingness retained for pipeline
imputation, and a fail-fast error when the original data is absent.
Python module and notebook cells pass syntax checks. Notebook structure is
checked locally, but full nbformat validation and end-to-end model execution
remain pending: this environment lacks scikit-learn, matplotlib, nbformat and
the original dataset. No model tests or chart inspection are claimed.

Requirements are compatible version ranges, not an environment lockfile.
The executed run manifest records exact installed versions. Data snapshot
identification and redistribution permission still need confirmation.

## Method references

- [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
- [MAE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
