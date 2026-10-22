# Exploring football player valuations with regression

**Question:** which relationships help explain historical FIFA/SoFIFA values, and how should regression results be evaluated?

[Visual case study](https://tola-adelase.github.io/football-case-study.html) · [Original notebook](Football%20Players%20Market%20Value%20Prediction.ipynb)

MSc Data Science and Computational Intelligence dissertation project at Coventry University. Independent of my employer.

## What the target means

The saved cleaned dataset has **8,480 players and 65 columns**. Values are FIFA/SoFIFA estimates, **not observed transfer fees**. A prediction difference does not establish that a player is a transfer bargain or that the market is wrong.

| Feature | Pearson correlation with FIFA value |
|---|---:|
| Release clause | 0.974431 |
| Wage | 0.774808 |
| Overall rating | 0.642495 |
| Potential | 0.582892 |

These are descriptive associations, not causal effects. Financial variables can be close target proxies; their validity depends on the prediction setting and input availability.

The notebook explores linear, ridge, lasso and polynomial regression. Saved predictive scores are **not presented as validated out-of-sample performance**.

## Reproducibility and evaluation limits

The original input dataset is no longer available here. Original package versions were not retained, so a full training rerun is blocked. No new scores have been produced.

- Separate full-dataset splits use the same seed with different holdout sizes, creating overlap between validation and test populations.
- Polynomial cross-validation does not consistently include the same transformation as the evaluated model.
- Repeated player appearances and financial proxy features need leakage checks.

## Prepared redesign

[Draft PR #1](https://github.com/Tola-adelase/predicting-market-value-of-football-players-using-machine-learning-algorithm/pull/1) contains a separate `portfolio-v2/` companion. It prepares disjoint player-group train/validation/test sets, train-only group cross-validation, pipeline preprocessing, a median baseline and separate attribute/proxy feature sets. Contract tests were run; training still requires the missing dataset and modelling environment. It has not been merged or treated as a completed experiment.

Once an authorised input is available, follow the companion README, validate the schema and execute the notebook top-to-bottom before reporting fresh results.

## A related working demo

[Football Match Lab](https://tola-adelase.github.io/football-match-lab.html) is a separate descriptive project using public StatsBomb data. It does not reuse the missing SoFIFA input or make valuation predictions.
