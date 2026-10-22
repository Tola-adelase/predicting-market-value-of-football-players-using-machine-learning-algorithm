"""Reproducible evaluation of historical FIFA/SoFIFA valuations, in EUR millions."""
from pathlib import Path
import hashlib
import random

import numpy as np
import pandas as pd

SEED = 42
ATTRIBUTES = ['Age', 'Overall', 'Potential', 'Int_Reputation', 'Height', 'Weight']
PROXIES = ATTRIBUTES + ['Wage', 'Release_Clause']


def load_data(path):
    """Read the original cleaned CSV; never unpickle data automatically."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'Missing original dataset: {path}. Supply a CSV export of '
            'PlayerListFinalPickle; see data/README.md. No replacement data is generated.'
        )
    frame = pd.read_csv(path)
    required = ['Name', 'Value'] + PROXIES
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    report = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_rows': len(frame)}
    frame = frame.drop_duplicates().copy()
    report['exact_duplicates_removed'] = report['source_rows'] - len(frame)
    for col in ['Value'] + PROXIES:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    # Missing predictors are imputed inside CV. Targets and identity must be valid.
    identity = frame['ID'] if 'ID' in frame else frame['Name']
    identity = identity.astype('string').str.strip().str.casefold()
    valid = (np.isfinite(frame['Value']) & frame['Value'].gt(0)
             & identity.notna() & identity.ne(''))
    report['invalid_target_or_identity_removed'] = int((~valid).sum())
    frame = frame.loc[valid].copy()
    frame['_player_group'] = identity.loc[valid]
    frame[PROXIES] = frame[PROXIES].replace([np.inf, -np.inf], np.nan)
    frame = frame.reset_index(drop=True)
    if len(frame) < 100 or frame['_player_group'].nunique() < 50:
        raise ValueError('At least 100 valid records and 50 distinct players are required.')
    if frame[PROXIES].isna().all().any():
        raise ValueError('One or more required predictors are entirely missing.')
    report.update(clean_rows=len(frame), distinct_players=frame['_player_group'].nunique(),
                  grouping='ID' if 'ID' in frame else 'normalised Name')
    return frame, report


def split_players(frame, seed=SEED):
    """Split distinct player groups ~60/20/20; row proportions may vary."""
    groups = sorted(frame['_player_group'].unique().tolist())
    random.Random(seed).shuffle(groups)
    n_train, n_val = int(len(groups) * .6), int(len(groups) * .2)
    selected = [groups[:n_train], groups[n_train:n_train+n_val], groups[n_train+n_val:]]
    parts = [frame.loc[frame['_player_group'].isin(g)].copy() for g in selected]
    assert_disjoint(parts)
    return tuple(parts)


def assert_disjoint(parts):
    group_sets = [set(p['_player_group']) for p in parts]
    for i, a in enumerate(group_sets):
        for b in group_sets[i+1:]:
            if a & b:
                raise AssertionError('Player identity overlap across splits.')
    if any(p.empty for p in parts):
        raise AssertionError('An evaluation split is empty.')


def candidate_pipelines():
    # Imports are deferred so data and split checks do not require model packages.
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.dummy import DummyRegressor
    steps = [('impute', SimpleImputer(strategy='median')),
             ('scale', StandardScaler())]
    return {
        'Median baseline': Pipeline(steps + [('model', DummyRegressor(strategy='median'))]),
        'Linear': Pipeline(steps + [('model', LinearRegression())]),
        'Ridge': Pipeline(steps + [('model', Ridge(alpha=10.0))]),
        'Lasso': Pipeline(steps + [('model', Lasso(alpha=.01, max_iter=20000))]),
        'Polynomial ridge': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
            ('scale', StandardScaler()), ('model', Ridge(alpha=10.0))]),
    }


def score_predictions(actual, predicted):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {'mae_eur_m': mean_absolute_error(actual, predicted),
            'rmse_eur_m': float(np.sqrt(mean_squared_error(actual, predicted))),
            'r2': r2_score(actual, predicted)}


def select_models(train, validation):
    """CV uses training groups only; validation selects the overall workflow."""
    from sklearn.model_selection import GroupKFold, cross_validate
    records, finalists = [], {}
    for feature_set, features in [('Player attributes', ATTRIBUTES),
                                  ('Attributes + financial proxies', PROXIES)]:
        for name, pipeline in candidate_pipelines().items():
            result = cross_validate(
                pipeline, train[features], train['Value'],
                groups=train['_player_group'], cv=GroupKFold(n_splits=5),
                scoring='neg_mean_absolute_error', error_score='raise', n_jobs=1)
            errors = -result['test_score']
            records.append({'feature_set': feature_set, 'model': name,
                            'cv_mae_eur_m': errors.mean(),
                            'cv_fold_sd_eur_m': errors.std(ddof=1)})
        subset = [r for r in records if r['feature_set'] == feature_set]
        winner = min(subset, key=lambda r: r['cv_mae_eur_m'])
        pipeline = candidate_pipelines()[winner['model']]
        pipeline.fit(train[features], train['Value'])
        validation_metrics = score_predictions(validation['Value'],
                                               pipeline.predict(validation[features]))
        finalists[feature_set] = dict(model=winner['model'], features=features,
                                      validation_metrics=validation_metrics)
    chosen = min(finalists, key=lambda k: finalists[k]['validation_metrics']['mae_eur_m'])
    return pd.DataFrame(records), finalists, chosen


def evaluate_final(train, validation, test, finalists, chosen):
    """Refit the selected workflow, then evaluate the untouched test set once."""
    assert_disjoint([train, validation, test])
    development = pd.concat([train, validation], ignore_index=True)
    selected = finalists[chosen]
    pipeline = candidate_pipelines()[selected['model']]
    features = selected['features']
    pipeline.fit(development[features], development['Value'])
    predicted = pipeline.predict(test[features])
    baseline = np.repeat(development['Value'].median(), len(test))
    scores = pd.DataFrame([
        {'model': selected['model'], **score_predictions(test['Value'], predicted)},
        {'model': 'Median baseline', **score_predictions(test['Value'], baseline)},
    ])
    predictions = test[['Name', 'Value']].rename(columns={'Value': 'actual_eur_m'}).copy()
    if 'Position' in test:
        predictions['Position'] = test['Position'].astype(str).str.strip()
    predictions['predicted_eur_m'] = predicted
    predictions['residual_eur_m'] = predictions['actual_eur_m'] - predicted
    predictions['absolute_error_eur_m'] = predictions['residual_eur_m'].abs()
    predictions['value_band'] = pd.cut(predictions['actual_eur_m'],
        [0, 1, 5, 20, float('inf')], labels=['0–1m', '1–5m', '5–20m', '20m+'])
    return scores, predictions
