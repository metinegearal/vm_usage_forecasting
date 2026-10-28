"""Optional reproduction of the failed group-level RF/GBM experiments and GAM comparator."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits
from workload.evaluation import metrics
HORIZON=24
LEARNERS=['RandomForest','GBM','GAM']

PARAMS = {
    "RandomForest": dict(n_estimators=150, max_depth=4, min_samples_leaf=5,
                         max_features=.8, random_state=42, n_jobs=1),
    "GBM": dict(n_estimators=150, learning_rate=.03, max_depth=2,
                min_samples_leaf=5, loss="squared_error", random_state=42),
    "GAM": dict(n_knots=3, degree=3, ridge_alpha=10.),
}

def features(state, step):
    """Only values strictly before step; time is relative, not time of day."""
    return np.array([*[state[-lag] for lag in (1, 2, 3, 6, 12, 24)],
        state[-6:].mean(), state[-24:].mean(), state[-12:].std(),
        state[-1]-state[-2], step/12,
        np.sin(2*np.pi*step/12), np.cos(2*np.pi*step/12),
        np.sin(2*np.pi*step/24), np.cos(2*np.pi*step/24)])

def make_model(name):
    if name == "RandomForest":
        return RandomForestRegressor(**PARAMS[name])
    if name == "GBM":
        return GradientBoostingRegressor(**PARAMS[name])
    # Gaussian identity-link GAM: sum_j f_j(x_j) + linear time/Fourier terms.
    # Feature-wise B-splines have no cross-feature interaction terms. Ridge
    # penalizes basis coefficients; this is not pyGAM's curvature penalty.
    transform = ColumnTransformer([
        ("smooth", SplineTransformer(n_knots=3, degree=3, include_bias=False,
                                     extrapolation="linear"), list(range(10))),
        ("linear", StandardScaler(), list(range(10, 15))),
    ])
    return make_pipeline(transform, Ridge(alpha=10.))

def forecast_groups(history, counts):
    """history is a prefix-only (time, nine groups) matrix."""
    predictions = {
        "Persistence": np.repeat(history[-1:], HORIZON, axis=0),
        "SeasonalNaive12": np.tile(history[-12:], (2, 1)),
        "SeasonalNaive24": history[-24:].copy(),
        **{name: np.full((HORIZON, 9), np.nan) for name in LEARNERS},
    }
    if len(history) <= 24:
        raise ValueError("Need more than 24 training observations")
    with threadpool_limits(limits=1):
        for g in np.flatnonzero(counts):
            y = history[:, g]
            mean, scale = y.mean(), y.std()
            if scale < 1e-10:
                for name in LEARNERS:
                    predictions[name][:, g] = y[-1]
                continue
            z = (y-mean)/scale
            x = np.stack([features(z[:t], t) for t in range(24, len(z))])
            # Predict a correction to the 12-step seasonal value, not the
            # complete CPU level. The seasonal anchor is recursive after h=12.
            target = z[24:] - z[12:-12]
            for name in LEARNERS:
                model = make_model(name)
                model.fit(x, target)
                state = z.copy()
                for h in range(HORIZON):
                    correction = float(model.predict(features(state, len(state))[None, :])[0])
                    value = float(np.clip((state[-12]+correction)*scale+mean, 0, 100))
                    predictions[name][h, g] = value
                    state = np.append(state, (value-mean)/scale)
    return predictions

def run_experiments(data):
    if len(data.targets)!=134: raise ValueError('This recorded experiment uses the official 110/24 window')
    records=[]
    for origin in [62,86,110]:
        print('Tree/GAM experiment origin:',origin,flush=True)
        counts=data.counts_at(origin)
        predictions=forecast_groups(data.group_means(origin),counts)
        actual=data.targets.aggregate_cpu_utilization.iloc[origin:origin+24].to_numpy()
        # All forecasts exist before this window's observed values enter scoring.
        for name,pred in predictions.items():
            fleet=np.nansum(pred*(counts/counts.sum()),axis=1)
            records.append(dict(phase='validation' if origin<110 else 'retrospective_test',origin=origin,
                model=name,level='fleet',**metrics(actual,fleet)))
    return pd.DataFrame(records)
