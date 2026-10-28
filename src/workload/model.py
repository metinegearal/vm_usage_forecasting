"""Fixed SARIMA-12 oscillation plus the linear SARIMA-24 forecast trend."""
import warnings
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from threadpoolctl import threadpool_limits

METHODS = ["SARIMA12_24Trend", "SARIMA12", "SARIMA24", "Persistence", "SeasonalNaive12", "SeasonalNaive24"]
MIN_SARIMA_HISTORY = 60
SPEC = dict(order=(1,0,1), seasonal_components=(2,1,0), periods=(12,24),
            enforce_stationarity=True, enforce_invertibility=False)


def seasonal_naive(history, period, horizon):
    last = np.asarray(history)[-min(period,len(history)):]
    return np.resize(last, horizon)


def sarima(history, period, horizon, log, context, maxiter=300):
    y = np.asarray(history, dtype=float)
    # Small internal smoke datasets and incomplete segment histories must work
    # without pretending a seasonal model can be identified from a few rows.
    if len(y) < MIN_SARIMA_HISTORY or not np.isfinite(y).all() or np.ptp(y) < 1e-9:
        available = y[np.isfinite(y)]
        if not len(available): raise ValueError("Cannot forecast a group without training CPU")
        pred = seasonal_naive(available, period, horizon) if np.isfinite(y).all() else np.repeat(available[-1],horizon)
        log.append(dict(**context, period=period, status="short_constant_or_incomplete_history_fallback", optimizer="none", warnings=""))
        return pred
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = SARIMAX(y, order=SPEC['order'], seasonal_order=(*SPEC['seasonal_components'],period),
                enforce_stationarity=True, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=maxiter)
            optimizer = "lbfgs"
            if not fit.mle_retvals.get("converged",False):
                optimizer = "powell"
                fit = model.fit(start_params=fit.params, method=optimizer, disp=False, maxiter=maxiter)
            pred = np.asarray(fit.forecast(horizon), dtype=float)
            if not fit.mle_retvals.get("converged",False) or not np.isfinite(pred).all():
                raise RuntimeError("No converged finite SARIMA forecast")
            status = "converged"
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            pred = seasonal_naive(y,period,horizon)
            status = "failed_fit_seasonal_naive_fallback"
            optimizer = "none"
            caught_messages = str(error)
    message = " | ".join(dict.fromkeys(str(w.message) for w in caught))
    if status != "converged": message = caught_messages+" | "+message
    log.append(dict(**context,period=period,status=status,optimizer=optimizer,warnings=message))
    return pred


def forecast(data, end, horizon):
    """All model arguments are extracted only from data.cpu.iloc[:end]."""
    grouped = data.group_means(end)
    counts = data.counts_at(end)
    if counts.sum() == 0: raise ValueError("No valid VM observations at the forecast origin")
    weights = counts/counts.sum()
    group_predictions = {name:np.full((horizon,len(counts)),np.nan) for name in METHODS}
    log=[]
    with threadpool_limits(limits=1):
        for g in np.flatnonzero(counts):
            context=dict(origin=end,group_id=int(g),vm_type=data.groups[g][0],subscription_size=data.groups[g][1])
            y=grouped[:,g]
            p12=sarima(y,12,horizon,log,context)
            p24=sarima(y,24,horizon,log,context)
            t=np.arange(horizon)
            linear=lambda values: np.polyval(np.polyfit(t,values,1),t) if horizon>1 else values.copy()
            candidates=dict(SARIMA12_24Trend=p12-linear(p12)+linear(p24), SARIMA12=p12, SARIMA24=p24,
                Persistence=np.repeat(y[-1],horizon),SeasonalNaive12=seasonal_naive(y,12,horizon),
                SeasonalNaive24=seasonal_naive(y,24,horizon))
            for name,pred in candidates.items(): group_predictions[name][:,g]=np.clip(pred,0,100)
    fleet={name:np.nansum(pred*weights,axis=1) for name,pred in group_predictions.items()}
    # Baselines predict the same fleet target directly, even if roster varies.
    target=data.targets.aggregate_cpu_utilization.iloc[:end].to_numpy()
    fleet['Persistence']=np.repeat(target[-1],horizon)
    for period in (12,24): fleet[f'SeasonalNaive{period}']=seasonal_naive(target,period,horizon)
    return fleet,group_predictions,counts,log
