"""Temporal validation, retrospective metrics, and an illustrative capacity policy."""
import numpy as np
import pandas as pd
from .data import TARGETS


def metrics(actual, predicted):
    a,p=np.asarray(actual,dtype=float),np.asarray(predicted,dtype=float)
    valid=np.isfinite(a)&np.isfinite(p)
    if not valid.any(): return dict(MAE=np.nan,RMSE=np.nan,WAPE_pct=np.nan,observations=0)
    error=a[valid]-p[valid]; demand=np.abs(a[valid]).sum()
    return dict(MAE=float(np.abs(error).mean()),RMSE=float(np.sqrt(np.mean(error**2))),
        WAPE_pct=float(100*np.abs(error).sum()/demand) if demand>1e-8 else (0. if not np.any(error) else np.nan),
        observations=int(valid.sum()))


def validation_origins(n,horizon):
    return [origin for origin in (n-2*horizon,n-horizon) if origin>=max(4,min(12,horizon))]


def evaluate(data, end, horizon, fleet, counts, phase):
    actual=data.targets.iloc[end:end+horizon]
    train=data.targets.iloc[:end]
    peak_threshold=float(train.aggregate_cpu_utilization.quantile(.9))
    peak=actual.aggregate_cpu_utilization.to_numpy()>=peak_threshold
    rows=[]; predictions=[]
    for name,cpu in fleet.items():
        for target,pred in [('aggregate_cpu_utilization',cpu),('active_vm_count',np.repeat(counts.sum(),horizon))]:
            a=actual[target].to_numpy()
            rows.append(dict(phase=phase,origin=end,model=name,target=target,**metrics(a,pred),
                peak_threshold=peak_threshold,peak_steps=int(peak.sum()),
                peak_MAE=metrics(a[peak],np.asarray(pred)[peak])['MAE'] if peak.any() else np.nan))
            for lead,(timestamp,y,p) in enumerate(zip(actual.index,a,pred),1):
                predictions.append(dict(phase=phase,origin=end,timestamp=int(timestamp),lead_step=lead,model=name,target=target,
                    actual=float(y),prediction=float(p),residual=float(y-p),is_peak=bool(timestamp in actual.index[peak])))
    result=pd.DataFrame(rows)
    baseline=result.loc[result.model.eq('Persistence'),['target','MAE']].set_index('target').MAE
    result['persistence_MAE']=result.target.map(baseline)
    result['MAE_improvement_vs_persistence_pct']=np.divide(
        100*(result.persistence_MAE-result.MAE),result.persistence_MAE,
        out=np.full(len(result),np.nan),where=result.persistence_MAE.to_numpy()>1e-8)
    return result,pd.DataFrame(predictions)


def capacity_plan(predictions,validation_predictions,training):
    if len(validation_predictions):
        residual=validation_predictions.loc[
            validation_predictions.model.eq('SARIMA12_24Trend') &
            validation_predictions.target.eq('aggregate_cpu_utilization'),'residual'].dropna()
    else: residual=pd.Series(dtype=float)
    margin=max(1.,float(residual.quantile(.95))) if len(residual) else 1.
    threshold=float(training.aggregate_cpu_utilization.quantile(.9))
    plan=predictions.copy()
    plan['cpu_planning_level_pct']=np.minimum(100,plan.aggregate_cpu_utilization+margin)
    plan['vm_slots_with_2pct_reserve']=np.ceil(plan.active_vm_count*1.02).astype(int)
    plan['peak_review_trigger']=plan.cpu_planning_level_pct>=threshold
    policy=dict(cpu_margin_percentage_points=margin,validation_residual_samples=len(residual),
        training_peak_threshold_pct=threshold,vm_slot_reserve_fraction=.02,
        interpretation='Illustrative workload planning levels, not physical host capacity or calibrated uncertainty bounds')
    return plan,policy


def evaluate_groups(data,end,horizon,group_predictions,counts,phase):
    actual=data.group_means(end+horizon)[end:]
    result=[]
    for name,pred in group_predictions.items():
        for g in np.flatnonzero(counts):
            result.append(dict(phase=phase,origin=end,model=name,vm_type=data.groups[g][0],
                subscription_size=data.groups[g][1],vm_count=int(counts[g]),
                **metrics(actual[:,g],pred[:,g])))
    return pd.DataFrame(result)
