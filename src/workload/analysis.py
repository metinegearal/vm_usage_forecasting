"""Seven focused, training-window analyses distilled from the notebooks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_figure(fig,path):
    fig.tight_layout(); fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)


def analyze(data,n,output):
    output=Path(output); tables=output/'tables'; figures=output/'figures'
    tables.mkdir(parents=True,exist_ok=True); figures.mkdir(exist_ok=True)
    cpu=data.cpu.iloc[:n]; fleet=data.targets.iloc[:n]; values=cpu.to_numpy()
    hours=(cpu.index.to_numpy()-cpu.index[0])/3600
    valid=values[np.isfinite(values)]; means=cpu.mean(axis=0); std=cpu.std(axis=0)
    coverage=cpu.notna().sum(axis=0).rename('observations').to_frame()
    coverage['coverage_fraction']=coverage.observations/n
    coverage.groupby('observations').size().rename('vm_count').to_csv(tables/'coverage_summary.csv')
    presence=cpu.notna().to_numpy()
    arrivals=(presence[1:]&~presence[:-1]).sum(axis=1)
    departures=(~presence[1:]&presence[:-1]).sum(axis=1)
    pd.DataFrame(dict(timestamp=cpu.index[1:],arrivals=arrivals,departures=departures)).to_csv(tables/'membership_changes.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    axes[0].plot(hours,fleet.active_vm_count); axes[0].set(xlabel='Relative hours',ylabel='Represented VMs')
    axes[1].plot(hours[1:],arrivals,label='Arrivals/reappearances')
    axes[1].plot(hours[1:],departures,label='Departures/missing'); axes[1].legend()
    axes[1].set(xlabel='Relative hours',ylabel='Membership changes')
    save_figure(fig,figures/'coverage_churn.png')
    per_vm=pd.DataFrame(dict(mean_cpu=means,std_cpu=std,idle_fraction=cpu.le(1).sum()/cpu.notna().sum(),
        high_fraction=cpu.ge(80).sum()/cpu.notna().sum())).join(data.metadata[['vm_category','subscription_group_size','subscription_size']])
    pd.DataFrame({'quantile':[0,.1,.5,.9,.95,.99,1], 'avg_cpu':np.quantile(valid,[0,.1,.5,.9,.95,.99,1])}).to_csv(tables/'cpu_quantiles.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    axes[0].hist(valid,bins=np.linspace(0,100,51),color='steelblue'); axes[0].set(xlabel='VM average CPU (%)',ylabel='Reading count',yscale='log')
    axes[1].hist(means.dropna(),bins=40); axes[1].set(xlabel='Per-VM training mean CPU (%)',ylabel='VM count')
    save_figure(fig,figures/'cpu_distributions.png')
    counts=data.counts_at(n); segments=[]
    for g in range(len(data.groups)):
        mask=data.group_id==g
        segments.append(dict(vm_type=data.groups[g][0],subscription_size=data.groups[g][1],
            vm_count=int(mask.sum()),forecast_vm_count=int(counts[g]),
            training_mean_cpu=float(means.iloc[mask].mean()),median_vm_std=float(std.iloc[mask].median()),
            idle_fraction=float(per_vm.idle_fraction.iloc[mask].mean())))
    segments=pd.DataFrame(segments); segments.to_csv(tables/'segments.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for category,part in per_vm.groupby('vm_category'):
        axes[0].hist(part.mean_cpu.dropna(),bins=np.linspace(0,100,41),density=True,histtype='step',label=category)
    axes[0].legend(); axes[0].set(xlabel='Per-VM mean CPU (%)',ylabel='Density')
    axes[1].bar(np.arange(len(segments)),segments.training_mean_cpu)
    axes[1].set_xticks(np.arange(len(segments)),labels=[f'{a} / {b}' for a,b in data.groups],rotation=65,ha='right',fontsize=7)
    axes[1].set_ylabel('Group training mean CPU (%)'); save_figure(fig,figures/'segments.png')
    low,high=fleet.aggregate_cpu_utilization.quantile([.1,.9])
    demand=fleet.copy(); demand['period']=np.select([demand.aggregate_cpu_utilization<=low,demand.aggregate_cpu_utilization>=high],['low','peak'],default='normal')
    demand.to_csv(tables/'demand_periods.csv')
    fig,ax=plt.subplots(figsize=(11,3.8))
    ax.plot(hours,fleet.aggregate_cpu_utilization,color='black',label='Training fleet CPU')
    ax.axhline(high,color='red',linestyle='--',label='Training P90'); ax.axhline(low,color='blue',linestyle='--',label='Training P10')
    ax.set(xlabel='Relative hours',ylabel='Fleet mean CPU (%)'); ax.legend(); save_figure(fig,figures/'temporal_demand.png')
    autocorrelation={str(lag):float(fleet.aggregate_cpu_utilization.autocorr(lag)) if n>lag+1 and fleet.aggregate_cpu_utilization.std()>1e-10 else None for lag in (1,3,6,12,24)}
    top=max(1,int(np.ceil(len(means)*.1))); total=float(means.sum())
    creations=pd.to_numeric(data.metadata.get('timestamp_vm_created',pd.Series(dtype=float)),errors='coerce')
    deletions=pd.to_numeric(data.metadata.get('timestamp_vm_deleted',pd.Series(dtype=float)),errors='coerce')
    lifetime=(deletions-creations)/86400 if len(creations) and len(deletions) else pd.Series(dtype=float)
    summary=dict(training_timestamps=n,training_start=int(cpu.index[0]),training_end=int(cpu.index[-1]),
        represented_interval_hours=n*5/60,elapsed_span_hours=float(hours[-1]),vm_count=cpu.shape[1],
        mean_cpu=float(valid.mean()),median_cpu=float(np.median(valid)),idle_reading_fraction=float((valid<=1).mean()),
        high_reading_fraction=float((valid>=80).mean()),median_vm_std=float(std.median()),
        vm_category_counts=data.metadata.vm_category.value_counts().to_dict(),
        missing_training_vm_readings=int(cpu.isna().sum().sum()),arrivals_after_initial_boundary=int(arrivals.sum()),
        departures_before_final_boundary=int(departures.sum()),
        metadata_lifetime_median_days=float(lifetime.median()) if len(lifetime) else None,
        vms_at_max_metadata_deletion=int(deletions.eq(deletions.max()).sum()) if len(deletions) else None,
        peak_threshold=float(high),low_threshold=float(low),peak_steps=int((demand.period=='peak').sum()),
        low_steps=int((demand.period=='low').sum()),max_absolute_ramp_pp=float(fleet.aggregate_cpu_utilization.diff().abs().max()),
        top_10pct_vm_demand_share=float(means.nlargest(top).sum()/total) if total>0 else None,
        fleet_autocorrelation=autocorrelation)
    (output/'analysis_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    parts=[('Timestamp cadence and coverage',f'{n} training observations, 300-second cadence, {summary["missing_training_vm_readings"]} missing VM readings. Relative seconds do not identify dates or local business hours.'),
        ('Missingness, duplicates, and irregularities','See data_audit.json. Exact duplicate target keys are collapsed; conflicting values stop execution. Invalid average CPU readings are excluded; no CPU target is interpolated. Source metadata CPU statistics never enter forecasting.'),
        ('VM lifetimes and churn',f'{summary["arrivals_after_initial_boundary"]} membership arrivals and {summary["departures_before_final_boundary"]} departures inside training. Complete-trace sampling prevents an unbiased churn conclusion. Creation at the trace boundary is not proof of simultaneous provisioning; deletion metadata can describe the future and is descriptive only.'),
        ('CPU utilization distributions',f'Mean {summary["mean_cpu"]:.3f}%, median {summary["median_cpu"]:.3f}%; {100*summary["idle_reading_fraction"]:.2f}% of readings are at most 1%. Low fleet utilization can coexist with hot individual VMs. No physical-core weighting is inferred.'),
        ('Temporal patterns and recurring behavior','See temporal_demand.png and fleet_autocorrelation in analysis_summary.json. Periods 12 and 24 mean one and two hours, not daily cycles. This short history cannot establish daily/weekly seasonality. Autocorrelation may reflect trend.'),
        ('Workload differences across meaningful VM segments','Nine slots combine three VM categories with subscription sizes 1-3, 4-20, >20. Sizes count VMs in supplied metadata, not all customer VMs. Unknown labels are retained. Resource buckets do not establish prices or exact capacity.'),
        ('Peak and low-demand periods',f'Training P10/P90 are {low:.4f}% / {high:.4f}%; largest absolute five-minute ramp {summary["max_absolute_ramp_pp"]:.4f} pp. Thresholds are descriptive workload levels, not physical capacity violations.')]
    findings='\n\n'.join(f'## {title}\n\n{body}' for title,body in parts)+'\n'
    (output/'analysis_findings.md').write_text(findings)
    return summary


def plot_forecasts(data,n,horizon,group_predictions,fleet,output,evaluate_test=False):
    output=Path(output); figures=output/'figures'; figures.mkdir(parents=True,exist_ok=True)
    actual=data.group_means(len(data.cpu))
    x=(data.cpu.index.to_numpy()-data.cpu.index[0])/3600; future=x[n:]
    colors={'SARIMA12_24Trend':'tab:red','SARIMA12':'tab:blue','SARIMA24':'tab:orange'}
    fig,axes=plt.subplots(3,3,figsize=(16,10.5),sharex=True)
    for g,ax in enumerate(axes.flat):
        ax.set_title(f'{data.groups[g][0]} / {data.groups[g][1]}')
        if not np.any(data.group_id==g): continue
        ax.plot(x[:n],actual[:n,g],color='black',label='Observed training',linewidth=1.2)
        if evaluate_test: ax.plot(x[n-1:],actual[n-1:,g],color='black',linestyle=':',label='Observed holdout')
        for name,color in colors.items(): ax.plot(future,group_predictions[name][:,g],color=color,label=name,linewidth=1.2)
        ax.axvline(future[0],color='gray',linestyle='--'); ax.axvspan(future[0],future[-1],color='gray',alpha=.08)
        ax.set_xlim(0,x[-1]); ax.grid(alpha=.15)
    handles,labels=next(ax.get_legend_handles_labels() for ax in axes.flat if ax.lines)
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.supxlabel('Hours since first observation'); fig.supylabel('Group mean CPU (%)')
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(figures/'groups_full_history.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for ax,target in zip(axes,['aggregate_cpu_utilization','active_vm_count']):
        ax.plot(x[:n],data.targets[target].iloc[:n],color='black',label='Training')
        if evaluate_test: ax.plot(future,data.targets[target].iloc[n:],color='black',linestyle=':',label='Observed holdout')
        if target=='active_vm_count': ax.plot(future,np.repeat(data.counts_at(n).sum(),horizon),color='tab:red',label='Forecast')
        else:
            for name in ['SARIMA12_24Trend','Persistence','SeasonalNaive24']: ax.plot(future,fleet[name],label=name)
        ax.axvline(future[0],color='gray',linestyle='--'); ax.set_ylabel(target); ax.legend(fontsize=8)
    axes[-1].set_xlabel('Hours since first observation'); save_figure(fig,figures/'forecast.png')


def plot_errors(validation,holdout,output):
    figures=Path(output)/'figures'; figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2 if holdout is not None else 1,figsize=(12,3.8),squeeze=False)
    frames=[('Training-only validation',validation)]
    if holdout is not None: frames.append(('Retrospective holdout',holdout))
    for ax,(title,frame) in zip(axes.flat,frames):
        for name in ['SARIMA12_24Trend','Persistence','SeasonalNaive12','SeasonalNaive24']:
            part=frame.loc[frame.model.eq(name)&frame.target.eq('aggregate_cpu_utilization')]
            if part.empty: continue
            error=part.assign(abs_error=part.residual.abs()).groupby('lead_step').abs_error.mean()
            ax.plot(error.index*5,error,label=name)
        ax.set(title=title,xlabel='Forecast lead (minutes)',ylabel='Absolute CPU error (pp)')
        if ax.lines: ax.legend(fontsize=7)
    save_figure(fig,figures/'forecast_errors.png')
