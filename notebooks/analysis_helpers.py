"""Training-only jump attribution and descriptive metadata comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
import matplotlib.pyplot as plt


def jump_analysis(data,n,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    y=data.targets.aggregate_cpu_utilization.iloc[:n]
    delta=y.diff(); magnitude=delta.abs(); center=magnitude.median()
    threshold=float(center+2*(magnitude-center).abs().median())
    jumps=pd.DataFrame({'cpu':y,'change_pp':delta}).loc[magnitude>threshold].copy()
    jumps['hours']=(jumps.index-y.index[0])/3600
    jumps['minutes_since_previous_flag']=jumps.index.to_series().diff()/60
    jumps.to_csv(output/'jumps.csv')
    grouped=data.group_means(n); counts=data.counts_at(n)
    if not data.cpu.iloc[:n].notna().all().all():
        raise ValueError('Exact fixed-roster jump attribution requires complete training observations')
    contribution=grouped*(counts/counts.sum())
    attribution=[]
    for timestamp in jumps.index:
        i=data.targets.index.get_loc(timestamp)
        for g in np.flatnonzero(counts):
            attribution.append(dict(timestamp=int(timestamp),vm_type=data.groups[g][0],subscription_size=data.groups[g][1],
                contribution_change_pp=float(contribution[i,g]-contribution[i-1,g])))
    attribution=pd.DataFrame(attribution,columns=['timestamp','vm_type','subscription_size','contribution_change_pp'])
    if len(attribution):
        np.testing.assert_allclose(attribution.groupby('timestamp').contribution_change_pp.sum(),jumps.change_pp,atol=1e-9)
    attribution.to_csv(output/'jump_attribution.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    hours=(y.index-y.index[0])/3600
    axes[0].plot(hours,y,color='black',label='Training fleet CPU')
    axes[0].scatter(jumps.hours,jumps.cpu,color='red',s=25,label='Flagged changes')
    axes[0].plot(hours,y.rolling(6,min_periods=6).mean(),'--',label='Trailing 30-minute average')
    axes[0].set_ylabel('CPU (%)'); axes[0].legend()
    axes[1].plot(hours,delta,color='gray'); axes[1].axhline(threshold,color='red',linestyle='--')
    axes[1].axhline(-threshold,color='red',linestyle='--')
    axes[1].set(xlabel='Hours since first observation',ylabel='Five-minute change (pp)')
    fig.tight_layout(); fig.savefig(output/'jumps.png',dpi=150); plt.close(fig)
    summary=dict(training_steps=n,threshold_pp=threshold,flags=len(jumps),
        largest_absolute_change_timestamp=int(delta.abs().idxmax()),largest_absolute_change_pp=float(delta.loc[delta.abs().idxmax()]))
    return jumps,attribution,summary


def metadata_analysis(data,n,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    vm=data.metadata.copy(); vm['training_mean_cpu']=data.cpu.iloc[:n].mean()
    vm['training_std_cpu']=data.cpu.iloc[:n].std()
    sharing=[]
    for column in ['subscription_id','deployment_id']:
        if column not in vm: continue
        ids=vm[column].astype('string').str.strip().replace('',pd.NA)
        size=ids.map(ids.value_counts())
        singleton=vm.loc[size.eq(1),'training_mean_cpu'].dropna()
        shared=vm.loc[size.gt(1),'training_mean_cpu'].dropna()
        test=ttest_ind(singleton,shared,equal_var=False) if min(len(singleton),len(shared))>1 else None
        sharing.append(dict(id_type=column,singleton_n=len(singleton),shared_n=len(shared),
            singleton_mean=singleton.mean(),shared_mean=shared.mean(),difference_pp=shared.mean()-singleton.mean(),
            welch_p_value=float(test.pvalue) if test is not None else np.nan))
    sharing=pd.DataFrame(sharing); sharing.to_csv(output/'metadata_sharing.csv',index=False)
    categories=vm.groupby('vm_category').agg(vm_count=('training_mean_cpu','size'),mean_cpu=('training_mean_cpu','mean'),
        median_cpu=('training_mean_cpu','median'),mean_within_vm_std=('training_std_cpu','mean'))
    categories.to_csv(output/'metadata_categories.csv')
    resources=pd.DataFrame()
    if {'virtual_core_bucket','memory_gb_bucket'}.issubset(vm):
        resources=vm.groupby(['virtual_core_bucket','memory_gb_bucket'],observed=True).agg(
            vm_count=('training_mean_cpu','size'),training_mean_cpu=('training_mean_cpu','mean')).reset_index()
        resources['small_sample']=resources.vm_count<100
        resources.to_csv(output/'metadata_resources.csv',index=False)
    lifetime=[]
    if 'timestamp_vm_deleted' in vm:
        deletion=pd.to_numeric(vm.timestamp_vm_deleted,errors='coerce')/86400
        for column in ['training_mean_cpu','avg_cpu','p95_max_cpu','max_cpu']:
            if column not in vm: continue
            values=pd.to_numeric(vm[column],errors='coerce')
            mask=values.notna()&deletion.notna()
            rho,p=spearmanr(values[mask],deletion[mask])
            lifetime.append(dict(metric=column,spearman_rho=float(rho),p_value=float(p),sample_count=int(mask.sum())))
    lifetime=pd.DataFrame(lifetime); lifetime.to_csv(output/'metadata_lifetime_associations.csv',index=False)
    gap=None
    if {'avg_cpu','p95_max_cpu'}.issubset(vm):
        vm['source_burst_gap']=(pd.to_numeric(vm.p95_max_cpu)-pd.to_numeric(vm.avg_cpu)).abs()/np.sqrt(2)
        gap=vm.groupby('vm_category').source_burst_gap.agg(['count','mean','median'])
        gap.to_csv(output/'metadata_burst_gap.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar(categories.index,categories.mean_cpu); axes[0].set_ylabel('Training per-VM mean CPU (%)')
    axes[0].tick_params(axis='x',rotation=15)
    if len(sharing):
        pos=np.arange(len(sharing)); axes[1].bar(pos-.18,sharing.singleton_mean,.36,label='Singleton')
        axes[1].bar(pos+.18,sharing.shared_mean,.36,label='Shared')
        axes[1].set_xticks(pos,sharing.id_type); axes[1].legend()
    axes[1].set_ylabel('Training per-VM mean CPU (%)')
    fig.tight_layout(); fig.savefig(output/'metadata_connections.png',dpi=150); plt.close(fig)
    return dict(categories=categories,sharing=sharing,resources=resources,lifetime=lifetime,burst_gap=gap)


def deletion_analysis(data,n,output):
    """Recorded source deletion timing, not observed training-window churn."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    vm=data.metadata.copy()
    vm['training_mean_cpu']=data.cpu.iloc[:n].mean()
    vm['recorded_deletion_day']=pd.to_numeric(vm.timestamp_vm_deleted,errors='coerce')/86400
    boundary=vm.recorded_deletion_day.max()
    vm['at_maximum_boundary']=vm.recorded_deletion_day.eq(boundary)
    vm['timing_group']=np.where(vm.at_maximum_boundary,'at maximum boundary','before maximum boundary')
    categories=vm.groupby('vm_category').agg(vm_count=('training_mean_cpu','size'),
        median_recorded_deletion_day=('recorded_deletion_day','median'),
        mean_recorded_deletion_day=('recorded_deletion_day','mean'),
        at_maximum_boundary_fraction=('at_maximum_boundary','mean'),
        mean_training_cpu=('training_mean_cpu','mean'))
    categories.to_csv(output/'deletion_by_category.csv')
    summaries=[]; correlations=[]
    for name,part in [('All',vm),*list(vm.groupby('vm_category'))]:
        summary=part.groupby('timing_group').agg(vm_count=('training_mean_cpu','size'),
            mean_training_cpu=('training_mean_cpu','mean'),median_training_cpu=('training_mean_cpu','median')).reset_index()
        summary['vm_category']=name; summaries.append(summary)
        mask=part.training_mean_cpu.notna() & part.recorded_deletion_day.notna()
        rho,p=spearmanr(part.loc[mask,'training_mean_cpu'],part.loc[mask,'recorded_deletion_day'])
        correlations.append(dict(vm_category=name,observations=int(mask.sum()),spearman_rho=float(rho),p_value=float(p)))
    summaries=pd.concat(summaries,ignore_index=True); summaries.to_csv(output/'deletion_usage_groups.csv',index=False)
    correlations=pd.DataFrame(correlations); correlations.to_csv(output/'deletion_usage_correlations.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    names=categories.index.tolist()
    axes[0].boxplot([vm.loc[vm.vm_category.eq(name),'recorded_deletion_day'].dropna() for name in names],showfliers=False)
    axes[0].set_xticks(np.arange(1,len(names)+1),names,rotation=15)
    axes[0].set_ylabel('Recorded deletion day (relative to source start)')
    overall=summaries.loc[summaries.vm_category.eq('All')]
    x=np.arange(len(overall))
    axes[1].bar(x-.18,overall.mean_training_cpu,.36,label='Mean')
    axes[1].bar(x+.18,overall.median_training_cpu,.36,label='Median')
    axes[1].set_xticks(x,overall.timing_group,rotation=10)
    axes[1].set_ylabel('Training mean CPU per VM (%)'); axes[1].legend()
    fig.suptitle('Source deletion timing: descriptive associations, boundary censoring unresolved')
    fig.tight_layout(); fig.savefig(output/'deletion_usage.png',dpi=150); plt.close(fig)
    return dict(categories=categories,usage_groups=summaries,correlations=correlations)


def subscription_size_analysis(data,n,output):
    """Restore the notebook's fine size bins and show the final three-bin pooling."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    vm=data.metadata.copy(); vm['training_mean_cpu']=data.cpu.iloc[:n].mean()
    bins=[0,1,2,3,5,10,20,50,100,np.inf]
    labels=['1','2','3','4–5','6–10','11–20','21–50','51–100','100+']
    summaries=[]; compositions=[]
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for j,kind in enumerate(['subscription','deployment']):
        idcol=kind+'_id'
        if idcol not in vm:
            axes[0,j].set_title(kind+' metadata unavailable'); continue
        ids=vm[idcol].astype('string').str.strip().replace('',pd.NA)
        size=ids.map(ids.value_counts())
        frame=vm.loc[ids.notna(),['vm_category','training_mean_cpu']].copy()
        frame['organization_id']=ids.loc[ids.notna()]
        frame['size_bin']=pd.cut(size.loc[ids.notna()],bins,labels=labels)
        profile=frame.groupby('size_bin',observed=False).agg(vm_count=('training_mean_cpu','size'),
            distinct_groups=('organization_id','nunique'),mean_cpu=('training_mean_cpu','mean'),median_cpu=('training_mean_cpu','median')).reindex(labels)
        profile['id_type']=kind;summaries.append(profile.reset_index())
        mix=pd.crosstab(frame.size_bin,frame.vm_category).reindex(labels,fill_value=0)
        mix['id_type']=kind; compositions.append(mix.reset_index())
        x=np.arange(len(labels));ax=axes[0,j]
        ax.plot(x,profile.mean_cpu,'o-',label='Mean per-VM training CPU')
        ax.plot(x,profile.median_cpu,'s--',label='Median per-VM training CPU')
        for left,right,color,label in [(-.5,2.5,'tab:green','1–3'),(2.5,5.5,'tab:orange','4–20'),(5.5,8.5,'tab:blue','>20')]:
            ax.axvspan(left,right,color=color,alpha=.08)
            ax.text((left+right)/2,.96,label,transform=ax.get_xaxis_transform(),ha='center',va='top')
        ax.axvline(2.5,color='gray',linestyle=':');ax.axvline(5.5,color='gray',linestyle=':')
        ax.set_xticks(x,labels,rotation=25);ax.set(title=kind.title()+' size vs CPU',ylabel='Training CPU (%)')
        ax.legend(loc='lower left',fontsize=8)
        bottom=np.zeros(len(labels));ax=axes[1,j]
        for category in data.groups.get_level_values(0).unique():
            values=mix[category].to_numpy() if category in mix else np.zeros(len(labels))
            ax.bar(x,values,bottom=bottom,label=category);bottom+=values
        ax.axvline(2.5,color='gray',linestyle=':');ax.axvline(5.5,color='gray',linestyle=':')
        ax.set_xticks(x,labels,rotation=25);ax.set(ylabel='VM count in supplied sample',xlabel='VMs sharing the ID (inclusive)')
        ax.legend(fontsize=8)
    fig.suptitle('Fine size profiles and the three-bin pooling choice; training observations only')
    fig.tight_layout();fig.savefig(output/'subscription_size_rationale.png',dpi=150);plt.close(fig)
    profiles=pd.concat(summaries,ignore_index=True);profiles.to_csv(output/'organization_size_profiles.csv',index=False)
    pd.concat(compositions,ignore_index=True).to_csv(output/'organization_size_category_counts.csv',index=False)
    coarse=vm.groupby('subscription_size',observed=False).agg(vm_count=('training_mean_cpu','size'),
        mean_cpu=('training_mean_cpu','mean'),median_cpu=('training_mean_cpu','median'))
    coarse.to_csv(output/'subscription_three_bins.csv')
    return dict(fine_profiles=profiles,three_bins=coarse)
