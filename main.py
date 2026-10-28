"""Reproduce forecasts directly from vm_metadata and vm_cpu_readings."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import platform
import sys

SRC=Path(__file__).resolve().parent/"src"
sys.path.insert(0,str(SRC))
from importlib.metadata import version
import numpy as np
import pandas as pd
from workload.data import load_data, split, validate_predictions
from workload.model import forecast, SPEC
from workload.evaluation import validation_origins, evaluate, evaluate_groups, capacity_plan
from workload.analysis import analyze, plot_forecasts, plot_errors

ROOT=Path(__file__).resolve().parent


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir','--data',type=Path,required=True,help='Directory with the two raw release CSV/CSV.GZ files')
    parser.add_argument('--output',type=Path,default=ROOT/'predictions.csv')
    parser.add_argument('--results-dir',type=Path,default=ROOT/'src/results')
    parser.add_argument('--horizon',type=int,default=24,help='Official horizon is 24; other values are for smoke tests only')
    parser.add_argument('--train-steps',type=int,default=None,help='Default: all timestamps except final horizon (110 for the release)')
    parser.add_argument('--evaluate',action='store_true',help='Score/plot observed holdout only after predictions are frozen and saved')
    parser.add_argument('--skip-analysis',action='store_true',help='Skip workload figures; still validate data and forecasting')
    args=parser.parse_args(argv)
    output=args.output.resolve(); results=args.results_dir.resolve()
    results.mkdir(parents=True,exist_ok=True)
    if not args.evaluate:
        for name in ('holdout_metrics.csv','holdout_predictions.csv','holdout_group_metrics.csv'):
            (results/name).unlink(missing_ok=True)
    print('Loading raw metadata and usage files:',args.data_dir.resolve(),flush=True)
    data=load_data(args.data_dir)
    data.targets.to_csv(results/'targets_full.csv')
    (results/'data_audit.json').write_text(json.dumps(data.audit,indent=2))
    n,timestamps=split(data,args.horizon,args.train_steps)
    print(f'Training {n} observations; forecasting {len(timestamps)} timestamps, {timestamps[0]}..{timestamps[-1]} seconds.',flush=True)
    if not args.skip_analysis: analyze(data,n,results)
    validation_metrics=[]; validation_predictions=[]; validation_groups=[]; fit_log=[]
    origins=validation_origins(n,args.horizon)
    for origin in origins:
        print('Validation origin:',origin,flush=True)
        fleet,groups,counts,log=forecast(data,origin,args.horizon)
        metrics,predictions=evaluate(data,origin,args.horizon,fleet,counts,'validation')
        validation_metrics.append(metrics); validation_predictions.append(predictions); fit_log.extend(log)
        validation_groups.append(evaluate_groups(data,origin,args.horizon,groups,counts,'validation'))
    vm=pd.concat(validation_metrics,ignore_index=True) if validation_metrics else pd.DataFrame(columns=['phase','origin','model','target','MAE','RMSE','WAPE_pct'])
    vp=pd.concat(validation_predictions,ignore_index=True) if validation_predictions else pd.DataFrame(columns=['phase','origin','timestamp','model','target','actual','prediction','residual','is_peak'])
    vm.to_csv(results/'validation_metrics.csv',index=False); vp.to_csv(results/'validation_predictions.csv',index=False)
    vg=pd.concat(validation_groups,ignore_index=True) if validation_groups else pd.DataFrame(columns=['phase','origin','model','vm_type','subscription_size','vm_count','MAE','RMSE'])
    vg.to_csv(results/'validation_group_metrics.csv',index=False)
    print('Fitting fixed recipe on training observations only...',flush=True)
    fleet,groups,counts,log=forecast(data,n,args.horizon); fit_log.extend(log)
    prediction=pd.DataFrame(dict(timestamp=timestamps.to_numpy(),active_vm_count=np.repeat(int(counts.sum()),args.horizon),
        aggregate_cpu_utilization=fleet['SARIMA12_24Trend']))
    validate_predictions(prediction,timestamps)
    output.parent.mkdir(parents=True,exist_ok=True); prediction.to_csv(output,index=False)
    code=b''.join(p.read_bytes() for p in [ROOT/'main.py',*sorted((SRC/'workload').glob('*.py'))])
    manifest=dict(model='SARIMA12_24Trend',recipe_selection='Fixed explicitly by the candidate; no automatic holdout selection',
        specification=SPEC,train_steps=n,horizon=args.horizon,validation_origins=origins,
        last_training_timestamp=int(data.targets.index[n-1]),holdout_timestamps=timestamps.tolist(),
        prediction_sha256=sha256(output.read_bytes()).hexdigest(),source_sha256=sha256(code).hexdigest(),
        training_target_sha256=sha256(data.targets.iloc[:n].to_numpy().tobytes()).hexdigest(),
        training_cpu_sha256=sha256(data.cpu.iloc[:n].to_numpy().tobytes()).hexdigest(),
        versions={name:version(name) for name in ['numpy','pandas','scipy','statsmodels','matplotlib','threadpoolctl']},
        python=platform.python_version(),analysis_generated=not args.skip_analysis,holdout_evaluation_requested=args.evaluate,status='predictions_frozen_before_optional_holdout_evaluation')
    (results/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    pd.DataFrame(fit_log).to_csv(results/'fit_status.csv',index=False)
    records=[]
    for g in range(len(counts)):
        for name,p in groups.items():
            if not counts[g]: continue
            records.append(pd.DataFrame(dict(timestamp=timestamps,model=name,vm_type=data.groups[g][0],
                subscription_size=data.groups[g][1],vm_count=int(counts[g]),mean_cpu=p[:,g])))
    pd.concat(records,ignore_index=True).to_csv(results/'group_forecasts.csv',index=False)
    pd.DataFrame(dict(vm_type=data.groups.get_level_values(0),subscription_size=data.groups.get_level_values(1),
        vm_count=counts,weight=counts/counts.sum())).to_csv(results/'group_counts.csv',index=False)
    plan,policy=capacity_plan(prediction,vp,data.targets.iloc[:n])
    plan.to_csv(results/'capacity_plan.csv',index=False)
    (results/'capacity_policy.json').write_text(json.dumps(policy,indent=2))
    holdout_predictions=None
    if args.evaluate:
        metrics,predictions=evaluate(data,n,args.horizon,fleet,counts,'retrospective_holdout')
        holdout_predictions=predictions
        evaluate_groups(data,n,args.horizon,groups,counts,'retrospective_holdout').to_csv(results/'holdout_group_metrics.csv',index=False)
        metrics.to_csv(results/'holdout_metrics.csv',index=False); predictions.to_csv(results/'holdout_predictions.csv',index=False)
        print(metrics.loc[metrics.target.eq('aggregate_cpu_utilization'),['model','MAE','RMSE','WAPE_pct','peak_steps']].to_string(index=False),flush=True)
    if not args.skip_analysis:
        plot_forecasts(data,n,args.horizon,groups,fleet,results,args.evaluate)
        plot_errors(vp,holdout_predictions,results)
    print('Submission:',output,flush=True)
    print('Results:',results,flush=True)
    return prediction,manifest


if __name__=='__main__':
    main()
