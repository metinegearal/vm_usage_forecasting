"""Behavioral checks for raw inputs, leakage boundaries and portable CLI."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

SRC=Path(__file__).resolve().parents[1]/"src"
sys.path.insert(0,str(SRC))
from unittest.mock import patch
import numpy as np
import pandas as pd
from workload.data import build_data, split, validate_predictions
from workload.model import forecast
from workload.evaluation import metrics, validation_origins

ROOT=Path(__file__).resolve().parents[1]


def fixture(steps=134):
    metadata=[]; readings=[]
    vm=0
    for size in (3,20,21):
        for i in range(size):
            key=f'vm-{vm:03}'
            metadata.append(dict(vm_id=key,subscription_id=f'sub-{size}',
                vm_category=['Delay-insensitive','Interactive','Unknown'][i%3],
                avg_cpu=999.,p95_max_cpu=999.,timestamp_vm_deleted=999999))
            for t in range(steps):
                readings.append(dict(timestamp=600+t*300,vm_id=key,avg_cpu=10+vm/10+np.sin(t*2*np.pi/12)+.005*t))
            vm+=1
    return pd.DataFrame(readings),pd.DataFrame(metadata)


class PipelineTests(unittest.TestCase):
    def test_full_targets_and_nonzero_timestamp_origin(self):
        r,m=fixture(); d=build_data(r,m); n,times=split(d)
        self.assertEqual(n,110); self.assertEqual(times.tolist(),list(range(33600,40800,300)))
        np.testing.assert_allclose(d.targets.active_vm_count,44)
        expected=r.groupby('timestamp').avg_cpu.mean()
        np.testing.assert_allclose(d.targets.aggregate_cpu_utilization,expected)

    def test_inclusive_subscription_bucket_edges(self):
        r,m=fixture(); d=build_data(r,m)
        bucket=d.metadata.groupby('subscription_group_size',observed=True).subscription_size.first().astype(str).to_dict()
        self.assertEqual(bucket,{3:'1-3',20:'4-20',21:'>20'})
        self.assertEqual(len(d.groups),9)

    def test_exact_duplicate_counted_once(self):
        r,m=fixture(); expected=build_data(r,m)
        d=build_data(pd.concat([r,r.iloc[:1]],ignore_index=True),m)
        pd.testing.assert_frame_equal(d.targets,expected.targets)
        self.assertEqual(d.audit['exact_target_duplicates_removed'],1)

    def test_conflicting_duplicate_rejected(self):
        r,m=fixture(); conflict=r.iloc[:1].copy(); conflict['avg_cpu']+=1
        with self.assertRaisesRegex(ValueError,'Conflicting duplicate'):
            build_data(pd.concat([r,conflict]),m)

    def test_invalid_reading_is_excluded_not_imputed(self):
        r,m=fixture(); r.loc[0,'avg_cpu']=np.nan
        d=build_data(r,m)
        self.assertEqual(d.targets.active_vm_count.iloc[0],43)
        self.assertTrue(pd.isna(d.cpu.iloc[0,0]))
        self.assertEqual(d.audit['invalid_avg_cpu_rows_excluded'],1)

    def test_whole_timestamp_gap_rejected(self):
        r,m=fixture(); d=build_data(r.loc[r.timestamp.ne(900)],m)
        self.assertEqual(d.audit['wholly_missing_source_timestamps'],1)
        with self.assertRaisesRegex(ValueError,'source gaps'): split(d)

    def test_unique_metadata_required(self):
        r,m=fixture()
        with self.assertRaisesRegex(ValueError,'one row per VM'):
            build_data(r,pd.concat([m,m.iloc[:1]]))

    def test_idle_vms_are_still_active(self):
        r,m=fixture(); r['avg_cpu']=0.
        d=build_data(r,m); fleet,_,counts,_=forecast(d,110,24)
        self.assertEqual(int(counts.sum()),44)
        np.testing.assert_allclose(fleet['SARIMA12_24Trend'],0,atol=1e-10)

    def test_no_holdout_or_metadata_summary_leakage(self):
        r,m=fixture(); before=build_data(r,m)
        changed=r.copy(); changed.loc[changed.timestamp>=33600,'avg_cpu']=99.
        meta=m.copy(); meta[['avg_cpu','p95_max_cpu','timestamp_vm_deleted']]=-99999
        after=build_data(changed,meta); seen=[]
        def fake(history,period,horizon,log,context,maxiter=300):
            seen.append(np.asarray(history).copy())
            t=np.arange(horizon)
            return np.repeat(np.mean(history[-12:]),horizon)+(.01*t if period==24 else .3*np.sin(t))
        with patch('workload.model.sarima',side_effect=fake):
            a,ga,counts,_=forecast(before,110,24); inputs=seen.copy(); seen.clear()
            b,gb,_,_=forecast(after,110,24)
        self.assertEqual(len(seen),18)
        for old,new in zip(inputs,seen):
            self.assertEqual(len(new),110); np.testing.assert_array_equal(old,new)
        for name in a:
            np.testing.assert_array_equal(a[name],b[name])
        weighted=np.nansum(ga['SARIMA12_24Trend']*(counts/counts.sum()),axis=1)
        np.testing.assert_allclose(weighted,a['SARIMA12_24Trend'],atol=1e-10)

    def test_validation_stays_inside_training(self):
        self.assertEqual(validation_origins(110,24),[62,86])
        self.assertTrue(all(origin+24<=110 for origin in validation_origins(110,24)))

    def test_percentage_metric_zero_demand(self):
        self.assertEqual(metrics([0,0],[0,0])['WAPE_pct'],0)
        self.assertTrue(np.isnan(metrics([0,0],[1,1])['WAPE_pct']))

    def test_prediction_schema_and_ranges(self):
        frame=pd.DataFrame(dict(timestamp=[100,400],active_vm_count=[2,2],aggregate_cpu_utilization=[0.,100.]))
        validate_predictions(frame,[100,400])
        with self.assertRaises(ValueError): validate_predictions(frame.iloc[::-1],[100,400])
        frame.loc[0,'aggregate_cpu_utilization']=101
        with self.assertRaises(ValueError): validate_predictions(frame,[100,400])

    def test_empty_groups_and_incomplete_group_history(self):
        r,m=fixture(32); r.loc[0,'avg_cpu']=np.nan
        d=build_data(r,m)
        fleet,_,_,log=forecast(d,8,24)
        self.assertTrue(all(np.isfinite(v).all() for v in fleet.values()))
        self.assertTrue(any('fallback' in row['status'] for row in log))
        only=m.vm_id.iloc[0]
        single=build_data(r.loc[r.vm_id.eq(only)&r.timestamp.ne(600)],m)
        single_fleet,_,counts,_=forecast(single,7,24)
        self.assertEqual(int((counts>0).sum()),1)
        self.assertTrue(np.isfinite(single_fleet['SARIMA12_24Trend']).all())

    def test_main_from_arbitrary_directory_with_only_raw_files(self):
        r,m=fixture(32)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); raw=tmp/'raw'; raw.mkdir()
            r.to_csv(raw/'vm_cpu_readings.csv',index=False); m.to_csv(raw/'vm_metadata.csv',index=False)
            env=os.environ.copy(); env.pop('PYTHONPATH',None); env['PYTHONDONTWRITEBYTECODE']='1'
            outputs=[]
            for evaluate in (False,True):
                out=tmp/('eval.csv' if evaluate else 'predictions.csv')
                command=[sys.executable,'-B',str(ROOT/'main.py'),'--data-dir',str(raw),'--output',str(out),
                    '--results-dir',str(tmp/('eval_results' if evaluate else 'results')),'--skip-analysis']
                if evaluate: command.append('--evaluate')
                result=subprocess.run(command,cwd=tmp,env=env,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                outputs.append(pd.read_csv(out))
            pd.testing.assert_frame_equal(*outputs)
            self.assertEqual(len(outputs[0]),24)
            self.assertEqual(outputs[0].active_vm_count.iloc[0],44)
            self.assertEqual(outputs[0].timestamp.iloc[0],3000)


if __name__=='__main__': unittest.main()
