from pathlib import Path
import importlib.util, json
import pandas as pd, numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('m', HERE/'paper022_analysis_v2.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
res=pd.read_csv(ROOT/'results_v2'/'all_replicates_v2.csv')
df=m.load_data()
checks=[]

def compare_rows(scenario, seed, source, target, stype, sev=np.nan):
    fresh=pd.DataFrame(m.evaluate_partition(source,target,seed,scenario,stype,sev))
    saved=res[(res.scenario==scenario)&(res.seed==seed)].copy()
    for method in ['Split-CP','Estimated-WCP']:
        a=fresh[fresh.method==method].iloc[0]; b=saved[saved.method==method].iloc[0]
        for col in ['coverage','mae','rmse','r2','mean_width_finite','infinite_frac','support_rate']:
            av=a[col]; bv=b[col]
            if np.isinf(av) and np.isinf(bv): diff=0.0
            elif pd.isna(av) and pd.isna(bv): diff=0.0
            else: diff=abs(float(av)-float(bv))
            checks.append({'scenario':scenario,'seed':seed,'method':method,'field':col,'difference':diff,'pass':diff<1e-12})

seed=20260901
src,tgt=m.weighted_target_split(df,seed,2.5)
compare_rows('SCM-enrichment-2.5',seed,src,tgt,'graded',2.5)
src=df[df.family!='No-SCM'].copy(); tgt=df[df.family=='No-SCM'].copy()
compare_rows('Holdout-No-SCM',seed,src,tgt,'regime',np.nan)

# structural checks
counts=res.groupby(['scenario_type','scenario','method']).size()
struct={
 'all_primary_cells_have_30': bool((counts==30).all()),
 'n_cells': int(len(counts)),
 'mix_designs': int(df.mix_group.nunique()),
 'exact8': int(df.x8_group.nunique()),
 'rows': int(len(df)),
 'sentinel_all_pass': bool(all(x['pass'] for x in checks)),
 'max_abs_difference': float(max(x['difference'] for x in checks))
}
out={'structural':struct,'sentinel_checks':checks}
(ROOT/'logs'/'v2_sentinel_reproduction_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(struct,indent=2))
