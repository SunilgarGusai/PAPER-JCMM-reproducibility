"""PAPER022 sensitivity analyses for the mix-design-grouped revision.
No model tuning is performed. The Random-Forest check uses 10 fixed seeds.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('mainmod', HERE/'paper022_analysis_v2.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
OUT=ROOT/'results_v2'; OUT.mkdir(exist_ok=True)

RF_SEEDS=list(range(20260901,20260911))
RF_CONFIG=dict(n_estimators=100,min_samples_leaf=2,max_features=1.0,n_jobs=1)


def rf_eval(source,target,seed,scenario):
    train,cal=m.split_source_groups(source,seed)
    model=RandomForestRegressor(random_state=seed,**RF_CONFIG)
    model.fit(train[m.XCOLS],train['CMS'])
    s=np.abs(cal['CMS'].to_numpy()-model.predict(cal[m.XCOLS]))
    q=m.finite_sample_q(s)
    pred=model.predict(target[m.XCOLS])
    y=target['CMS'].to_numpy()
    return {'scenario':scenario,'seed':seed,
            'coverage':float(np.mean((y>=pred-q)&(y<=pred+q))),
            'mae':float(np.mean(np.abs(y-pred))),
            'width':float(2*q)}


def support_rate(source,target,q):
    cols=['scm_frac','slag_frac','fly_frac','wb','sp_b','ca_b','fa_b']
    su=m.mix_group_frame(source)
    sc=StandardScaler().fit(su[cols])
    Xs=sc.transform(su[cols]); Xt=sc.transform(target[cols])
    nn_src=NearestNeighbors(n_neighbors=2).fit(Xs)
    dsrc=nn_src.kneighbors(Xs,return_distance=True)[0][:,1]
    thr=float(np.quantile(dsrc,q))
    nn=NearestNeighbors(n_neighbors=1).fit(Xs)
    dt=nn.kneighbors(Xt,return_distance=True)[0][:,0]
    return float(np.mean(dt<=thr))


def main():
    df=m.load_data(); rows=[]
    for sev in [0.0,2.5]:
        for seed in RF_SEEDS:
            src,tgt=m.weighted_target_split(df,seed,sev)
            rows.append(rf_eval(src,tgt,seed,f'SCM-enrichment-{sev:g}'))
    for fam in ['No-SCM','Slag-only','Fly-ash-only','Dual-SCM']:
        src=df[df.family!=fam].copy(); tgt=df[df.family==fam].copy()
        for seed in RF_SEEDS:
            rows.append(rf_eval(src,tgt,seed,f'Holdout-{fam}'))
    rf=pd.DataFrame(rows)
    rf.to_csv(OUT/'model_sensitivity_random_forest_v2.csv',index=False)
    rf.groupby('scenario', as_index=False).agg(
        coverage_mean=('coverage','mean'), coverage_std=('coverage','std'),
        mae_mean=('mae','mean'), mae_std=('mae','std'),
        width_mean=('width','mean'), width_std=('width','std')
    ).to_csv(OUT/'model_sensitivity_random_forest_summary_v2.csv', index=False)

    qvals=[0.90,0.95,0.975,0.99]; srows=[]
    for prefix,d0 in [('',df),('28d-',df[df.Age==28].copy())]:
        for fam in ['No-SCM','Slag-only','Fly-ash-only','Dual-SCM']:
            src=d0[d0.family!=fam].copy(); tgt=d0[d0.family==fam].copy()
            row={'scenario':f'{prefix}Holdout-{fam}'}
            for q in qvals: row[f'support_q{100*q:g}']=support_rate(src,tgt,q)
            srows.append(row)
    for sev in [0.0,0.75,1.5,2.5]:
        vals={q:[] for q in qvals}
        for seed in m.SEEDS:
            src,tgt=m.weighted_target_split(df,seed,sev)
            for q in qvals: vals[q].append(support_rate(src,tgt,q))
        row={'scenario':f'SCM-enrichment-{sev:g}'}
        for q in qvals: row[f'support_q{100*q:g}']=float(np.mean(vals[q]))
        srows.append(row)
    pd.DataFrame(srows).to_csv(OUT/'support_threshold_sensitivity_v2.csv',index=False)

if __name__=='__main__': main()
