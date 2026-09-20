import json, hashlib, platform, sys, os, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import sklearn
from joblib import Parallel, delayed

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'Concrete_Data.csv'
OUT=ROOT/'results_v2'; LOG=ROOT/'logs'
OUT.mkdir(exist_ok=True); LOG.mkdir(exist_ok=True)
ALPHA=0.10
SEEDS=list(range(20260901,20260931))
XCOLS=['Cement','Blast','Fly Ash','Water','Superplasticizer','CA','FA','Age']
MIXCOLS=['Cement','Blast','Fly Ash','Water','Superplasticizer','CA','FA']
AGE_ORDER=['<=7','8-28','29-100','>100']


def strat_age(age):
    age=np.asarray(age)
    return np.select([age<=7,age<=28,age<=100,age>100],AGE_ORDER,default='>100')


def load_data():
    df=pd.read_csv(DATA)
    assert df.shape==(1030,9)
    assert df.isna().sum().sum()==0
    cond=[
        (df['Blast']==0)&(df['Fly Ash']==0),
        (df['Blast']>0)&(df['Fly Ash']==0),
        (df['Blast']==0)&(df['Fly Ash']>0),
        (df['Blast']>0)&(df['Fly Ash']>0)]
    df['family']=np.select(cond,['No-SCM','Slag-only','Fly-ash-only','Dual-SCM'],default='Other')
    binder=df['Cement']+df['Blast']+df['Fly Ash']
    df['binder']=binder
    df['scm_frac']=(df['Blast']+df['Fly Ash'])/binder
    df['slag_frac']=df['Blast']/binder
    df['fly_frac']=df['Fly Ash']/binder
    df['wb']=df['Water']/binder
    df['sp_b']=df['Superplasticizer']/binder
    df['ca_b']=df['CA']/binder
    df['fa_b']=df['FA']/binder
    # Exact 8-predictor identity is retained for audit only.
    df['x8_group']=pd.util.hash_pandas_object(df[XCOLS], index=False).astype(str)
    # Evaluation unit: exact 7-component mix design. All curing ages of one mix stay together.
    df['mix_group']=pd.util.hash_pandas_object(df[MIXCOLS], index=False).astype(str)
    df['age_stratum']=strat_age(df['Age'])
    return df


def mix_group_frame(d):
    # one row per mix design with an age-profile label capturing every age stratum present
    base=d.drop_duplicates('mix_group').copy()
    prof=(d.groupby('mix_group')['age_stratum']
            .agg(lambda x: '+'.join([a for a in AGE_ORDER if a in set(x)])))
    base['age_profile']=base['mix_group'].map(prof)
    return base


def split_source_groups(source, seed, cal_frac=0.25):
    ug=mix_group_frame(source)
    # Prefer regime + age-profile stratification; back off to regime where cells are sparse.
    strata=ug['family'].astype(str)+'|'+ug['age_profile'].astype(str)
    vc=strata.value_counts()
    strata2=strata.where(strata.map(vc)>=4, ug['family'].astype(str))
    try:
        train_g, cal_g=train_test_split(ug['mix_group'], test_size=cal_frac,
                                        random_state=seed, stratify=strata2)
    except ValueError:
        train_g, cal_g=train_test_split(ug['mix_group'], test_size=cal_frac,
                                        random_state=seed, stratify=ug['family'])
    train=source[source['mix_group'].isin(set(train_g))].copy()
    cal=source[source['mix_group'].isin(set(cal_g))].copy()
    assert set(train.mix_group).isdisjoint(set(cal.mix_group))
    return train,cal


def fit_model(train, seed):
    m=ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2,
                          max_features=1.0, random_state=seed, n_jobs=1)
    m.fit(train[XCOLS],train['CMS'])
    return m


def finite_sample_q(scores, alpha=ALPHA):
    scores=np.sort(np.asarray(scores,float))
    n=len(scores)
    k=min(n, int(math.ceil((n+1)*(1-alpha))))
    return scores[k-1]


def composition_features(d):
    return pd.DataFrame({
      'Cement':d['Cement'].values,'Blast':d['Blast'].values,'FlyAsh':d['Fly Ash'].values,
      'Water':d['Water'].values,'SP':d['Superplasticizer'].values,'CA':d['CA'].values,'FA':d['FA'].values,
      'logAge':np.log1p(d['Age'].values),
      'scm_frac':d['scm_frac'].values,'slag_frac':d['slag_frac'].values,'fly_frac':d['fly_frac'].values,
      'wb':d['wb'].values,'sp_b':d['sp_b'].values,'ca_b':d['ca_b'].values,'fa_b':d['fa_b'].values,
      'has_slag':(d['Blast'].values>0).astype(float),'has_fly':(d['Fly Ash'].values>0).astype(float)
    })


def estimate_ratio(source, target, seed):
    Xs=composition_features(source); Xt=composition_features(target)
    X=pd.concat([Xs,Xt],ignore_index=True)
    y=np.r_[np.zeros(len(Xs),dtype=int),np.ones(len(Xt),dtype=int)]
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=5000,random_state=seed))
    clf.fit(X,y)
    p_src=np.clip(clf.predict_proba(Xs)[:,1],1e-6,1-1e-6)
    p_tgt=np.clip(clf.predict_proba(Xt)[:,1],1e-6,1-1e-6)
    prior_corr=len(Xt)/len(Xs)
    w_src=(p_src/(1-p_src))/prior_corr
    w_tgt=(p_tgt/(1-p_tgt))/prior_corr
    # In-sample AUC is retained only as a descriptive separability score, not a generalization metric.
    auc=roc_auc_score(y,clf.predict_proba(X)[:,1])
    ess=(w_src.sum()**2)/(np.square(w_src).sum()+1e-12)
    return w_src,w_tgt,auc,ess


def weighted_q_per_target(scores, w_cal, w_test, alpha=ALPHA):
    scores=np.asarray(scores,float); w_cal=np.asarray(w_cal,float); w_test=np.asarray(w_test,float)
    order=np.argsort(scores); s=scores[order]; w=w_cal[order]
    cum=np.cumsum(w)
    out=np.empty(len(w_test))
    total_cal=w.sum()
    for j,wt in enumerate(w_test):
        threshold=(1-alpha)*(total_cal+wt)
        if cum[-1] + 1e-15 < threshold:
            out[j]=np.inf
        else:
            idx=np.searchsorted(cum,threshold,side='left')
            out[j]=s[min(idx,len(s)-1)]
    return out


def support_flags(source,target):
    cols=['scm_frac','slag_frac','fly_frac','wb','sp_b','ca_b','fa_b']
    su=mix_group_frame(source)
    scaler=StandardScaler().fit(su[cols])
    Xs=scaler.transform(su[cols]); Xt=scaler.transform(target[cols])
    if len(Xs)<3:
        return np.ones(len(target),dtype=bool),np.full(len(target),np.nan),np.nan
    nn_src=NearestNeighbors(n_neighbors=2).fit(Xs)
    dsrc=nn_src.kneighbors(Xs,return_distance=True)[0][:,1]
    thr=float(np.quantile(dsrc,0.95))
    nn=NearestNeighbors(n_neighbors=1).fit(Xs)
    dt=nn.kneighbors(Xt,return_distance=True)[0][:,0]
    return dt<=thr,dt,thr


def metrics(y,pred,lo,hi,support=None):
    y=np.asarray(y); pred=np.asarray(pred); lo=np.asarray(lo); hi=np.asarray(hi)
    covered=(y>=lo)&(y<=hi)
    width=hi-lo
    finite=np.isfinite(width)
    res={
      'n':len(y),'mae':mean_absolute_error(y,pred),'rmse':mean_squared_error(y,pred)**0.5,
      'r2':r2_score(y,pred),'coverage':covered.mean(),
      'mean_width_finite':float(width[finite].mean()) if finite.any() else np.inf,
      'median_width_finite':float(np.median(width[finite])) if finite.any() else np.inf,
      'infinite_frac':float((~finite).mean()),
      'coverage_finite':float(covered[finite].mean()) if finite.any() else np.nan,
    }
    if support is not None:
        support=np.asarray(support,bool)
        res['support_rate']=support.mean()
        res['coverage_supported']=covered[support].mean() if support.any() else np.nan
        res['coverage_unsupported']=covered[~support].mean() if (~support).any() else np.nan
        res['mae_supported']=mean_absolute_error(y[support],pred[support]) if support.any() else np.nan
        res['mae_unsupported']=mean_absolute_error(y[~support],pred[~support]) if (~support).any() else np.nan
    return res


def evaluate_partition(source,target,seed,scenario,scenario_type,severity=np.nan):
    train,cal=split_source_groups(source,seed)
    model=fit_model(train,seed)
    pred_cal=model.predict(cal[XCOLS]); scores=np.abs(cal['CMS'].values-pred_cal)
    pred=model.predict(target[XCOLS])
    q=finite_sample_q(scores)
    lo=pred-q; hi=pred+q
    support,sdist,sth=support_flags(source,target)
    m=metrics(target['CMS'],pred,lo,hi,support)
    m.update({'seed':seed,'scenario':scenario,'scenario_type':scenario_type,'severity':severity,
              'method':'Split-CP','q':q,'n_train':len(train),'n_cal':len(cal),
              'n_train_mix':train['mix_group'].nunique(),'n_cal_mix':cal['mix_group'].nunique(),
              'n_target_mix':target['mix_group'].nunique(),'support_threshold':sth,
              'target_scm_mean':target['scm_frac'].mean(),'source_scm_mean':source['scm_frac'].mean(),
              'target_age_mean':target['Age'].mean(),'source_age_mean':source['Age'].mean()})
    rows=[m]

    w_source,w_target,auc,ess=estimate_ratio(source,target,seed)
    ws=pd.Series(w_source,index=source.index)
    wcal=ws.loc[cal.index].values
    qx=weighted_q_per_target(scores,wcal,w_target)
    lo2=pred-qx; hi2=pred+qx
    m2=metrics(target['CMS'],pred,lo2,hi2,support)
    m2.update({'seed':seed,'scenario':scenario,'scenario_type':scenario_type,'severity':severity,
               'method':'Estimated-WCP','q':np.nan,'n_train':len(train),'n_cal':len(cal),
               'n_train_mix':train['mix_group'].nunique(),'n_cal_mix':cal['mix_group'].nunique(),
               'n_target_mix':target['mix_group'].nunique(),'support_threshold':sth,
               'domain_auc':auc,'weight_ess':ess,'weight_ess_frac':ess/len(source),
               'target_scm_mean':target['scm_frac'].mean(),'source_scm_mean':source['scm_frac'].mean(),
               'target_age_mean':target['Age'].mean(),'source_age_mean':source['Age'].mean()})
    rows.append(m2)
    return rows


def weighted_target_split(df,seed,severity,target_frac=0.25):
    ug=mix_group_frame(df).copy()
    rng=np.random.default_rng(seed)
    z=(ug['scm_frac']-ug['scm_frac'].mean())/(ug['scm_frac'].std()+1e-12)
    ug['_z']=z
    # Merge very rare age profiles to avoid a singleton stratum.
    vc=ug['age_profile'].value_counts()
    ug['_profile']=ug['age_profile'].where(ug['age_profile'].map(vc)>=4,'RARE')
    target_groups=[]
    for _,g in ug.groupby('_profile'):
        if len(g)<2:
            continue
        k=max(1,int(round(target_frac*len(g))))
        k=min(k,len(g)-1)
        logits=np.clip(severity*g['_z'].values,-20,20)
        w=np.exp(logits-logits.max()); w=w/w.sum()
        chosen=rng.choice(g['mix_group'].values,size=k,replace=False,p=w)
        target_groups.extend(chosen.tolist())
    target=df[df['mix_group'].isin(set(target_groups))].copy()
    source=df[~df['mix_group'].isin(set(target_groups))].copy()
    assert set(source.mix_group).isdisjoint(set(target.mix_group))
    return source,target


def main():
    df=load_data()
    audit={
      'n_rows':len(df),
      'n_unique_exact_8_predictor_rows':df['x8_group'].nunique(),
      'n_unique_7_component_mix_designs':df['mix_group'].nunique(),
      'duplicate_rows_by_exact_8_predictors':len(df)-df['x8_group'].nunique(),
      'rows_beyond_unique_mix_designs':len(df)-df['mix_group'].nunique(),
      'regime_counts_rows':df['family'].value_counts().to_dict(),
      'regime_counts_mix_designs':mix_group_frame(df).groupby('family').size().to_dict(),
      'age_counts':df['Age'].value_counts().sort_index().to_dict(),
      'sha256':hashlib.sha256(DATA.read_bytes()).hexdigest()
    }
    (OUT/'data_audit_v2.json').write_text(json.dumps(audit,indent=2,default=str))

    rows=[]
    for sev in [0.0,0.75,1.5,2.5]:
      for seed in SEEDS:
        source,target=weighted_target_split(df,seed,sev)
        rows.extend(evaluate_partition(source,target,seed,f'SCM-enrichment-{sev:g}','graded',sev))

    fams=['No-SCM','Slag-only','Fly-ash-only','Dual-SCM']
    for fam in fams:
      source=df[df.family!=fam].copy(); target=df[df.family==fam].copy()
      for seed in SEEDS:
        rows.extend(evaluate_partition(source,target,seed,f'Holdout-{fam}','regime',np.nan))

    d28=df[df.Age==28].copy()
    for fam in fams:
      source=d28[d28.family!=fam].copy(); target=d28[d28.family==fam].copy()
      for seed in SEEDS:
        rows.extend(evaluate_partition(source,target,seed,f'28d-Holdout-{fam}','regime_28d',np.nan))

    res=pd.DataFrame(rows)
    res.to_csv(OUT/'all_replicates_v2.csv',index=False)
    keys=['coverage','mean_width_finite','infinite_frac','coverage_finite','mae','rmse','r2',
          'support_rate','coverage_supported','coverage_unsupported','domain_auc','weight_ess_frac',
          'target_scm_mean','source_scm_mean','target_age_mean','source_age_mean',
          'n_train_mix','n_cal_mix','n_target_mix']
    compact=res.groupby(['scenario_type','scenario','method'])[keys].agg(['mean','std']).round(6)
    compact.to_csv(OUT/'submission_summary_v2.csv')
    counts=res.groupby(['scenario_type','scenario','method']).size().rename('n_replicates').reset_index()
    counts.to_csv(OUT/'repetition_counts_v2.csv',index=False)
    env={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
         'seeds':SEEDS,'alpha':ALPHA,
         'evaluation_group':'exact seven-component mix design; all ages of a mix remain in one partition',
         'model':'ExtraTreesRegressor(n_estimators=100,min_samples_leaf=2,max_features=1.0,n_jobs=1)',
         'density_ratio':'StandardScaler + LogisticRegression(C=1); probabilities clipped [1e-6,1-1e-6]; prior-corrected odds',
         'support':'95th percentile source leave-one-out 1NN distance over unique mix designs in standardized composition-ratio space'}
    (OUT/'environment_and_design_v2.json').write_text(json.dumps(env,indent=2))
    print(json.dumps(audit,indent=2))
    for st in ['graded','regime','regime_28d']:
      print('\n###',st)
      sub=res[res.scenario_type==st]
      show=sub.groupby(['scenario','method'])[['coverage','mean_width_finite','infinite_frac','mae','support_rate','domain_auc','weight_ess_frac']].mean().round(3)
      print(show.to_string())

if __name__=='__main__':
    main()
