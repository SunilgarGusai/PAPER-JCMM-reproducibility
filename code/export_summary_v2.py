"""Flatten the primary replicate-level output into the manuscript summary table."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_v2'
res=pd.read_csv(OUT/'all_replicates_v2.csv')
keys=['coverage','mean_width_finite','infinite_frac','coverage_finite','mae','rmse','r2',
      'support_rate','coverage_supported','coverage_unsupported','domain_auc','weight_ess_frac',
      'target_scm_mean','source_scm_mean','target_age_mean','source_age_mean',
      'n_train_mix','n_cal_mix','n_target_mix']
compact=res.groupby(['scenario_type','scenario','method'])[keys].agg(['mean','std'])
compact.columns=[f'{metric}_{stat}' for metric,stat in compact.columns]
compact=compact.reset_index()
compact.to_csv(OUT/'submission_summary_v2.csv',index=False)
print('wrote', OUT/'submission_summary_v2.csv')
