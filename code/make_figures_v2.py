from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[2]
RES=ROOT/'reproducibility'/'results_v2'
FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)
summary=pd.read_csv(RES/'submission_summary_v2.csv')

def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=350,bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

# Fig 1: workflow schematic
fig,ax=plt.subplots(figsize=(11.2,4.8))
ax.set_xlim(0,12); ax.set_ylim(0,5); ax.axis('off')
boxes=[
    (0.2,1.75,2.0,1.45,'Yeh/UCI data\n1,030 rows\n427 mix designs'),
    (2.75,1.75,2.15,1.45,'Group by exact\n7-component mix\n(all ages locked)'),
    (5.45,2.65,2.25,1.35,'Graded SCM shift\n$\\lambda$ = 0, 0.75,\n1.5, 2.5'),
    (5.45,0.75,2.25,1.35,'Hard SCM-regime\nholdout + 28-day\ncontrol'),
    (8.25,1.75,1.85,1.45,'Fixed Extra Trees\ntrain + calibration\nby mix group'),
    (10.55,2.65,1.25,1.35,'Split CP\n90%'),
    (10.55,0.75,1.25,1.35,'Estimated\nWCP\n90%')]
for x,y,w,h,t in boxes:
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',fill=False,linewidth=1.3)
    ax.add_patch(p); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9.5)
for a,b in [((2.2,2.48),(2.75,2.48)),((4.9,2.48),(5.45,3.32)),((4.9,2.48),(5.45,1.42)),((7.7,3.32),(8.25,2.7)),((7.7,1.42),(8.25,2.25)),((10.1,2.48),(10.55,3.32)),((10.1,2.48),(10.55,1.42))]:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,linewidth=1.1))
ax.text(6.55,4.55,'Deployment stress tests',ha='center',va='center',fontsize=11,fontweight='bold')
ax.text(11.17,4.35,'Evaluate: coverage, finite width,\nunbounded fraction, ESS, support',ha='center',va='center',fontsize=8.8)
ax.text(11.17,0.25,'Unbounded intervals are treated as\na failure to provide finite transfer, not a success metric.',ha='center',va='center',fontsize=8.3)
fig.tight_layout()
save(fig,'fig1_evaluation_workflow')

# Graded subset helper
G=summary[summary.scenario_type.eq('graded')].copy()
G['severity']=G['scenario'].str.replace('SCM-enrichment-','',regex=False).astype(float)

# Fig 2 coverage
fig,ax=plt.subplots(figsize=(7.2,4.8))
for method,label,marker in [('Split-CP','Split CP','o'),('Estimated-WCP','Estimated WCP','s')]:
    d=G[G.method.eq(method)].sort_values('severity')
    ax.errorbar(d.severity,d.coverage_mean,yerr=d.coverage_std,marker=marker,capsize=3,label=label)
ax.axhline(0.90,linestyle='--',linewidth=1,label='Nominal 90%')
ax.set_xlabel('SCM-enrichment severity, $\\lambda$')
ax.set_ylabel('Empirical coverage')
ax.set_ylim(0.76,1.01)
ax.set_xticks([0,0.75,1.5,2.5])
ax.legend(frameon=False)
ax.grid(alpha=.2)
fig.tight_layout()
save(fig,'fig2_graded_coverage')

# Fig 3 overlap/cost
D=G[G.method.eq('Estimated-WCP')].sort_values('severity')
fig,ax=plt.subplots(figsize=(7.5,4.9))
ax.plot(D.severity,D.support_rate_mean,marker='o',label='95th-percentile support rate')
ax.plot(D.severity,D.infinite_frac_mean,marker='s',label='Unbounded-interval fraction')
ax.set_xlabel('SCM-enrichment severity, $\\lambda$')
ax.set_ylabel('Fraction')
ax.set_ylim(0,1.02)
ax.set_xticks([0,0.75,1.5,2.5])
ax2=ax.twinx()
ax2.plot(D.severity,D.mean_width_finite_mean,marker='^',linestyle='--',label='Mean finite width')
ax2.set_ylabel('Finite interval width (MPa)')
ax2.set_ylim(18, max(27, float(D.mean_width_finite_mean.max()) + 1.5))
lines=ax.get_lines()+ax2.get_lines(); labels=[l.get_label() for l in lines]
ax.legend(lines,labels,frameon=False,loc='center left')
ax.grid(alpha=.2)
fig.tight_layout()
save(fig,'fig3_overlap_cost')

# Fig 4 hard regimes: split CP all-age vs 28 day
order=['No-SCM','Slag-only','Fly-ash-only','Dual-SCM']
allage=summary[(summary.scenario_type.eq('regime'))&(summary.method.eq('Split-CP'))].copy()
d28=summary[(summary.scenario_type.eq('regime_28d'))&(summary.method.eq('Split-CP'))].copy()
allage['regime']=allage.scenario.str.replace('Holdout-','',regex=False)
d28['regime']=d28.scenario.str.replace('28d-Holdout-','',regex=False)
A=allage.set_index('regime').loc[order]; B=d28.set_index('regime').loc[order]
x=np.arange(len(order)); off=.12
fig,ax=plt.subplots(figsize=(7.7,4.9))
ax.errorbar(x-off,A.coverage_mean,yerr=A.coverage_std,fmt='o',capsize=3,label='All ages')
ax.errorbar(x+off,B.coverage_mean,yerr=B.coverage_std,fmt='s',capsize=3,label='28-day only')
ax.axhline(.90,linestyle='--',linewidth=1,label='Nominal 90%')
ax.set_xticks(x,order)
ax.set_ylabel('Split-CP empirical coverage')
ax.set_ylim(0.15,1.02)
ax.legend(frameon=False)
ax.grid(axis='y',alpha=.2)
fig.tight_layout()
save(fig,'fig4_hard_regime_coverage')
print('wrote',len(list(FIG.glob('*'))),'figure files to',FIG)
