#!/usr/bin/env python3
"""Generate every main figure and energy table from stored 1.2 records."""
from pathlib import Path
import argparse,json,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_plot_style import save_figure
from matplotlib.patches import FancyBboxPatch
from analyze_v12 import LABELS,OUT
from energy_v12 import workload,decode,score
ROOT=Path(__file__).resolve().parent;DEST=ROOT/'output/v1_2'
COLORS={'milp':'#34495e','heuristic':'#009e73','local_search':'#56b4e9','classical_uniform_onehot':'#cc79a7','classical_uniform_full':'#999999','quantum_uniform_onehot':'#e69f00','quantum_uniform_full':'#7f8c8d','onehot_transfer':'#d55e00','onehot_structured':'#0072b2','onehot_p2':'#6c3483','qaoa_transfer':'#c0392b','qaoa_online':'#795548'}
CHOSEN=[('milp',0),('heuristic',0),('local_search',0),('classical_uniform_full',16),('classical_uniform_onehot',64),('quantum_uniform_full',16),('quantum_uniform_onehot',64),('qaoa_online',64),('qaoa_transfer',64),('onehot_transfer',64),('onehot_structured',64),('onehot_p2',64)]

def fmt(x,d=2):
 if pd.isna(x) or np.isinf(x):return '--'
 if abs(x)<.5*10**(-d):x=0.
 return f'{x:,.{d}f}'

def table(filename,headers,rows,align=None):
 align=align or 'l'+'r'*(len(headers)-1)
 text=['\\begin{longtable}{'+align+'}',r'\toprule',' & '.join(headers)+r'\\',r'\midrule',r'\endhead']
 text+=[' & '.join(map(str,row))+r'\\' for row in rows];text += [r'\bottomrule',r'\end{longtable}']
 (DEST/filename).write_text('\n'.join(text)+'\n')

def tables(s):
 rows=[]
 for alg,shots in CHOSEN:
  r=s[(s.algorithm==alg)&(s.shots==shots)].iloc[0]
  rows.append([LABELS[alg],shots or '--',f'{r.p_quality:.3f}/{r.p_quality_median:.3f}',f'{100*r.quality:.1f}/{100*r.timely:.1f}',fmt(r.compute_j),f'{fmt(r.excess_j)} [{fmt(r.ci_low_j)}, {fmt(r.ci_high_j)}]',f'{r.wins:.0f}/60'])
 text=[r'\begin{tabular}{lrccrrr}',r'\toprule',r'Pipeline & Shots & $p_\epsilon$ mean/median & Quality/timely (\%) & Compute (J) & $\Delta E$ [95\% CI] (J) & Wins\\',r'\midrule']
 text+=[' & '.join(map(str,row))+r'\\' for row in rows];text+=[r'\bottomrule',r'\end{tabular}'];(DEST/'v12_principal.tex').write_text('\n'.join(text))
 # Complete budget table and family/physical diagnostics.
 rows=[]
 for r in s.itertuples():
  rows.append([LABELS[r.algorithm],r.shots or '--',fmt(100*r.quality,1),fmt(100*r.useful,1),fmt(r.compute_per_useful_j),f'{fmt(r.excess_j)} [{fmt(r.ci_low_j)}, {fmt(r.ci_high_j)}]',f'{r.wins:.0f}/{r.ties:.0f}/{r.losses:.0f}'])
 table('v12_budgets.tex',['Pipeline','$S$','Quality \\%','Useful \\%','J/useful','$\\Delta E$ [95\\% CI] J','W/T/L'],rows)
 rows=[]
 for name in ['all_on','sleep_0','sleep_10','sleep_50']:
  for f in json.loads((OUT/name/'transfer.json').read_text()):
   if 'depth' not in f:continue
   rows.append([name.replace('_',' '),f['size'],f['depth'],', '.join(fmt(v,2) for v in f['theta']),', '.join(fmt(v,2) for v in f['circular_resultant']),fmt(f['offline_s'],3)])
 table('v12_angles.tex',['Family','$n$','$p$','Angles (rad)','Resultant lengths','Training s'],rows,'lrrllr')
 util=pd.read_csv(OUT/'utilizations.csv');rows=[];pr=[]
 for name in ['all_on','sleep_0','sleep_10','sleep_50']:
  inst=pd.read_csv(OUT/name/'instances.csv');inst=inst[inst.seed>=15];cl=pd.read_csv(OUT/name/'classical.csv');cl=cl[(cl.seed>=15)&(cl.algorithm=='milp')].set_index(['size','seed'])
  same=[int(r.incumbent)==int(cl.loc[(r.size,r.seed),'solution']) for r in inst.itertuples()]
  pr.append([name.replace('_',' '),fmt(100*inst.perturbation.mean(),2),int((inst.halvings>0).sum()),fmt(100*np.mean(same),1),fmt(100*np.mean(abs(inst.incumbent_w-inst.optimum_w)<1e-8),1)])
  for stage in ['previous','optimum']:
   v=util[(util.workload==name)&(util.seed>=15)&(util.stage==stage)].utilization
   rows.append([name.replace('_',' '),stage,fmt(v.quantile(.1),3),fmt(v.median(),3),fmt(v.quantile(.9),3),fmt(100*np.mean(v>0),1)])
 table('v12_utilization.tex',['Family','Placement','$u_{10}$','$u_{50}$','$u_{90}$','Nonempty \\%'],rows,'llrrrr')
 table('v12_incumbents.tex',['Family','Mean max perturb. \\%','Halved cases','Identical assignment \\%','Current-optimal power \\%'],pr)
 rows=[]
 ci=pd.read_csv(OUT/'all_on/circuits.csv');ci=ci[(ci.seed>=15)&ci.is_quantum]
 for (n,alg),g in ci.groupby(['size','algorithm']):
  rows.append([n,LABELS[alg],fmt(g.cx.median(),0),fmt(g.shot_s.median()*1e6,1),fmt(g.compile_s.median(),3)])
 table('v12_resources.tex',['$n$','Pipeline','Median CX','Median shot $\\mu$s','Compile s'],rows)
 speed=pd.read_csv(OUT/'service_compression.csv');rows=[]
 for r in speed.itertuples():
  if r.algorithm in ['onehot_transfer','onehot_structured','onehot_p2','quantum_uniform_onehot'] and r.shots==64:
   rows.append([LABELS[r.algorithm],r.service_duration_factor,fmt(100*r.useful,1),fmt(r.compute_j),fmt(r.excess_j)])
 table('v12_speed.tex',['Pipeline','Duration factor','Useful \\%','Compute J','$\\Delta E$ J'],rows)
 # Compact per-size table: same configurations as main table, no missing sizes.
 ps=pd.read_csv(OUT/'all_on/per_size.csv');rows=[]
 for r in ps.itertuples():
  if (r.algorithm,int(r.shots)) in CHOSEN:rows.append([r.size,LABELS[r.algorithm],r.shots or '--',fmt(100*r.quality,1),fmt(100*r.timely,1),fmt(100*r.useful,1),fmt(r.compute_j),f'{fmt(r.excess_j)} [{fmt(r.ci_low_j)}, {fmt(r.ci_high_j)}]',f'{r.wins:.0f}/15'])
 table('v12_per_size.tex',['$n$','Pipeline','$S$','Quality \\%','Timely \\%','Useful \\%','Compute J','$\\Delta E$ [95\\% CI] J','Wins'],rows,'llrrrrrrl')
 ins=pd.read_csv(OUT/'all_on/instances.csv');ins=ins[ins.seed>=15];rows=[]
 for n,g in ins.groupby('size'):
  values=[]
  for alg in ['quantum_uniform_full','quantum_uniform_onehot','onehot_transfer','qaoa_transfer']:
   x=ps[(ps['size']==n)&(ps.algorithm==alg)&(ps.shots==64)].iloc[0];values.append(f'{x.p_quality:.3f}')
  rows.append([n,int(g.onehot_states.iloc[0]),f'{g.cooptimal_states.min()}--{g.cooptimal_states.max()}',fmt(g.uniform_optimum_probability.mean(),3)]+values)
 table('v12_uniform.tex',['$n$','$N^F$','Co-optima','$p^*_{unif,1h}$','Full $p_\\epsilon$','One-hot $p_\\epsilon$','QAOA 1h $p_\\epsilon$','QAOA X $p_\\epsilon$'],rows)
 sr=pd.read_csv(OUT/'all_on/shot_requirements.csv');sr=sr[sr.seed>=15];rows=[]
 for (n,alg),g in sr.groupby(['size','algorithm']):
  if alg.startswith('classical') or alg in ['milp','local_search','heuristic']:continue
  vals=[]
  for eps,alpha in [(.05,.9),(.05,.95),(0,.9),(0,.95)]:
   x=g[(g.epsilon==eps)&(g.alpha==alpha)].required_shots;vals.append(f'{fmt(x.median(),0)} ({fmt(x.max(),0)})')
  rows.append([n,LABELS[alg]]+vals)
 table('v12_shots.tex',['$n$','Pipeline','5\\%,90\\%','5\\%,95\\%','Exact,90\\%','Exact,95\\%'],rows)
 th=pd.read_csv(OUT/'all_on/thresholds.csv');rows=[]
 for (alg,shots),g in th.groupby(['algorithm','shots']):
  if shots!=64:continue
  r=g.minimum_reuse.dropna();p=g.qpu_power_threshold_w.dropna()
  rows.append([LABELS[alg],len(r),fmt(r.median(),0),fmt(r.max(),0),len(p),fmt(p.median())])
 table('v12_reuse.tex',['Pipeline, 64 shots','Finite $M$','Median $M$','Maximum $M$','Power crossings','Median threshold W'],rows)
 grid=pd.read_csv(OUT/'all_on/sensitivity.csv');rows=[]
 for eta in [.001,.01,.1]:
  for alg,shots in [('quantum_uniform_full',16),('quantum_uniform_full',64),('quantum_uniform_onehot',64),('onehot_transfer',64),('onehot_structured',64),('onehot_p2',64)]:
   g=grid[(grid.boundary=='marginal')&(grid.marginal_fraction==eta)&(grid.algorithm==alg)&(grid.shots==shots)]
   primary=g[(g.profile=='optimistic')&(g.local_w==50)&(g.qpu_w==20000)&(g.reuse==100)&(g.horizon==60)&(g.migration_j==10)].iloc[0]
   rows.append([eta,LABELS[alg],shots,fmt(primary.excess_j),fmt(g.excess_j.min()),int((g.excess_j< -1e-8).sum())])
 table('v12_marginal.tex',['$\\eta$','Pipeline','$S$','Primary $\\Delta E$ J','Minimum grid $\\Delta E$ J','Favorable cells'],rows)
 rows=[]
 cir=pd.read_csv(OUT/'all_on/circuits.csv');qu=pd.read_csv(OUT/'all_on/qubo_audit.csv')
 for n in [6,8,9,12]:
  g=qu[(qu['size']==n)&(qu.seed>=15)];rows.append([n,int(g.auxiliary_qubits.iloc[0]),int(g.qubo_qubits.iloc[0]),int(g.quadratic_terms.median()),fmt(g.penalty.median(),1),fmt(g.cx.median(),0),fmt(g.shot_s.median()*1e6,1),g.target.iloc[0]])
 table('v12_qubo.tex',['Assignment bits','Auxiliary','Total','Quadratic terms','Penalty','CX','$t_s$ ($\\mu$s)','Target'],rows,'rrrrrrrl')
 rows=[]
 for name in ['sleep_0','sleep_10','sleep_50']:
  d=pd.read_csv(OUT/name/'primary_summary.csv')
  for alg,shots in [('milp',0),('heuristic',0),('quantum_uniform_onehot',64),('onehot_structured',64),('onehot_p2',64)]:
   r=d[(d.algorithm==alg)&(d.shots==shots)].iloc[0];rows.append([name.split('_')[1],LABELS[alg],shots or '--',fmt(100*r.quality,1),fmt(r.compute_j),f'{fmt(r.excess_j)} [{fmt(r.ci_low_j)}, {fmt(r.ci_high_j)}]'])
 table('v12_sleep.tex',['Sleep W','Pipeline','$S$','Quality \\%','Compute J','$\\Delta E$ [95\\% CI] J'],rows,'rlrrrr')
 overhead=pd.read_csv(OUT/'all_on/overheads.csv');rows=[]
 for r in overhead.itertuples():
  if r.algorithm not in ['quantum_uniform_onehot','onehot_structured','onehot_p2'] or r.shots!=64:continue
  rows.append([LABELS[r.algorithm],int(r.repetition_s*1e6),int(r.setup_s*1e3),r.utilization,fmt(100*r.timely,1),fmt(r.compute_j),fmt(r.excess_j)])
 table('v12_overheads.tex',['Pipeline','$t_r$ $\\mu$s','$t_j$ ms','$u$','Timely \\%','Compute J','$\\Delta E$ J'],rows)
 pol=pd.read_csv(OUT/'all_on/policy_summary.csv');rows=[]
 for r in pol.itertuples():rows.append([r.policy.replace('_',' '),fmt(100*r.invoked_quantum,1),fmt(100*r.useful,1),fmt(100*r.applied_quality,1),fmt(r.compute_j),f'{fmt(r.excess_j)} [{fmt(r.ci_low_j)}, {fmt(r.ci_high_j)}]'])
 table('v12_policies.tex',['Policy','Quantum \\%','Useful candidate \\%','Applied quality \\%','Compute J','$\\Delta E$ J'],rows)
 # Transition sensitivity table includes the original H=1,migration=100 stress corner.
 mig=pd.read_csv(OUT/'all_on/migration_sensitivity.csv');conc=pd.read_csv(OUT/'all_on/concentrated_incumbent.csv');rows=[]
 for name,d in [('Previous optimum',mig),('Concentrated',conc)]:
  for r in d.itertuples():
   if r.horizon not in [1,60,3600] or r.migration_j not in [10,100] or r.algorithm not in ['onehot_transfer','onehot_structured','quantum_uniform_onehot'] or r.shots!=64:continue
   if name=='Previous optimum' and r.migration_s not in [0,.1]:continue
   rows.append([name,r.horizon,r.migration_j,fmt(1000*r.migration_s,0),'on' if r.benefit_gate else 'off',LABELS[r.algorithm],fmt(r.excess_j)])
 table('v12_transitions.tex',['Incumbent','$H$ s','Move J','Move ms','Gate','Pipeline','$\\Delta E$ J'],rows,'lrrrllr')
 st=pd.read_csv(OUT/'stress_decomposition.csv');rows=[]
 for r in st.itertuples():
  if (r.algorithm,int(r.shots)) in [('onehot_transfer',64),('qaoa_transfer',64),('local_search',0)]:rows.append([LABELS[r.algorithm],fmt(r.compute_j),fmt(r.transition_j),fmt(r.delay_loss_j),fmt(r.quality_loss_j),fmt(r.fallback_loss_j),fmt(r.total_j)])
 table('v12_stress_decomposition.tex',['Pipeline','Compute','Transition','Arrival','Quality','Fallback','Total'],rows)
 noise=OUT/'noise_instances.csv'
 if noise.exists():
  nr=pd.read_csv(noise);rows=[]
  for (alg,rate),g in nr[nr.shots==64].groupby(['algorithm','noise']):rows.append([LABELS[alg],rate,fmt(g.p_quality.mean(),3),fmt(g.required_shots.median(),0),fmt(g.quality.mean()*100,1),fmt(g.compute_j.mean()),fmt(g.excess_j.mean())])
  table('v12_noise.tex',['Pipeline','$\\lambda$','$p_\\epsilon$','Median $S_{90}$','Quality \\%','Compute J','$\\Delta E$ J'],rows)
 else:(DEST/'v12_noise.tex').write_text('Noise computation is in progress.\n')
 return grid


def figures(s,grid):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':6.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 pi=pd.read_csv(OUT/'all_on/primary_instances.csv');pi=pi[pi.seed>=15]
 ins=pd.read_csv(OUT/'all_on/instances.csv');ins=ins[ins.seed>=15]
 case=ins[ins.incumbent_w-ins.optimum_w>1e-7].sort_values(['size','seed']).iloc[0]
 fig=plt.figure(figsize=(7.2,4.6));card_text=[('Previous placement',f'n={case["size"]}, seed={case.seed}\n{case.incumbent_w:,.1f} W\nDeadline {1000*case.deadline_s:.0f} ms'),('Physical optimizer',f'HiGHS optimum\n{case.optimum_w:,.1f} W\nAll servers powered'),('Sampling reference',f'{int(case.onehot_states)} one-hot states\n{int(case.cooptimal_states)} co-optimal states\nUniform P(opt)={100*case.uniform_optimum_probability:.2f}%'),('Actuation gate','Feasible and timely\nRemaining savings\nexceed transition cost')]
 for k,(title,body) in enumerate(card_text):
  x=.025+k*.245;box=FancyBboxPatch((x,.69),.225,.26,boxstyle='round,pad=.008',transform=fig.transFigure,facecolor=['#e8eef3','#e0f1e9','#fff1d3','#e2eff8'][k],edgecolor='#536878',linewidth=.8);fig.add_artist(box)
  fig.text(x+.1125,.915,title,ha='center',weight='bold',fontsize=8);fig.text(x+.1125,.865,body,ha='center',va='top',linespacing=1.5,fontsize=8)
 a=fig.add_axes([.08,.14,.39,.42]);b=fig.add_axes([.59,.14,.38,.42]);chosen=[('milp',0),('classical_uniform_onehot',64),('quantum_uniform_onehot',64),('onehot_structured',64),('onehot_transfer',64),('qaoa_online',64)]
 names=['MILP','CPU uniform','Q uniform','Structured','Dense 1h','Online'];rs=[s[(s.algorithm==alg)&(s.shots==sh)].iloc[0] for alg,sh in chosen]
 a.barh(names,[r.compute_j for r in rs],color=[COLORS[x[0]] for x in chosen]);a.set_xscale('log');a.set_xlabel('Mean computation energy (J)');a.set_title('(a) Matched 64-shot comparisons',loc='left');a.invert_yaxis()
 for j,r in enumerate(rs[:-1]):b.errorbar(r.excess_j,j,xerr=[[max(0,r.excess_j-r.ci_low_j)],[max(0,r.ci_high_j-r.excess_j)]],fmt='o',color=COLORS[chosen[j][0]],capsize=2)
 b.set_yticks(range(5),names[:5]);b.invert_yaxis();b.axvline(0,color='#888',lw=.6);b.set_xlabel('Extra total energy versus MILP (J)');b.set_title('(b) Paired means and 95% intervals',loc='left')
 save_figure(fig,DEST/'fig1_overview.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
 f=pd.read_csv(OUT/'frontier.csv');fig,ax=plt.subplots(1,2,figsize=(7.2,2.7),layout='constrained')
 for alg in ['onehot_transfer','onehot_structured','onehot_p2','quantum_uniform_onehot']:
  g=f[f.algorithm==alg];ax[0].scatter(g.time_ratio,g.required_j,s=13,alpha=.7,label=LABELS[alg],color=COLORS[alg]);budgets=np.geomspace(.01,1e5,200)
  ax[1].plot(budgets,[np.mean(g.admissible&(g.required_j<=v))*100 for v in budgets],label=LABELS[alg],color=COLORS[alg])
 ax[0].axvline(1,c='black',ls='--',lw=.8);ax[0].set(xscale='log',yscale='log',xlabel='Required arrival / deadline',ylabel='Required computation energy (J)',title='(a) Per-instance frontier corners')
 ax[1].set(xscale='log',xlabel='Computation budget (J)',ylabel='Joint-feasible instances (%)',title='(b) Deadline-limited coverage');fig.legend(*ax[1].get_legend_handles_labels(),loc='outside upper center',ncol=4,fontsize=6.5,frameon=False)
 save_figure(fig,DEST/'frontier_cases.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(7.2,3.6),layout='constrained')
 for alg in ['classical_uniform_full','quantum_uniform_full','classical_uniform_onehot','quantum_uniform_onehot','onehot_transfer','onehot_structured','onehot_p2','qaoa_transfer','qaoa_online']:
  g=s[s.algorithm==alg].sort_values('shots');ax[0].errorbar(100*g.useful,g.compute_j,xerr=np.maximum(0,100*np.array([g.useful-g.useful_ci_low,g.useful_ci_high-g.useful])),fmt='o-',ms=3,elinewidth=.5,capsize=1,color=COLORS[alg],label=LABELS[alg])
  for r in g.itertuples():
   if alg=='onehot_structured':ax[0].annotate(str(r.shots),(100*r.useful,r.compute_j),xytext=(3,2),textcoords='offset points',fontsize=6)
 for alg in ['milp','heuristic']:
  r=s[s.algorithm==alg].iloc[0];ax[0].scatter(100*r.useful,r.compute_j,marker='s',s=20,color=COLORS[alg],label=LABELS[alg])
 ax[0].set(yscale='log',xlabel='Timely, feasible, 5%-quality candidate (%)',ylabel='Mean computation energy (J)',title='(a) Budget-quality trade-off');fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',fontsize=6,ncol=3)
 c=pd.read_csv(OUT/'all_on/circuits.csv');c=c[c.seed>=15]
 for alg in ['onehot_transfer','onehot_structured','onehot_p2','quantum_uniform_onehot']:
  g=c[c.algorithm==alg].groupby('size').shot_s;med=g.median();ax[1].plot(med.index,med*1e6,'o-',color=COLORS[alg],label=LABELS[alg]);ax[1].fill_between(med.index,g.quantile(.25)*1e6,g.quantile(.75)*1e6,color=COLORS[alg],alpha=.15)
 ax[1].set(yscale='log',xlabel='Assignment variables',ylabel='Scheduled shot duration (µs)',title='(b) Complete grid-target circuits');ax[1].legend(fontsize=6)
 save_figure(fig,DEST/'quality_energy.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
 g=pd.read_csv(OUT/'horizon_intervals.csv')
 for alg in ['milp','classical_uniform_onehot','quantum_uniform_onehot','onehot_structured','onehot_transfer']:
  x=g[(g.algorithm==alg)&(g.shots==(0 if alg=='milp' else 64))].sort_values('horizon');ax[0].plot(x.horizon,-x.excess_j/1000,'o-',color=COLORS[alg],label=LABELS[alg],ms=3);ax[0].fill_between(x.horizon,-x.ci_high_j/1000,-x.ci_low_j/1000,color=COLORS[alg],alpha=.12)
 ax[0].axhline(0,c='#777',lw=.6);ax[0].set(xscale='log',yscale='symlog',xlabel='Constant-demand horizon (s)',ylabel='Net saving versus MILP (kJ)',title='(a) Previous-optimum incumbent');ax[0].legend(fontsize=5.8,loc='lower left')
 dec=pd.read_csv(OUT/'all_on/decomposition.csv');algs=['quantum_uniform_onehot','onehot_structured','onehot_transfer'];base=np.zeros(3)
 for col,label,color in [('compute_j','Computation','#0072b2'),('delay_loss_j','Arrival','#e69f00'),('quality_loss_j','Placement quality','#009e73'),('fallback_loss_j','Fallback/rejection','#cc79a7'),('transition_j','Transitions','#999999')]:
  values=np.array([dec[(dec.algorithm==a)&(dec.shots==64)][col].mean()/1000 for a in algs]);ax[1].bar(np.arange(3),values,bottom=base,label=label,color=color);base+=values
 ax[1].set_xticks(range(3),['Q uniform','Structured','Dense 1h']);ax[1].set(ylabel='Extra total energy versus MILP (kJ)',title='(b) Additive decomposition, 64 shots');ax[1].legend(fontsize=6,loc='upper left')
 save_figure(fig,DEST/'net_energy.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
 pol=pd.read_csv(OUT/'all_on/policy_summary.csv').set_index('policy');names=['always_classical','always_quantum_selected','deadline_only','energy_aware','quantum_with_backup'];labels=['Classical','Selected Q','Deadline','Energy','Q + backup'];p=pol.loc[names]
 fig,ax=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained');bars=ax[0].bar(labels,p.excess_j,color='#3579a8',yerr=np.maximum(0,np.array([p.excess_j-p.ci_low_j,p.ci_high_j-p.excess_j])),capsize=2,error_kw={'elinewidth':.7});ax[0].set_yscale('symlog',linthresh=.1);ax[0].bar_label(bars,labels=[fmt(v,2) for v in p.excess_j],padding=3,fontsize=6);ax[0].set(ylabel='Extra total energy versus direct MILP (J)',title='(a) Calibration and lookup included');ax[0].tick_params(axis='x',labelsize=6.5)
 x=np.arange(5);ax[1].bar(x-.18,p.useful,width=.36,label='Useful candidate',color='#3579a8');ax[1].bar(x+.18,p.invoked_quantum,width=.36,label='Quantum invocation',color='#e69f00');ax[1].set_xticks(x,labels,fontsize=6.5);ax[1].set(ylim=(0,1.22),ylabel='Fraction',title='(b) Training-only policy evaluation');ax[1].legend(loc='upper center',ncol=2,fontsize=6)
 save_figure(fig,DEST/'admission.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def macros(s,grid,write_headline=True):
 values={}
 for prefix,alg,shots in [('Milp','milp',0),('Greedy','heuristic',0),('ClassUniform','classical_uniform_onehot',64),('QuantumUniform','quantum_uniform_onehot',64),('Dense','onehot_transfer',64),('Structured','onehot_structured',64),('DepthTwo','onehot_p2',64)]:
  r=s[(s.algorithm==alg)&(s.shots==shots)].iloc[0]
  for suffix,val in [('Compute',r.compute_j),('Excess',r.excess_j),('Low',r.ci_low_j),('High',r.ci_high_j),('Quality',100*r.quality),('Useful',100*r.useful),('Timely',100*r.timely),('Wins',r.wins)]:values[prefix+suffix]=fmt(val,0 if suffix=='Wins' else 1 if suffix in ['Quality','Useful','Timely'] else 2)
 q=grid[~grid.algorithm.isin(['milp','heuristic','local_search'])&~grid.algorithm.str.startswith('classical')];w=q[q.excess_j< -1e-8];values['FavorableCells']=str(len(w));values['FavorableScenarios']=str(w.scenario_id.nunique());values['BestMarginal']=fmt(w.excess_j.min(),3)
 ins=pd.read_csv(OUT/'all_on/instances.csv');ins=ins[ins.seed>=15];values['WarmOptimal']=str(int((abs(ins.incumbent_w-ins.optimum_w)<1e-8).sum()));values['HeadroomW']=fmt((ins.incumbent_w-ins.optimum_w).mean(),2)
 pol=pd.read_csv(OUT/'all_on/policy_summary.csv').set_index('policy')
 for name,prefix in [('always_quantum_selected','PolicyQ'),('energy_aware','PolicyEnergy'),('quantum_with_backup','PolicyBackup')]:
  for col,suffix in [('excess_j','Excess'),('compute_j','Compute')]:values[prefix+suffix]=fmt(pol.loc[name,col])
  values[prefix+'Useful']=fmt(100*pol.loc[name,'useful'],1)
 noise=pd.read_csv(OUT/'noise_instances.csv')
 for alg,prefix in [('onehot_structured','NoiseStructured'),('onehot_transfer','NoiseDense')]:
  for rate,suffix in [(0.,'Zero'),(.01,'High')]:
   g=noise[(noise.algorithm==alg)&(noise.noise==rate)&(noise.shots==64)]
   values[prefix+suffix+'Probability']=fmt(g.p_quality.mean(),3)
   values[prefix+suffix+'Quality']=fmt(100*g.quality.mean(),1)
 values['MultipleOptima']=str(int((ins.cooptimal_states>1).sum()))
 (DEST/'v12_numbers.tex').write_text('\n'.join('\\newcommand{\\V'+k+'}{'+v+'}' for k,v in values.items())+'\n')
 if write_headline:(OUT/'headline_numbers.json').write_text(json.dumps(values,indent=2))


def captions():
 text={'Overview': 'NetworkQBench-Wireless decision accounting. The cards trace one declared held-out placement case; the panels summarize all COUNT held-out cases in the primary allocated scenario. The incumbent is a previous optimum under higher demand, and a beneficial transition is required before actuation. Quantum and network joules are modeled. Bars and intervals are generated from stored records, with a size-stratified paired 95\\% interval for the total-energy contrast. Diagram and plotting code were prepared with OpenAI Codex assistance and are included in the artifact.', 'Frontier': 'Resource frontier for fixed transferred circuits and zero-layer controls. Each point is one held-out instance and pipeline. Each corner uses the instance-specific probability required for a 90\\% chance of 5\\%-quality output. The primary circuit target and operation-duration scenario are held fixed. Postprocessing overhead is frozen at its measured 64-shot reference for this diagnostic. The figure is noiseless and uses evaluation-only probabilities. Separately matched noisy-target results appear in the supplement.', 'Quality': '(a) Mean computation energy versus useful-candidate rate for the four budgets, with structured-circuit budgets annotated. Full-space and one-hot uniform controls share ideal sample outcomes across classical and quantum implementations but have different execution costs. Horizontal intervals are size-stratified 95\\% instance bootstrap intervals. (b) Median complete-shot durations with interquartile bands across held-out instances; narrow bands can be difficult to distinguish from the median line. All resource comparisons use the same grid target.', 'Net': '(a) Net operating-plus-computation saving relative to matched MILP across constant-demand horizons. Positive values favor the displayed pipeline; shading gives paired 95\\% instance intervals. (b) Exact decomposition of the primary 64-shot contrast. Each term is a paired difference, so transition terms can be negative. Units are kJ; near-zero contrasts are retained in the numerical tables.', 'Admission': 'Training-only controllers in the primary scenario. (a) Total-energy contrasts and paired 95\\% intervals include calibration acquisition, amortized training, lookup, and every attempted computation. Values near zero are explicitly labeled. (b) Useful candidate and quantum-invocation rates. All controllers preserve the same benefit-aware fallback; retaining an already good incumbent is distinct from producing a useful candidate.'}
 count=int(pd.read_csv(OUT/'all_on/instances.csv').query('seed >= 15').shape[0])
 (DEST/'v12_captions.tex').write_text('\n'.join('\\newcommand{\\V'+key+'Caption}{'+value.replace('COUNT',str(count))+'}' for key,value in text.items())+'\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manuscript',type=Path);args=ap.parse_args();DEST.mkdir(parents=True,exist_ok=True)
 s=pd.read_csv(OUT/'all_on/primary_summary.csv');grid=tables(s);figures(s,grid);macros(s,grid,write_headline=False);captions()
 if args.manuscript:
  for p in DEST.iterdir():
   target=args.manuscript/('figures' if p.suffix=='.pdf' else 'tables');target.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target/p.name)
 print('generated main figures and tables',DEST,flush=True)
if __name__=='__main__':main()
