#!/usr/bin/env python3
"""Standalone frozen-data renderer for revision 1.2.1; no external plot kit."""
import argparse,json,shutil
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_plot_style import save_figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter,FixedLocator
import render_v12 as old
from revision_v121 import ROOT,FROZEN,RESULTS

LABELS={**old.LABELS,'keep_incumbent':'Keep incumbent','useful_mass_transfer':'Useful-mass transfer'}
COLORS={**old.COLORS,'keep_incumbent':'#555555'}
CHOSEN=[('keep_incumbent',0)]+[x for x in old.CHOSEN if x[0]!='qaoa_online']

def get(d,alg,shots=0):return d[(d.algorithm==alg)&(d.shots==shots)].iloc[0]
def fmt(v,n=2):return old.fmt(v,n)
def tex_table(dest,name,headers,rows,long=True):
    env='longtable' if long else 'tabular'
    text=['\\begin{'+env+'}{'+'l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(headers)+r'\\',r'\midrule']
    if long:text+=[r'\endhead']
    text+=[' & '.join(str(x) for x in r)+r'\\' for r in rows]
    text+=[r'\bottomrule','\\end{'+env+'}'];(dest/name).write_text('\n'.join(text)+'\n')

def interval(r,prefix=''):
    return f"{fmt(r['greedy_excess_j' if prefix else 'excess_j'])} [{fmt(r['greedy_low_j' if prefix else 'ci_low_j'])}, {fmt(r['greedy_high_j' if prefix else 'ci_high_j'])}]"

def tables(dest,data):
    s=pd.read_csv(data/'primary_summary.csv');rows=[]
    for alg,shots in CHOSEN:
        r=get(s,alg,shots);stay=alg=='keep_incumbent'
        rows.append([LABELS[alg],shots or '--','--' if stay else f'{r.p_quality:.3f}/{r.p_quality_median:.3f}',
                     '--' if stay else f'{100*r.quality:.1f}/{100*r.timely:.1f}','--' if stay else fmt(100*r.useful,1),
                     fmt(r.compute_j),interval(r),interval(r,'greedy'),f'{r.wins:.0f}/60'])
    tex_table(dest,'v12_principal.tex',['Pipeline','$S$','$p_\\epsilon$ mean/median','Quality/timely \\%','Useful \\%','Compute J','$\\Delta E_{\\rm HiGHS}$ [95\\% CI] J','$\\Delta E_{\\rm greedy}$ [95\\% CI] J','Wins vs. HiGHS'],rows,False)
    d=pd.read_csv(data/'decomposition.csv');rows=[]
    for boundary in ['allocated','marginal']:
        for r in d[(d.boundary==boundary)&(d.algorithm=='onehot_structured')].itertuples():
            rows.append([boundary,r.component.replace('_j','').replace('_',' '),fmt(r.quantum_j,3),fmt(r.classical_j,3),fmt(r.difference_j,3)])
    tex_table(dest,'v121_ledger.tex',['Boundary','Term','QAOA J','HiGHS J','Difference J'],rows)
    d=pd.read_csv(data/'training_summary.csv');rows=[]
    for (family,n,p),g in d.groupby(['workload','size','depth']):
        tr=g[g.split=='train'].iloc[0];te=g[g.split=='test'].iloc[0]
        rows.append([family.replace('_',' '),n,p,fmt(tr.uniform,3),fmt(tr.fitted,3),fmt(tr.transferred,3),fmt(te.uniform,3),fmt(te.transferred,3)])
    tex_table(dest,'v121_training.tex',['Family','$n$','$p$','Train uniform','Train fitted','Train transfer','Test uniform','Test transfer'],rows)
    d=pd.read_csv(data/'aligned_summary.csv');rows=[]
    for r in d[(d.algorithm=='useful_mass_transfer')&(d.shots==64)].itertuples():
        rows.append([r.workload.replace('_',' '),fmt(r.p_quality,3),fmt(100*r.quality,1),fmt(100*r.useful,1),fmt(r.offline_j,3),fmt(r.compute_j),interval(r._asdict())])
    tex_table(dest,'v121_aligned.tex',['Family','$p_\\epsilon$','Quality \\%','Useful \\%','Offline J','Compute J','$\\Delta E$ [95\\% CI] J'],rows)
    d=pd.read_csv(data/'demand_summary.csv');rows=[]
    for alg,shots in [('milp',0),('heuristic',0),('classical_uniform_onehot',64),('onehot_structured',64),('onehot_transfer',64)]:
        r=get(d,alg,shots);rows.append([LABELS[alg],fmt(100*r.quality,1),fmt(100*r.useful,1),fmt(r.compute_j),interval(r)])
    tex_table(dest,'v121_demand.tex',['Pipeline','Quality \\%','Useful \\%','Compute J','$\\Delta E$ [95\\% CI] J'],rows)
    q=pd.read_csv(data/'larger_summary.csv');res=pd.read_csv(data/'larger_resources.csv');rows=[]
    for r in q.itertuples():
        g=res[res['size']==r.size]
        rows.append([r.size,r.shots,int(r.onehot_states),fmt(r.uniform_p_quality,3),fmt(100*r.exact_batch_quality,2),fmt(100*r.sampled_quality,1),fmt(r.structured_p_quality,3),fmt(g.cx.median(),0),fmt(g.shot_s.median()*1e6,1)])
    tex_table(dest,'v121_larger.tex',['$n$','$S$','$N^F$','Uniform $p_\\epsilon$','Exact batch \\%','Observed \\%','QAOA $p_\\epsilon$','Median CX','Shot $\\mu$s'],rows)
    d=pd.read_csv(data/'scaling_summary.csv');rows=[]
    for r in d.itertuples():
        rows.append([r.nodes,r.functions,int(r.exact),int(r.feasible),fmt(r.median_build_s,3),fmt(r.median_solve_s,3),fmt(r.median_greedy_s,4),fmt(r.median_gap,4),fmt(r.mean_milp_compute_j),fmt(r.mean_greedy_compute_j)])
    tex_table(dest,'v121_scaling.tex',['Servers','Functions','Exact/10','Feasible/10','Build s','Solve s','Greedy s','MIP gap','MILP J','Greedy J'],rows)
    d=pd.read_csv(data/'timing_summary.csv');rounds=pd.read_csv(data/'timing_rounds.csv');rows=[]
    for alg,sh in [('milp',0),('heuristic',0),('classical_uniform_onehot',64),('classical_uniform_onehot',256)]:
        r=get(d,alg,sh);g=rounds[(rounds.algorithm==alg)&(rounds.shots==sh)]
        rows.append([LABELS[alg],sh or '--',f'{1000*g.runtime_s.min():.3f}--{1000*g.runtime_s.max():.3f}',f'[{g.excess_j.min():.3f}, {g.excess_j.max():.3f}]',int((g.excess_j<0).sum()),interval(r)])
    tex_table(dest,'v121_timing.tex',['Pipeline','$S$','Round mean ms range','Round mean $\\Delta E$ J range','Negative/10','Pooled $\\Delta E$ [95\\% CI] J'],rows)
    n=pd.read_csv(data/'nominal_summary.csv');rows=[]
    for alg,sh in [('milp',0),('quantum_uniform_onehot',64),('onehot_structured',64),('onehot_transfer',64),('onehot_p2',64)]:
        r=get(n,alg,sh);rows.append([LABELS[alg],fmt(100*r.timely,1),fmt(100*r.useful,1),interval(r)])
    tex_table(dest,'v121_nominal.tex',['Pipeline','Timely \\%','Useful \\%','$\\Delta E$ [95\\% CI] J'],rows)
    return s


def figures(dest,data,s):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'legend.fontsize':6.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):save_figure(fig,dest/name,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    # One case, with a successful illustration clearly separated from averages.
    ins=pd.read_csv(FROZEN/'all_on/instances.csv');case=ins[(ins.seed>=15)&(ins.incumbent_w-ins.optimum_w>1e-7)].sort_values(['size','seed']).iloc[0]
    pi=pd.read_csv(data/'primary_instances.csv');p=pi[(pi['size']==case['size'])&(pi.seed==case.seed)];mil=get(p,'milp');q=get(p,'onehot_structured',64)
    fig=plt.figure(figsize=(7.4,4.7));extra=q.total_j-mil.total_j
    cards=[('1  Previous optimum',f'Demand falls; keep incumbent\n{case.incumbent_w:,.2f} W\n{case.onehot_states:.0f} one-hot assignments\nDeadline: {1000*case.deadline_s:.0f} ms'),
           ('2  HiGHS',f'Optimum: {case.optimum_w:,.2f} W\nArrival: {1000*mil.ready_s:.1f} ms\nCompute: {mil.compute_j:.2f} J\nBenefit gate checks transition'),
           ('3  Structured QAOA',f'64 shots, p = 1\nArrival: {1000*q.ready_s:.1f} ms\nP(optimum): {100*q.p_optimal:.2f}%\nUniform: {100*case.uniform_optimum_probability:.2f}%'),
           ('4  Mean of 30 batches',f'HiGHS total: {mil.total_j:,.2f} J\nQAOA total: {q.total_j:,.2f} J\nQAOA extra: {extra:.2f} J\nServer gap: {q.network_j-mil.network_j:.2f} J')]
    for k,(title,body) in enumerate(cards):
        x=.018+k*.247;fig.add_artist(FancyBboxPatch((x,.66),.229,.30,boxstyle='round,pad=.008',transform=fig.transFigure,facecolor=['#eef2f5','#eaf5ef','#eaf1fa','#fff2e6'][k],edgecolor='#607080',linewidth=.7))
        fig.text(x+.011,.925,title,fontsize=8,weight='bold');fig.text(x+.011,.88,body,va='top',fontsize=6.8,linespacing=1.65)
    fig.text(.03,.615,'A timely optimum is one possible QAOA outcome; card 4 averages all sampled outcomes.',fontsize=7)
    algs=['heuristic','classical_uniform_onehot','quantum_uniform_onehot','onehot_structured','onehot_transfer'];names=['Greedy','CPU uniform','Q uniform','Structured','Dense one-hot'];rs=[get(s,a,0 if a=='heuristic' else 64) for a in algs]
    ax=fig.add_axes([.09,.15,.37,.36]);ax.bar(range(5),[r.compute_j for r in rs],color=[COLORS[a] for a in algs]);ax.set_yscale('log');ax.set_xticks(range(5),names,rotation=25,ha='right',fontsize=6.5);ax.set_ylabel('Computation energy (J)');ax.set_title('(a) Primary 60-case means',loc='left')
    ax=fig.add_axes([.62,.15,.35,.36]);ax.errorbar([r.excess_j for r in rs],range(5),xerr=np.maximum(0,np.array([[r.excess_j-r.ci_low_j for r in rs],[r.ci_high_j-r.excess_j for r in rs]])),fmt='o',capsize=2,color='#0072b2');ax.set_yticks(range(5),names,fontsize=6.5);ax.invert_yaxis();ax.axvline(0,color='#777',lw=.7);ax.set_xlabel('Extra total energy vs. HiGHS (J)');ax.set_title('(b) Paired 95% intervals',loc='left');save(fig,'fig1_cartoon.pdf')
    cap=r'One held-out case (cards) and primary results over 60 cases (panels). Card 3 gives the distribution and arrival of a 64-shot attempt; a successful optimum draw is illustrative, whereas card 4 averages all 30 batches, including suboptimal outcomes within the five-percent tolerance. Thus its server-energy difference is not solely the delay of the optimum action. Totals and their contrast use unrounded data and are displayed to two decimals. All joules are modeled; quantum quality is simulated.'
    vals={'FigOneExtraJ':f'{extra:.2f}','FigOneSavedW':f'{case.incumbent_w-case.optimum_w:.0f}','FigOneQaoaPopt':f'{100*q.p_optimal:.1f}','FigOneUniformPopt':f'{100*case.uniform_optimum_probability:.2f}','FigOneQaoaJ':f'{q.compute_j:.1f}','FigOneMilpJ':f'{mil.compute_j:.2f}'}
    (dest/'fig1_cartoon.tex').write_text('\\newcommand{\\FigOneCartoonCaption}{'+cap+'}\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in vals.items()))
    # Classical / quantum trade-offs and resources.
    fig,ax=plt.subplots(1,3,figsize=(7.4,3.0),layout='constrained')
    groups=[['classical_uniform_full','classical_uniform_onehot'],['quantum_uniform_onehot','onehot_structured','onehot_transfer','onehot_p2']]
    for k,algs in enumerate(groups):
        for alg in algs:
            g=s[s.algorithm==alg].sort_values('shots');ax[k].errorbar(100*g.useful,g.compute_j,xerr=np.maximum(0,100*np.array([g.useful-g.useful_ci_low,g.useful_ci_high-g.useful])),fmt='o-',ms=3,elinewidth=.5,capsize=1,color=COLORS[alg],label=LABELS[alg])
            # Every quantum curve is parametric in the shot budget S, so each point carries its S.
            # Per-budget offsets (dx, dy, ha) keep labels off the horizontal error bars and other lines.
            offsets={'onehot_structured':{16:(-3,3,'right'),64:(-3,3,'right'),256:(4,0,'left'),1024:(3,3,'left')},
                     'quantum_uniform_onehot':{16:(0,-4,'center'),64:(4,0,'left'),256:(4,0,'left'),1024:(4,0,'left')},
                     'onehot_transfer':{16:(-3,3,'right'),64:(3,3,'left'),256:(-4,-2,'right'),1024:(3,3,'left')},
                     'onehot_p2':{16:(-3,3,'right'),64:(-3,3,'right'),256:(-6,-3,'right'),1024:(-3,3,'right')}}.get(alg)
            if k==1 and offsets:
                for r in g.itertuples():
                    dx,dy,ha=offsets[int(r.shots)]
                    ax[k].annotate(str(r.shots),(100*r.useful,r.compute_j),xytext=(dx,dy),textcoords='offset points',ha=ha,va='bottom' if dy>0 else ('top' if dy<0 else 'center'),fontsize=5.4,color=COLORS[alg])
        if k==0:
            for alg in ['milp','heuristic']:
                r=get(s,alg);ax[k].scatter(100*r.useful,r.compute_j,marker='s',s=15,color=COLORS[alg],label=LABELS[alg])
        ax[k].set(yscale='log',xlabel='Useful candidates (%)',ylabel='Computation energy (J)',title=['(a) Classical controls','(b) Quantum configurations'][k])
        if k==1:ax[k].set_ylim(2,1e5);ax[k].set_xlim(right=104.5);ax[k].legend(loc='upper right',fontsize=5.6)   # room for budget labels and the legend
        else:ax[k].legend(loc='best',fontsize=5.6)
    circuits=pd.read_csv(FROZEN/'all_on/circuits.csv');circuits=circuits[circuits.seed>=15]
    for alg in groups[1]:
        g=circuits[circuits.algorithm==alg].groupby('size').shot_s;med=g.median();ax[2].plot(med.index,med*1e6,'o-',ms=3,color=COLORS[alg],label=LABELS[alg]);ax[2].fill_between(med.index,g.quantile(.25)*1e6,g.quantile(.75)*1e6,color=COLORS[alg],alpha=.12)
    ax[2].set(yscale='log',xlabel='Assignment variables',ylabel='Shot duration (microseconds)',title='(c) Complete grid circuits');ax[2].set_xticks([6,8,9,12]);ax[2].legend(loc='upper left',fontsize=5.6);save(fig,'quality_energy.pdf')
    # Consistent sign; symmetric-log ticks use explicit round numbers.
    fig,ax=plt.subplots(1,2,figsize=(7.4,3.1),layout='constrained');h=pd.read_csv(data/'horizon_intervals.csv')
    for alg in ['keep_incumbent','classical_uniform_onehot','quantum_uniform_onehot','onehot_structured','onehot_transfer']:
        g=h[(h.algorithm==alg)&(h.shots==(0 if alg=='keep_incumbent' else 64))].sort_values('horizon');ax[0].plot(g.horizon,g.excess_j,'o-',ms=3,label=LABELS[alg],color=COLORS[alg]);ax[0].fill_between(g.horizon,g.ci_low_j,g.ci_high_j,alpha=.1,color=COLORS[alg])
    ax[0].set(xscale='log',yscale='symlog',xlabel='Constant-demand horizon (s)',ylabel='Extra total energy vs. HiGHS (J)',title='(a) Previous-optimum incumbent');ax[0].set_yscale('symlog',linthresh=10);ax[0].yaxis.set_major_locator(FixedLocator([-100,-10,0,10,100,1000,10000]));ax[0].yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'));ax[0].axhline(0,color='#888',lw=.7);ax[0].legend(fontsize=5.7)
    dec=pd.read_csv(FROZEN/'all_on/decomposition.csv');algs=['quantum_uniform_onehot','onehot_structured','onehot_transfer'];pos=np.zeros(3);neg=np.zeros(3)
    for col,label,color in [('compute_j','Computation','#0072b2'),('delay_loss_j','Arrival','#e69f00'),('quality_loss_j','Placement quality','#009e73'),('fallback_loss_j','Rejection','#cc79a7'),('transition_j','Transitions','#999999')]:
        v=np.array([dec[(dec.algorithm==a)&(dec.shots==64)][col].mean() for a in algs]);bottom=np.where(v>=0,pos,neg);ax[1].bar(range(3),v,bottom=bottom,label=label,color=color);pos+=np.maximum(v,0);neg+=np.minimum(v,0)
        if col=='transition_j':
            for j,x in enumerate(v):ax[1].annotate(f'transition {x:+.2f} J',(j,0),xytext=(0,-15),textcoords='offset points',ha='center',fontsize=5.7)
    ax[1].set_xticks(range(3),['Q uniform','Structured','Dense one-hot']);ax[1].tick_params(axis='x',pad=21);ax[1].set(ylabel='Extra total energy vs. HiGHS (J)',title='(b) Signed component differences');ax[1].legend(fontsize=5.6,loc='upper left');ax[1].axhline(0,c='#777',lw=.7);save(fig,'net_energy.pdf')
    # Invocation by size distinguishes policies without a near-constant quality panel.
    names=['always_classical','always_quantum_selected','deadline_only','energy_aware','quantum_with_backup'];labels=['Classical','Selected Q','Deadline','Energy','Q + backup']
    pol=pd.read_csv(FROZEN/'all_on/policy_summary.csv').set_index('policy').loc[names];ps=pd.read_csv(FROZEN/'all_on/policy_per_size.csv')
    fig,ax=plt.subplots(1,2,figsize=(7.4,2.8),layout='constrained');b=ax[0].bar(labels,pol.excess_j,color='#3579a8',yerr=np.maximum(0,np.array([pol.excess_j-pol.ci_low_j,pol.ci_high_j-pol.excess_j])),capsize=2);ax[0].set_yscale('log');ax[0].set_ylim(bottom=.01);ax[0].bar_label(b,labels=[fmt(x) for x in pol.excess_j],fontsize=6,padding=3);ax[0].set(ylabel='Extra total energy vs. HiGHS (J)',title='(a) All controller costs included');ax[0].tick_params(axis='x',labelsize=6)
    matrix=np.array([[100*ps[(ps.policy==p)&(ps['size']==n)].invoked_quantum.iloc[0] for n in [6,8,9,12]] for p in names]);energy=np.array([[ps[(ps.policy==p)&(ps['size']==n)].excess_j.iloc[0] for n in [6,8,9,12]] for p in names]);ax[1].imshow(energy,cmap='Blues',vmin=0,vmax=200,aspect='auto')
    for j in range(5):
        for k in range(4):ax[1].text(k,j,f'{matrix[j,k]:.0f}% / {energy[j,k]:.2f} J',ha='center',va='center',color='white' if energy[j,k]>100 else 'black',fontsize=5.5)
    ax[1].set_xticks(range(4),[6,8,9,12]);ax[1].set_yticks(range(5),labels);ax[1].set(xlabel='Assignment variables',title='(b) Quantum invocation / extra energy');save(fig,'admission.pdf')
    return {'Quality':r'(a) Classical and (b) quantum budget-quality comparisons, with at most four pipelines per panel. In (b), each point is one shot budget (16, 64, 256 or 1,024, as labeled), and lines join the budgets in increasing order. A curve turns left when longer runs miss more deadlines, because a useful candidate must be both good and timely. Bars are paired 95\% instance intervals, and other configurations are tabulated in the supplement. (c) Median complete-shot durations and interquartile bands on the same grid target.',
            'Net':r'(a) Total energy minus matched HiGHS across constant-demand horizons, including computing nothing. Positive values disfavor the displayed policy; bands are paired 95\% instance intervals. (b) Signed component differences at 64 shots, in joules, using the same sign convention. Transition differences are labeled because their bars are small.',
            'Admission':r'Training-only controllers. (a) Total-energy contrasts and paired 95\% intervals include calibration, lookup, and attempted computation. (b) Quantum invocation fractions by size. Each cell gives invocation percentage and mean energy excess, with color encoding the latter. No deadline-only fallback occurs in the primary cases. Useful-candidate and applied-quality rates remain in the supplement.'}


def macros(dest,data,s):
    vals={};diag=json.loads((data/'diagnostics.json').read_text());stay=get(s,'keep_incumbent')
    vals.update(BreakEven=f"{diag['break_even_us']:.2f}",KeepExcess=fmt(stay.excess_j),KeepLow=fmt(stay.ci_low_j),KeepHigh=fmt(stay.ci_high_j),KeepQuality=fmt(100*stay.applied_quality,1),HeadroomMax=fmt(diag['headroom_max_w']))
    for prefix,file in [('Marginal','marginal_summary.csv'),('Nominal','nominal_summary.csv'),('Component','component_summary.csv'),('Demand','demand_summary.csv')]:
        r=get(pd.read_csv(data/file),'onehot_structured',64)
        for key,col in [('Excess','excess_j'),('Compute','compute_j'),('Useful','useful'),('Timely','timely')]:vals[prefix+key]=fmt(r[col]*(100 if key in ['Useful','Timely'] else 1))
    vals['ComponentQaoaNegative']=str(diag['component_negative_qaoa_cells'])
    demand=pd.read_csv(data/'demand_incumbents.csv');vals.update(DemandOptimal=str(int((abs(demand.incumbent_w-demand.optimum_w)<1e-8).sum())),DemandHeadroom=fmt((demand.incumbent_w-demand.optimum_w).mean()),DemandHalved=str(int((demand.halvings>0).sum())),DemandPerturb=fmt(100*demand.realized_max.mean(),1))
    a=pd.read_csv(data/'aligned_summary.csv');r=a[(a.workload=='all_on')&(a.algorithm=='useful_mass_transfer')&(a.shots==64)].iloc[0]
    vals.update(AlignedProbability=fmt(r.p_quality,3),AlignedQuality=fmt(100*r.quality,1),AlignedExcess=fmt(r.excess_j),AlignedCompute=fmt(r.compute_j))
    for r in pd.read_csv(data/'larger_summary.csv').query('shots==64').itertuples():
        word={14:'Fourteen',16:'Sixteen'}[r.size];vals['Larger'+word+'Quality']=fmt(100*r.sampled_quality,1);vals['Larger'+word+'Exact']=fmt(100*r.exact_batch_quality,2)
    over=pd.read_csv(FROZEN/'all_on/overheads.csv');r=over[(over.algorithm=='onehot_structured')&(over.shots==64)&np.isclose(over.repetition_s,.0002)&np.isclose(over.setup_s,.1)&(over.utilization==1)].iloc[0];vals['OverheadCompute']=fmt(r.compute_j)
    post=pd.read_csv(data/'postprocessing.csv');g=post[post.algorithm=='onehot_structured'];vals['PostExtraMs']=fmt(1000*g.extra_1024_vs64_s.mean(),3);vals['PostExtraJ']=fmt(50*g.extra_1024_vs64_s.mean(),4)
    q=pd.read_csv(FROZEN/'all_on/qubo_audit.csv');q=q[(q.seed>=15)&(q['size']==12)];vals['QuboCX']=fmt(q.cx.median(),0);vals['QuboShotMs']=fmt(q.shot_s.median()*1000,3)
    rounds=pd.read_csv(data/'timing_rounds.csv');timing=pd.read_csv(data/'timing_summary.csv')
    parts=[]
    for alg,sh in [('heuristic',0),('classical_uniform_onehot',256)]:
        g=rounds[(rounds.algorithm==alg)&(rounds.shots==sh)];r=get(timing,alg,sh)
        label='Greedy' if alg=='heuristic' else 'Classical one-hot sampling at 256 shots'
        parts.append(f"{label} has a negative mean contrast in {int((g.excess_j<0).sum())} of ten timing rounds, with round means from {g.excess_j.min():.3f} to {g.excess_j.max():.3f}~J. Its instance-paired interval after averaging rounds is [{r.ci_low_j:.3f}, {r.ci_high_j:.3f}]~J.")
    (dest/'v121_timing_text.tex').write_text('\n'.join(parts)+'\n')
    fits=pd.read_csv(ROOT/'results/sustainability/fits.csv');fits=fits[fits.mixer=='x'];times=fits.simulation_s/fits.evals
    (dest/'v121_training_text.tex').write_text(f"For the standard-$X$ fits, recorded total fitting time divided by objective evaluations has mean {1e6*times.mean():.1f} and median {1e6*times.median():.1f}~$\\mu$s. This includes optimizer overhead and shows why 100~$\\mu$s should be read as an assumed host-update charge, not a measured exact-objective cost.\n")
    scaling=pd.read_csv(data/'scaling.csv');timeouts=int((scaling.status==1).sum());nofeasible=int((~scaling.feasible).sum())
    text=f"HiGHS certifies {int(scaling.exact.sum())} of {len(scaling)} cases within its solver limit. {timeouts} stop at the time limit and {nofeasible} return no feasible incumbent. The largest measured construction-plus-solve time is {scaling.runtime_s.max():.2f}~s. This sweep exposes a computation budget that is much larger than in the primary tiny cases, but it does not test a quantum algorithm at those dimensions.\n"
    (dest/'v121_scaling_text.tex').write_text(text)
    (dest/'v121_numbers.tex').write_text(''.join('\\newcommand{\\R'+k+'}{'+str(v)+'}\n' for k,v in vals.items()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=RESULTS);ap.add_argument('--output',type=Path,default=ROOT/'output/v1_2_1');ap.add_argument('--manuscript',type=Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    old.DEST=a.output;s0=pd.read_csv(FROZEN/'all_on/primary_summary.csv');grid=old.tables(s0);old.macros(s0,grid,write_headline=False);old.captions()
    # Retain the existing resource-frontier renderer, then replace the revised panels.
    old.figures(s0,grid);s=tables(a.output,a.results);caps=figures(a.output,a.results,s);macros(a.output,a.results,s)
    p=a.output/'v12_captions.tex';text=p.read_text();import re
    for key,value in caps.items():text=re.sub(r'\\newcommand\{\\V'+key+r'Caption\}\{[^\n]*\}',lambda m:'\\newcommand{\\V'+key+'Caption}{'+value+'}',text)
    p.write_text(text)
    if a.manuscript:
        for p in a.output.iterdir():
            if p.suffix in ['.tex','.pdf']:
                target=a.manuscript/('tables' if p.suffix=='.tex' else 'figures')/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    print('Rendered revision outputs',a.output)
if __name__=='__main__':main()
