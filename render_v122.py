#!/usr/bin/env python3
"""Portable publication renderer for round three; no historical outputs changed."""
import argparse,json,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_plot_style import save_figure
from matplotlib.ticker import FixedLocator,FuncFormatter
import render_v121 as prior
ROOT=Path(__file__).resolve().parent

def number(v,n=2):
    if not np.isfinite(float(v)):return '--'
    v=0.0 if abs(float(v)) < 0.5*10**(-n) else v
    return r'\ensuremath{'+f'{v:,.{n}f}'.replace(',',r'{,}')+r'}'
def interval(mean,lo,hi,n=2):
    f=lambda x:f'{0.0 if abs(float(x)) < 0.5*10**(-n) else x:,.{n}f}'.replace(',',r'{,}')
    return r'\mbox{$'+f(mean)+r'\;['+f(lo)+','+f(hi)+r']$}'
def table(dest,name,headers,rows,long=False):prior.tex_table(dest,name,headers,rows,long)
def fix_signs(text):
    # Existing math is retained; text-mode numeric intervals become unbreakable math.
    parts=re.split(r'(\$[^$]*\$|\\ensuremath\{[^{}]*\})',text)
    for k in range(0,len(parts),2):
        t=parts[k]
        t=re.sub(r'\[(-?[0-9][0-9,.]*),\s*(-?[0-9][0-9,.]*)\]',lambda m:r'\mbox{$['+m[1]+','+m[2]+']$}',t)
        t=re.sub(r'(?<![\w\-])-(\d[\d,]*(?:\.\d+)?)',lambda m:r'\ensuremath{-'+m[1]+'}',t)
        parts[k]=t
    result=''.join(parts)
    # A comma inside math is punctuation unless braced. Group only thousands,
    # with a decimal part or a clear integer cell, never interval separators.
    result=re.sub(r'(?<![\w{.])\d{1,3}(?:,\d{3})+\.\d+',lambda m:m[0].replace(',',r'{,}'),result)
    result=re.sub(r'(?<=& )([0-9]{4,})(?= ?(?:&|\\\\))',lambda m:format(int(m[1]),',').replace(',',r'{,}'),result)
    result=result.replace('all on','all-powered').replace('Transferred QAOA','Standard-X transfer')
    result=result.replace('One-hot structured p=1','Structured QAOA').replace('One-hot structured p=2','Structured QAOA p=2').replace('One-hot dense p=1','Dense QAOA')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=ROOT/'results/v1_2_2');ap.add_argument('--previous',type=Path,default=ROOT/'results/v1_2_1');ap.add_argument('--output',type=Path,default=ROOT/'output/v1_2_2');ap.add_argument('--manuscript',type=Path);a=ap.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(ROOT/'render_v121.py'),'--results',str(a.previous),'--output',str(o)],check=True)
    d=json.loads((a.results/'diagnostics.json').read_text());s=pd.read_csv(a.previous/'primary_summary.csv');get=prior.get
    rows=[]
    for alg,shots in prior.CHOSEN:
        r=get(s,alg,shots);keep=alg=='keep_incumbent';rows.append([prior.LABELS[alg],shots or '--','--' if keep else f'{r.p_quality:.3f}/{r.p_quality_median:.3f}','--' if keep else f'{100*r.quality:.1f}/{100*r.timely:.1f}','--' if keep else number(100*r.useful,1),number(r.compute_j),interval(r.excess_j,r.ci_low_j,r.ci_high_j),f'{r.wins:.0f}/60'])
    table(o,'v12_principal.tex',['Pipeline','$S$','$p_\\epsilon$ mean/median','Quality/timely \\%','Useful \\%','Compute J','$\\Delta E_{\\rm HiGHS}$ [95\\% CI] J','Wins'],rows)
    t=pd.read_csv(a.results/'training_main.csv');rows=[]
    for n,g in t[t.depth==1].groupby('size'):
        tr=g[g.split=='train'].iloc[0];te=g[g.split=='test'].iloc[0];rows.append([n]+[number(v,3) for v in [tr.uniform,tr.fitted,tr.transferred,te.uniform,te.transferred]])
    table(o,'v122_training_main.tex',['$n$','Train uniform','Train fitted','Train transfer','Test uniform','Test transfer'],rows)
    counts=pd.read_csv(a.results/'below_uniform.csv');rows=[]
    for r in counts.itertuples():rows.append([r.workload.replace('_',' '),r.size,f'{r.expectation_below}/{r.instances}',f'{r.useful_below}/{r.instances}',number(r.uniform,3),number(r.expectation,3),number(r.useful,3)])
    table(o,'v122_below_uniform.tex',['Family','$n$','Expectation below','Useful-mass below','Uniform','Expectation','Useful-mass'],rows,True)
    sensitivity=pd.read_csv(a.results/'sensitivity_summary.csv');names={'milp':'HiGHS','heuristic':'Greedy','local_search':'Local search','classical_uniform_onehot':'Uniform one-hot','classical_uniform_full':'Uniform full'}
    rows=[[r.scenario,number(r.useful,1),number(r.q_excess_j),number(r.keep_excess_j),names[r.best_classical],number(r.best_classical_excess_j)] for r in sensitivity.itertuples()]
    table(o,'v122_sensitivity.tex',['Scenario','Q useful \\%','$\\Delta E_Q$ J','$\\Delta E_{\\rm keep}$ J','Best classical','$\\Delta E_C$ J'],rows)
    demand=pd.read_csv(a.results/'demand_summary.csv');rows=[]
    for alg,shots in [('keep_incumbent',0),('milp',0),('heuristic',0),('classical_uniform_onehot',64),('onehot_structured',64),('onehot_transfer',64)]:
        r=get(demand,alg,shots);rows.append([prior.LABELS[alg],'--' if alg=='keep_incumbent' else number(100*r.useful,1),number(r.compute_j),interval(r.excess_j,r.ci_low_j,r.ci_high_j),interval(r.greedy_excess_j,r.greedy_low_j,r.greedy_high_j)])
    table(o,'v121_demand.tex',['Pipeline','Useful \\%','Compute J','$\\Delta E_{\\rm HiGHS}$ [95\\% CI] J','$\\Delta E_{\\rm greedy}$ [95\\% CI] J'],rows,True)
    # Full primary greedy-referenced intervals remain available in the supplement.
    table(o,'v122_greedy.tex',['Pipeline','$S$','$\\Delta E_{\\rm greedy}$ [95\\% CI] J'],[[prior.LABELS[alg],shots or '--',interval(get(s,alg,shots).greedy_excess_j,get(s,alg,shots).greedy_low_j,get(s,alg,shots).greedy_high_j)] for alg,shots in prior.CHOSEN],True)
    sq=pd.read_csv(a.results/'scaling_quality_summary.csv');rows=[]
    for r in sq.itertuples():rows.append([r.nodes,r.functions,f'{r.gap_count}/10',number(100*r.median_gap,3),number(100*r.max_gap,3),f'{r.greedy_exact_count}/10',number(r.greedy_exact_max_pct,3),f'{r.bound_count}/10',number(r.greedy_bound_max_pct,3)])
    table(o,'v122_scaling_quality.tex',['Servers','Functions','MIP gaps','Median \\%','Max \\%','Certified','Greedy max \\%','Bounds','Greedy bound max \\%'],rows,True)
    tp=pd.read_csv(a.results/'template_summary.csv');rows=[]
    for r in tp[tp.algorithm=='onehot_structured'].itertuples():rows.append([r.mode,r.boundary,number(r.preparation_j,3),number(r.offline_j,3),number(r.service_j,3),interval(r.excess_j,r.ci_low_j,r.ci_high_j)])
    table(o,'v122_template.tex',['Compilation','View','Preparation J','Offline J','Service J','$\\Delta E$ [95\\% CI] J'],rows,True)
    tm=pd.read_csv(a.results/'template_timing.csv').groupby('size').mean(numeric_only=True);rows=[]
    for n,r in tm.iterrows():rows.append([n,number(1000*r.binding_s,3),number(1000*r.numeric_compile_s,3),number(r.template_cx,0),number(r.numeric_cx,1),number(1e6*r.template_shot_s,2),number(1e6*r.numeric_shot_s,2)])
    table(o,'v122_template_resources.tex',['$n$','Bind ms','Numeric compile ms','Template CX','Numeric CX','Template $\\mu$s','Numeric $\\mu$s'],rows,True)
    cp=pd.read_csv(a.results/'common_profile_summary.csv');rows=[]
    for alg in ['onehot_structured','useful_mass_common']:
        r=get(cp,alg,64);rows.append(['Expectation' if alg=='onehot_structured' else 'Useful mass',number(100*r.quality,1),number(r.compute_j),interval(r.excess_j,r.ci_low_j,r.ci_high_j)])
    table(o,'v122_common_profile.tex',['Objective','Quality \\%','Shared compute J','$\\Delta E$ [95\\% CI] J'],rows,True)
    v={};k=d['q_minus_keep'];v.update(KeepDifference=k['mean'],KeepDiffLow=k['ci_low'],KeepDiffHigh=k['ci_high'],KeepWins=k['wins'],Crossing=d['crossing_horizon_s'],GreedyBudget=d['greedy_budget_us']);c=d['common_objective_difference'];v.update(ObjectiveDifference=c['mean'],ObjectiveLow=c['ci_low'],ObjectiveHigh=c['ci_high'],CommonAligned=get(cp,'useful_mass_common',64).excess_j)
    v.update(DemandGreedy=get(demand,'heuristic').excess_j,DemandGreedyLow=get(demand,'heuristic').ci_low_j,DemandGreedyHigh=get(demand,'heuristic').ci_high_j,DemandKeep=get(demand,'keep_incumbent').excess_j,DemandKeepLow=get(demand,'keep_incumbent').ci_low_j,DemandKeepHigh=get(demand,'keep_incumbent').ci_high_j)
    for label,key in [('Occupied overheads, u=0.5','HalfOverhead'),('10 kW','TenKW')]:
        r=sensitivity[sensitivity.scenario==label].iloc[0];v[key+'Compute']=r.q_compute_j;v[key+'Excess']=r.q_excess_j
    for mode in ['template','numeric']:
        for boundary in ['allocated','marginal']:
            r=tp[(tp['mode']==mode)&(tp.boundary==boundary)&(tp.algorithm=='onehot_structured')].iloc[0];v[mode.title()+boundary.title()]=r.excess_j
    z=pd.read_csv(a.results/'zero_transpilation_summary.csv');v['ZeroCompile']=get(z[z.boundary=='allocated'],'onehot_structured',64).excess_j;v['ZeroCompileMarginal']=get(z[z.boundary=='marginal'],'onehot_structured',64).excess_j
    (o/'v122_numbers.tex').write_text(''.join('\\newcommand{\\T'+key+'}{'+number(value,0 if key=='KeepWins' else 2)+'}\n' for key,value in v.items()))
    figures(o,a.previous,d,s)
    # Apply the current cartoon and its audited caption after legacy figure passes.
    subprocess.run([sys.executable, str(ROOT/'render_fig1_cartoon.py'),
                    '--output', str(o)], check=True)
    for p in o.glob('*.tex'):p.write_text(fix_signs(p.read_text()))
    if a.manuscript:
        for p in o.iterdir():
            if p.suffix in ['.tex','.pdf','.png']:shutil.copy2(p,a.manuscript/('tables' if p.suffix=='.tex' else 'figures')/p.name)
    print('Rendered',o)

def figures(o,previous,d,s):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):save_figure(fig,o/name,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,2.6));fig.subplots_adjust(top=.68,bottom=.19,left=.10,right=.98)
    fig.text(.02,.91,'Current demand + incumbent  →  Candidate sampling  →  Deadline / benefit gate  →  Applied placement',fontsize=8,weight='bold')
    fig.text(.02,.79,'Preparation + fitting + service + selection                         Arrival + transition + network operation',fontsize=7.5)
    algs=['milp','heuristic','classical_uniform_onehot','quantum_uniform_onehot','onehot_structured','onehot_transfer'];labels=['HiGHS','Greedy','CPU uniform','Q uniform','Structured QAOA','Dense QAOA'];rs=[prior.get(s,alg,0 if alg in ['milp','heuristic'] else 64) for alg in algs]
    bars=ax.bar(labels,[r.compute_j for r in rs],color=[prior.COLORS[alg] for alg in algs]);ax.set(yscale='log',ylabel='Computation energy (J)');ax.set_ylim(.004,2000);ax.bar_label(bars,labels=[f'{r.compute_j:.2f}' for r in rs],fontsize=7,padding=3);ax.tick_params(axis='x',labelsize=7);save(fig,'fig1_cartoon.pdf')
    p=o/'fig1_cartoon.tex';t=p.read_text();caption=r'Accounting workflow and mean computation charges over 60 primary held-out cases (64 shots for samplers). Candidate computation is charged even when the gate retains the incumbent. Applied-network and transition terms enter the total-energy comparison in Table~\ref{tab:principal}; all joules are modeled. Sampling results average 30 batches per instance.';t=re.sub(r'\\newcommand\{\\FigOneCartoonCaption\}\{[^\n]*\}',lambda m:'\\newcommand{\\FigOneCartoonCaption}{'+caption+'}',t);p.write_text(t)
    fig=plt.figure(figsize=(7.4,3.35),layout='constrained');grid=fig.add_gridspec(1,2,width_ratios=[1,1.06]);right=grid[1].subgridspec(3,1,height_ratios=[1,1,.23]);ax=[fig.add_subplot(grid[0]),fig.add_subplot(right[0])];detail=fig.add_subplot(right[1]);key=fig.add_subplot(right[2]);h=pd.read_csv(previous/'horizon_intervals.csv');lower=0
    for alg in ['keep_incumbent','classical_uniform_onehot','quantum_uniform_onehot','onehot_structured','onehot_transfer']:
        g=h[(h.algorithm==alg)&(h.shots==(0 if alg=='keep_incumbent' else 64))].sort_values('horizon');ax[0].plot(g.horizon,g.excess_j,'o-',ms=3,label=prior.LABELS[alg],color=prior.COLORS[alg]);ax[0].fill_between(g.horizon,g.ci_low_j,g.ci_high_j,alpha=.1,color=prior.COLORS[alg]);lower=min(lower,g.ci_low_j.min())
    ax[0].set(xscale='log',xlabel='Constant-demand horizon (s)',ylabel='Extra total energy vs. HiGHS (J)',title='(a) Horizon and incumbent');ax[0].set_yscale('symlog',linthresh=10);ax[0].set_ylim(min(-2,lower*1.1),20000);ax[0].yaxis.set_major_locator(FixedLocator([0,10,100,1000,10000]));ax[0].yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'));ax[0].axhline(0,color='#888',lw=.7);ax[0].axvline(d['crossing_horizon_s'],ls=':',color='#444',lw=.8);ax[0].annotate(f"Empirical mean\n{d['crossing_horizon_s']:.2f} s",(d['crossing_horizon_s'],115),xytext=(d['crossing_horizon_s']/1.35,260),ha='right',va='center',arrowprops={'arrowstyle':'->','lw':.6,'shrinkA':1,'shrinkB':1},fontsize=6.5,bbox={'boxstyle':'square,pad=0.15','fc':'white','ec':'none'});ax[0].legend(fontsize=5.6,loc='upper left')
    dec=pd.read_csv(ROOT/'results/v1_2/all_on/decomposition.csv')
    algs=['quantum_uniform_onehot','onehot_structured','onehot_transfer']
    names=['Q uniform','Structured QAOA','Dense QAOA']; y=np.arange(3)
    values=dec[dec.shots==64].groupby('algorithm').mean(numeric_only=True).loc[algs]
    computation=values.compute_j.to_numpy()
    ax[1].barh(y,computation,height=.48,color='#0072b2',zorder=3)
    for j,value in enumerate(computation):
        ax[1].text(value+8,j,f'{value:+.2f}',va='center',fontsize=6.5)
    ax[1].set(yticks=y,yticklabels=names,xlim=(0,620),xticks=[0,200,400,600],
              xlabel='Computation difference (J)',title='(b) Energy component differences')
    ax[1].invert_yaxis()
    positive=np.zeros(3); negative=np.zeros(3)
    for col,label,color in [('delay_loss_j','Arrival','#e69f00'),
                            ('quality_loss_j','Placement quality','#009e73'),
                            ('fallback_loss_j','Rejection','#cc79a7'),
                            ('transition_j','Transitions','#777777')]:
        v=values[col].to_numpy()
        detail.barh(y,v,left=np.where(v>=0,positive,negative),height=.48,
                    label=label,color=color,zorder=3)
        positive+=np.maximum(v,0);negative+=np.minimum(v,0)
    for j,(total,transition) in enumerate(zip(positive+negative,values.transition_j)):
        detail.text(positive[j]+.18,j,f'{total:+.2f}',va='center',fontsize=6.5)
        detail.text(-.30,j,f'{transition:+.2f}',ha='right',va='center',fontsize=6,color='#444444')
    detail.set(yticks=y,yticklabels=names,xlim=(-1.8,11.6),xticks=[0,5,10],
               xlabel='Network + transition difference (J)')
    detail.invert_yaxis();detail.axvline(0,color='#666666',lw=.7,zorder=2)
    for panel in [ax[1],detail]:
        panel.set_axisbelow(True);panel.grid(axis='x',color='#e4e4e4',lw=.5)
        panel.tick_params(axis='both',labelsize=6.8,length=2)
        panel.xaxis.label.set_fontsize(7)
        panel.spines['left'].set_visible(False);panel.spines['bottom'].set_color('#bbbbbb')
        panel.tick_params(axis='y',length=0)
        panel.margins(y=.24)
    key.axis('off')
    key.legend(*detail.get_legend_handles_labels(),loc='center',ncol=2,
               frameon=False,fontsize=6,handlelength=1.2,columnspacing=1.5)
    save(fig,'net_energy.pdf')
    p=o/'v12_captions.tex';t=p.read_text();caption=r'(a) Policy energy minus HiGHS across constant-demand horizons, with paired 95\% instance intervals. The line marks the empirical mean structured-QAOA/keep crossing from event replay. (b) Quantum-minus-HiGHS components at 64 shots, in joules: computation (top), applied network and transitions (bottom). Negative bars are transitions. Both panels use Primary timing.';t=re.sub(r'\\newcommand\{\\VNetCaption\}\{[^\n]*\}',lambda m:'\\newcommand{\\VNetCaption}{'+caption+'}',t);p.write_text(t)
if __name__=='__main__':main()
