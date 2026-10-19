#!/usr/bin/env python3
"""Render the local fourth-round evidence and caption every supplementary table."""
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from render_v122 import number as num, interval as ci, table, fix_signs
from artifact_paths import portable, output_directory
ROOT=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,default=ROOT/'results/round4_20261019');ap.add_argument('--third',type=Path,default=ROOT/'results/v1_2_2');ap.add_argument('--output',type=Path,default=ROOT/'output/round4');ap.add_argument('--manuscript',type=Path);ap.add_argument('--campaign',type=Path,default=ROOT/'results/round4_20261019');ap.add_argument('--skip-legacy',action='store_true');a=ap.parse_args();o=output_directory(a.output);m=a.manuscript
 if not a.skip_legacy:subprocess.run([sys.executable,str(ROOT/'render_v122.py'),'--output',str(o)],check=True)
 read=lambda n:pd.read_csv(a.input/(n+'.csv'))
 t=pd.read_csv(a.third/'training_main.csv');counts=pd.read_csv(a.third/'below_uniform.csv');rows=[]
 for n,g in t[t.depth==1].groupby('size'):
  tr=g[g.split=='train'].iloc[0];te=g[g.split=='test'].iloc[0];c=counts[(counts.workload=='all_on')&(counts['size']==n)].iloc[0]
  rows.append([n]+[num(v,3) for v in [tr.uniform,tr.fitted,tr.transferred,te.uniform,te.transferred]]+[f'{c.expectation_below}/15'])
 table(o,'round4_training_main.tex',['$n$','Train uniform','Train fitted','Train transfer','Test uniform','Test transfer','Test below'],rows)
 s=read('absolute_scaling_summary');rows=[]
 for r in s.itertuples():rows.append([r.nodes,r.functions,f'{r.certified}/10',num(r.exact_max_w),f'{r.bounds}/10',num(r.bound_max_w),num(r.idle_w,0),num(r.static_60s_bound_j)])
 table(o,'round4_absolute.tex',['Servers','Functions','Certified','Exact max W','Bounds','Bound max W','Idle W','Static upper J'],rows,True)
 vals=dict(AppliedPct=100*pd.read_csv(ROOT/'results/v1_2/all_on/primary_instances.csv').query("algorithm=='onehot_structured' and shots==64 and seed>=15").accepted.mean(),ExactFifty=s[s.nodes==50].exact_max_w.max(),ExactHundred=s[s.nodes==100].exact_max_w.max(),BoundFour=s[(s.nodes==100)&(s.functions==4)].bound_max_w.iloc[0],BoundEight=s[(s.nodes==100)&(s.functions==8)].bound_max_w.iloc[0],BoundCount=s[s.nodes==100].bounds.sum(),StaticFour=s[(s.nodes==100)&(s.functions==4)].static_60s_bound_j.iloc[0],StaticEight=s[(s.nodes==100)&(s.functions==8)].static_60s_bound_j.iloc[0])
 q=read('template_paired_summary');rows=[]
 mr=q[(q.boundary=='marginal')&(q.metric=='total_j')].iloc[0];vals.update(TemplateMarginalDifference=mr['mean'],TemplateMarginalLow=mr.low,TemplateMarginalHigh=mr.high)
 for r in q[q.metric.isin(['preparation_j','compute_j','total_j'])].itertuples():
  rows.append([r.boundary,r.metric.replace('_j',' (J)').replace('_',' '),ci(r.mean,r.low,r.high,3),ci(0,r.round_min,r.round_max,3).replace('0.000\\;','')])
  if r.metric=='preparation_j': rows.append([r.boundary,'preparation (ms)',ci(20*r.mean,20*r.low,20*r.high,3),ci(0,20*r.round_min,20*r.round_max,3).replace('0.000\\;','')])
 table(o,'round4_template.tex',['View','Difference','Paired mean [95\\% CI]','Round-mean range'],rows,True)
 r=q[(q.boundary=='allocated')&(q.metric=='total_j')].iloc[0];vals.update(TemplateDifference=r['mean'],TemplateLow=r.low,TemplateHigh=r.high)
 prob=read('transfer_probability_summary');rows=[]
 methods=['circular_expectation','direct_expectation','circular_useful_mass','direct_useful_mass'];short={'circular_expectation':'Circular E','direct_expectation':'Direct E','circular_useful_mass':'Circular U','direct_useful_mass':'Direct U'}
 for (family,n),g in prob.groupby(['workload','size']):
  g=g.set_index('method');rows.append([family.replace('_',' '),n,num(g.uniform.iloc[0],3)]+[num(g.loc[k,'p_quality'],3)+f' ({int(g.loc[k,"below"])}/15)' for k in methods])
 table(o,'round4_probability.tex',['Family','$n$','Uniform']+[short[k]+' (below)' for k in methods],rows,True)
 p=read('transfer_summary');paired=read('transfer_paired_summary');rows=[]
 # Main supplementary view is allocated at 64; complete CSV covers marginal and all budgets.
 for (family,profile),g in p[(p.shots==64)&(p.boundary=='allocated')].groupby(['workload','profile']):
  for k in methods:
   r=g[g.algorithm==k].iloc[0];rows.append([family.replace('_',' '),profile,short[k],num(100*r.useful,1),num(100*r.accepted,1),num(r.offline_j,3),ci(r['mean'],r.low,r.high)])
 table(o,'round4_transfer.tex',['Family','Profile','Rule','Useful \\%','Applied \\%','Fit J','$\\Delta E_{\\rm HiGHS}$ [95\\% CI] J'],rows,True)
 rows=[]
 for path in sorted((a.campaign/'transfer').glob('*/fits.json')):
  for r in json.loads(path.read_text()):rows.append([r['workload'].replace('_',' '),r['size'],'E' if r['objective']=='expectation' else 'U',r['instance_evaluations'],r['historical_instance_budget'],r['iterations'],num(r['offline_s'],3),'converged' if r['stopping_reason']=='Optimization terminated successfully.' else 'budget' if 'evaluations' in r['stopping_reason'] else 'iteration cap'])
 table(o,'round4_fits.tex',['Family','$n$','Objective','Evaluations','Cap','Iterations','Fit s','Stop'],rows,True)
 res=read('transfer_resources');rows=[]
 for (family,n,k),g in res.groupby(['workload','size','method']):rows.append([family.replace('_',' '),n,short[k],num(g.cx.median(),0),num(g.depth.median(),0),num(1e6*g.shot_s.median(),2),num(1e3*(g.preparation_s+g.circuit_build_s+g.compile_s).mean(),2)])
 table(o,'round4_resources.tex',['Family','$n$','Rule','Median CX','Depth','Shot $\\mu$s','Mean prep ms'],rows,True)
 refs=read('reference_summary');rows=[]
 for r in refs.itertuples():rows.append([r.nodes,r.functions,f'{r.feasible}/10',f'{r.certified}/10',f'{r.bounds}/10',num(100*r.max_gap,3),num(r.median_s,3)])
 table(o,'round4_references.tex',['Servers','Functions','Feasible','Certified','Bounds','Max gap \\%','Median s'],rows,True)
 d=read('deadline_summary');rows=[];energy=[]
 for r in d.itertuples():
  name='HiGHS' if r.algorithm=='milp' else 'Greedy';prefix=[r.nodes,r.functions,num(r.deadline_s,1),name]
  rows.append(prefix+[num(100*r.utilization,1),num(r.median_build_s,3),num(r.median_solve_s,3),num(r.median_total_s,3),num(r.max_total_s,3),f'{r.feasible_timely}/10',f'{r.accepted}/10'])
  energy.append(prefix+[num(r.mean_compute_j),ci(r.minus_keep_j,r.keep_low,r.keep_high),f'{r.exact_count}/10',num(r.exact_max_w),f'{r.bound_count}/10',num(r.bound_max_w)])
 table(o,'round4_deadline.tex',['$N$','$F$','$D$ s','Solver','Util. \\%','Build s','Solve s','Total s','Max s','Feasible timely','Applied cases'],rows,True)
 table(o,'round4_deadline_energy.tex',['$N$','$F$','$D$ s','Solver','Compute J','$\\Delta E_{\\rm keep}$ [95\\% CI] J','Exact','Max W','Bounds','Max W'],energy,True)
 rows=[[r.nodes,r.functions,num(r.deadline_s,1),ci(r.minus_greedy_j,r.greedy_low,r.greedy_high)] for r in d[d.algorithm=='milp'].itertuples()]
 table(o,'round4_deadline_pairs.tex',['$N$','$F$','$D$ s','$\\Delta E_{\\rm HiGHS-greedy}$ [95\\% CI] J'],rows,True)
 # Compact main-text summary: no pooling across deadline settings for inference.
 rows=[]
 for deadline,g in d.groupby('deadline_s'):
  for alg in ['milp','heuristic']:
   h=g[g.algorithm==alg];rows.append([num(deadline,1),'HiGHS' if alg=='milp' else 'Greedy',f'{int(h.feasible_timely.sum())}/100',f'{int(h.accepted.sum())}/100',num(h.max_total_s.max(),3)])
 table(o,'round4_deadline_main.tex',['Deadline s','Solver','Feasible timely','Applied cases','Max time s'],rows)
 # Paired candidate-only direct-minus-circular comparison.
 rows=[]
 for r in paired[(paired.profile=='common')&(paired.boundary=='allocated')&(paired.shots==64)].itertuples():rows.append([r.workload.replace('_',' '),'E' if r.objective=='expectation' else 'U',ci(r.mean,r.low,r.high),num(100*r.useful_difference,2),num(100*r.accepted_difference,2)])
 table(o,'round4_pairs.tex',['Family','Objective','$\\Delta E_{\\rm direct-circular}$ [95\\% CI] J','$\\Delta$ useful pp','$\\Delta$ applied pp'],rows)

 ap=p[(p.workload=='all_on')&(p.profile=='common')&(p.boundary=='allocated')&(p.shots==64)].set_index('algorithm')
 uq=prob[(prob.workload=='all_on')&(prob.method=='direct_useful_mass')]
 direct=ap.loc['direct_useful_mass'];circ=ap.loc['circular_useful_mass']
 text=(r'Direct family useful-mass fitting raises observed all-powered mean mass from '+num(circ.p_quality,3)+r' to '+num(direct.p_quality,3)+r', with '+str(int(uq.below.sum()))+r'/60 instances below uniform, compared with 26/60 for circular aggregation. Yet its 64-shot useful-candidate rate falls from '+num(100*circ.useful,1)+r'\% to '+num(100*direct.useful,1)+r'\%, illustrating the nonlinear relation between mass and batch success. Under common Primary charges, its excess over HiGHS remains '+ci(direct['mean'],direct.low,direct.high)+r'~J. Table~\ref{tab:family-rule} gives paired rule contrasts. No interval establishes a strictly negative interval at this budget. These findings concern the tested initialization, budget, and aggregation rules. Full-pipeline fitting charges, all four power families, and all budgets are retained in the supplement and artifact.'+'\n')
 (o/'round4_direct_text.tex').write_text(text)
 (o/'round4_numbers.tex').write_text(''.join('\\newcommand{\\RF'+k+'}{'+num(v,0 if k=='BoundCount' else 2)+'}\n' for k,v in vals.items()))
 if not a.skip_legacy:
  for script in ['analyze_results','analyze_revision_experiments','analyze_clear_accept','analyze_review_experiments','analyze_tnsm_revision']:
   subprocess.run([sys.executable,str(ROOT/(script+'.py')),'--output',str(o/'legacy')],check=True,stdout=subprocess.DEVNULL)
  for p in (o/'legacy/tables').glob('*'):
   if p.suffix in ['.tex','.txt']:shutil.copy2(p,o/(p.stem+'.tex'))
 caption_supplement(o)
 if m:
  (m/'tables').mkdir(parents=True,exist_ok=True)
  for p in o.glob('*.tex'):
   if p.stem not in {'solver_summary','clearaccept_numbers','review_numbers','tnsm_numbers'}:shutil.copy2(p,m/'tables'/p.name)
  shutil.copytree(o/'round4_supp',m/'tables/round4_supp',dirs_exist_ok=True)
  (m/'figures').mkdir(exist_ok=True)
  for p in o.glob('*.pdf'):shutil.copy2(p,m/'figures'/p.name)
 (o/'render_manifest.json').write_text(json.dumps(dict(input=portable(a.input),font='article Times through LaTeX; original cartoon lettering retained',supplement_tables='captioned copies with stable labels',figure1='layout and computation/outcome panels preserved'),indent=2)+'\n')

def caption_supplement(output):
 """Captioned copies keep historical generated inputs unchanged and labels stable."""
 captions={
 'clearaccept_baselines':'Dimensionless solver comparisons', 'clearaccept_hardness':'Larger dimensionless workload cases',
 'clearaccept_shot_replicates':'Repeated dimensionless sampling budgets', 'fable_depth_sensitivity':'Individually fitted depth sensitivity',
 'fable_transfer_aggregation':'Dimensionless transfer aggregation rules', 'fable_penalty_sensitivity':'Penalty-weight sensitivity',
 'fable_staleness_sensitivity':'Staleness sensitivity', 'revision_trace_summary':'Abilene trace replay',
 'clearaccept_netdata_5g':'5G trace replay', 'review_closed_loop':'Closed-loop trajectory comparisons',
 'fable_timing_frontier':'Queue and execution timing frontier', 'fable_mip_deadline':'Budget-matched dimensionless MILP controls',
 'review_constraint_aware':'Constraint-preserving mixer comparisons', 'review_full_resources':'Complete dimensionless objective circuits',
 'clearaccept_compiled_resources':'Compiled supporting circuits', 'revision_resource_ledger':'Supporting quantum resource ledger',
 'review_backend_emulation':'Calibration-snapshot emulations',
 'v12_budgets':'Base placement results at all shot budgets. A dash in J/useful means an infinite ratio', 'v12_per_size':'Primary placement contrasts by size',
 'v12_uniform':'Uniform useful mass and co-optimal multiplicity', 'v12_shots':'Per-instance shot requirements',
 'v12_marginal':'Marginal accounting grid', 'v12_reuse':'Offline reuse thresholds. A dash means no finite positive crossing', 'v12_speed':'Service compression sensitivity',
 'v12_resources':'Complete primary circuit resources', 'v12_qubo':'Quadratic encoding audit', 'v12_angles':'Transferred angles and dispersion',
 'v12_noise':'Matched-target CX-depolarizing study', 'v12_incumbents':'Preceding-demand incumbents', 'v12_utilization':'Physical utilization quantiles',
 'v12_policies':'Admission and fallback policies', 'v12_transitions':'Migration and wake sensitivities', 'v12_stress_decomposition':'Concentrated-incumbent energy decomposition',
 'v12_sleep':'Sleep-family comparisons', 'v12_overheads':'Occupied service overheads',
 'v121_ledger':'Structured 64-shot marginal component ledger', 'v121_nominal':'Nominal queue replay', 'v121_training':'Fitting and transfer on separate data splits',
 'v121_aligned':'Optimum-informed useful-mass fitting', 'v121_demand':'Larger preceding-demand changes', 'v121_larger':'Larger physical placement cases',
 'v121_scaling':'Original sparse classical scaling', 'v121_timing':'Repeated host timing',
 'v122_greedy':'Paired primary contrasts against greedy', 'v122_common_profile':'Common-profile training-objective replay', 'v122_below_uniform':'Instances below uniform useful mass',
 'v122_scaling_quality':'Original exact and bound-based scaling gaps', 'v122_template_resources':'Symbolic-template resources', 'v122_template':'Template and numeric energy',
 'round4_deadline_main':'Deadline outcomes over 100 unique cases per deadline. Applied counts are instances',
 'round4_absolute':'Absolute gaps in the original scaling study', 'round4_template':'Paired template-minus-numeric differences',
 'round4_probability':'Held-out mass: circular and direct family rules', 'round4_transfer':'Family-rule pipelines at 64 shots (allocated)',
 'round4_fits':'Direct-family fitting evaluations and costs. E: expectation, U: useful mass', 'round4_resources':'New compiled resources and preparation measurements',
 'round4_references':'Offline quality references for all 100 unique sparse instances', 'round4_deadline':'Deadline-limited timing and deployment counts',
 'round4_deadline_energy':'Deadline-limited energy and computed-candidate gaps. A dash means no certified reference or available bound', 'round4_deadline_pairs':'Paired deadline-limited HiGHS-minus-greedy contrasts', 'round4_pairs':'Common-profile paired direct-minus-circular energy at 64 shots',
 }
 dest=output/'round4_supp';dest.mkdir(parents=True,exist_ok=True)
 for name,title in captions.items():
  src=output/(name+'.tex')
  if not src.exists():continue
  content=fix_signs(src.read_text())
  label='sup:'+name.replace('_','-')
  content=content.replace(r' & Quality \% & ',r' & Candidate \% & ')
  if name in {'v12_angles','v121_training','review_full_resources','fable_depth_sensitivity'}:
   content=content.replace(r'$p$ & ',r'Depth & ')
  # Use one caption style, including compact tables formerly wrapped in captionof.
  content=content.replace(r'\begin{tabular}',r'\begin{longtable}').replace(r'\end{tabular}',r'\end{longtable}')
  content=re.sub(r'\\begin\{tabular\*\}\{[^}]*\}\{',lambda _:r'\begin{longtable}{',content).replace(r'\end{tabular*}',r'\end{longtable}')
  if r'\endhead' not in content:
   content=content.replace(r'\midrule',r'\midrule'+'\n'+r'\endhead',1)
  header=content.split('\n',1)[1].split(r'\endhead')[0]
  content=content.replace(r'\endhead',r'\endfirsthead'+'\n'+r'\caption[]{'+title+r' (continued)}\\'+'\n'+header+r'\endhead',1)
  content=re.sub(r'(\\begin\{longtable\}\{[^\n]*\})',lambda x:x[0]+'\n'+r'\caption{'+title+'}'+r'\label{'+label+r'}\\',content,count=1)
  content=content.replace('useful decisions','useful candidates').replace('Useful decisions','Useful candidates')
  # Keep the final three data rows together across a page break.
  lines=content.splitlines();end=next((i for i,l in enumerate(lines) if r'\bottomrule' in l),len(lines))
  rows=[i for i in range(end) if ' & ' in lines[i] and lines[i].rstrip().endswith(r'\\')]
  for i in rows[-3:-1]:lines[i]=lines[i].rstrip()+'*'
  (dest/(name+'.tex')).write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
