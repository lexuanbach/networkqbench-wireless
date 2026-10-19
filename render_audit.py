#!/usr/bin/env python3
"""Generate supplementary audit tables without touching manuscript prose."""
import argparse,json,shutil
from pathlib import Path
import pandas as pd
from artifact_paths import ROOT,output_directory
from render_v122 import number as num,interval as ci,table

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,default=ROOT/'output/analysis/audit');ap.add_argument('--output',type=Path,default=ROOT/'output/current');ap.add_argument('--manuscript',type=Path);a=ap.parse_args();o=output_directory(a.output)
 def read(name):return pd.read_csv(a.input/(name+'.csv'))
 def emit(name,caption,headers,rows):
  table(o,name+'.tex',headers,rows,True)
  p=o/(name+'.tex');s=p.read_text();head=s.split('\n',1)[1].split(r'\endhead')[0];s=s.replace(r'\endhead',r'\endfirsthead'+'\n'+r'\caption[]{'+caption+r' (continued)}\\'+'\n'+head+r'\endhead',1)
  lines=s.splitlines();lines.insert(1,r'\caption{'+caption+r'}\label{sup:'+name.replace('_','-')+r'}\\');p.write_text('\n'.join(lines)+'\n')
 s=read('template_components');names={'preparation_j':'Host preparation','optimization_j':'Optimization','communication_j':'Communication','postprocessing_j':'Selection','offline_j':'Offline','service_j':'Occupied service','network_j':'Applied network','transition_j':'Transitions','total_j':'Total'}
 emit('audit_template_components','Template-minus-numeric ledger. Each round is replayed before averaging within instances. All entries are joules.',['View','Component','Mean [95\\% paired CI]'],[[r.boundary,names[r.component],ci(r.mean,r.low,r.high,4)] for r in s.itertuples()])
 s=read('sensitivity_summary');names={'milp':'HiGHS','heuristic':'Greedy','local_search':'Local search','classical_uniform_onehot':'Uniform one-hot','classical_uniform_full':'Uniform full'}
 rows=[[r.scenario,num(r.useful,2),num(r.q_excess_j),num(r.keep_excess_j),names[r.best_classical],num(r.best_classical_excess_j)] for r in s.itertuples()]
 base=s.iloc[0]
 for r in read('crossed_overheads').itertuples():rows.append([f'{r.qpu_w/1000:g} kW + overheads, $u={r.utilization:g}$',num(100*r.useful,2),num(r.excess_j),num(base.keep_excess_j),names[base.best_classical],num(base.best_classical_excess_j)])
 emit('audit_sensitivity','Complete 64-shot sensitivity summary. Contrasts are policy minus matched HiGHS. Occupied overheads use 200-microsecond repetition delay and 100-ms setup.',['Scenario','Q useful \\%','Q excess J','Keep excess J','Best classical','Excess J'],rows)
 sleep=read('sleep_energy');rows=[[r.scenario,names.get(r.algorithm,{'keep_incumbent':'Keep incumbent','onehot_structured':'Structured QAOA'}.get(r.algorithm,r.algorithm)),ci(r.excess_j,r.ci_low_j,r.ci_high_j),num(100*r.useful,2)] for r in sleep.itertuples()]
 emit('audit_sleep_energy','Sleep-family contrasts at 64 shots and zero wake costs. A dash means that keeping the incumbent has no candidate metric.',['Family','Controller','Excess [95\\% CI] J','Useful \\%'],rows)
 s=read('sleep_consolidation');emit('audit_sleep_consolidation','Empty servers in the exact optimum and consolidation relative to the preceding optimum.',['Family','Instances','Mean empty servers','Consolidation \\%'],[[r.workload.replace('_',' '),r.cases,num(r.empty),num(100*r.consolidation,1)] for r in s.itertuples()])
 d=json.loads((a.input/'diagnostics.json').read_text());k=d['q_minus_keep']
 emit('audit_incumbent','Incumbent comparison and operating-headroom diagnostics in the base all-powered scenario.',['Quantity','Value'],[['Cases with multiple optima',f"{d['cooptimal_instances']}/60"],['Structured QAOA minus keep, mean [95\\% CI] J',ci(k['mean'],k['ci_low'],k['ci_high'])],['Structured QAOA wins against keep',f"{k['wins']}/60"],['Empirical mean crossing horizon, s',num(d['crossing_horizon_s'])],['Crossing residual, J',num(d['crossing_residual_j'],6)]])
 s=read('marginal_all_budgets');s=s[s.algorithm=='quantum_uniform_full']
 emit('audit_marginal_counts','Favorable zero-layer full-space marginal cells at every shot budget. No other quantum configuration has a favorable mean cell.',['Shots','Cells examined','Favorable cells','Minimum contrast J'],[[num(r.shots,0),num(r.cells,0),num(r.favorable,0),num(r.minimum_j,4)] for r in s.itertuples()]+[['All budgets','',str(d['favorable_cells']),num(d['best_grid_j'],4)],['Distinct scenarios','',str(d['favorable_scenarios']),'']])
 labels={'qaoa_transfer':'Standard-X transfer','onehot_transfer':'Dense QAOA','onehot_structured':'Structured QAOA'}
 s=read('noise_resources');emit('audit_noise_resources','All-to-all noise-study circuits at zero noise. Medians use five held-out cases per size. Error is the maximum absolute probability discrepancy after layout inversion.',['$n$','Pipeline','Cases','Median CX','Median depth','Shot $\\mu$s','Max error'],[[r.size,labels[r.algorithm],r.instances,num(r.cx,0),num(r.depth,0),num(r.shot_s*1e6,2),r'\ensuremath{'+f'{r.max_error:.1e}'.replace('e-',r'\times10^{-')+r'}}'] for r in s.itertuples()])
 s=read('admission_targets');emit('audit_admission_targets','Training-target failures and held-out outcomes. Failure means the selected configuration misses the training quality or arrival target. Timely arrival alone does not imply feasible timely output.',['Controller','Instances','Target failures','Q invocations','Useful \\%','Applied \\%'],[[r.policy.replace('_',' '),r.instances,r.target_failures,r.invocations,num(100*r.useful,2),num(100*r.applied,2)] for r in s.itertuples()])
 c=read('carbon');emit('audit_carbon','Illustrative operational carbon associated with a 118.05-J excess over 60 seconds. Grid intensities are scenarios.',['Intensity gCO$_2$e/kWh','Excess J','gCO$_2$e'],[[r.intensity_g_kwh,num(r.excess_j),num(r.carbon_g,5)] for r in c.itertuples()])
 if a.manuscript:
  dest=a.manuscript/'tables';dest.mkdir(parents=True,exist_ok=True)
  for p in o.glob('audit_*.tex'):shutil.copy2(p,dest/p.name)
if __name__=='__main__':main()
