#!/usr/bin/env python3
"""Render the condensed local manuscript and final-review diagnostics."""
import argparse,subprocess,sys,json,shutil,re
from pathlib import Path
import pandas as pd
from render_v122 import number as num,interval as ci,table
from artifact_paths import output_directory
from render_v122 import fix_signs
ROOT=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=ROOT/'results/round5_20261019');ap.add_argument('--output',type=Path,default=ROOT/'output/round5');ap.add_argument('--manuscript',type=Path);ap.add_argument('--fourth',type=Path,default=ROOT/'results/round4_20261019');ap.add_argument('--skip-legacy',action='store_true');a=ap.parse_args();o=output_directory(a.output);m=a.manuscript
 if not a.skip_legacy:subprocess.run([sys.executable,str(ROOT/'render_round4.py'),'--input',str(a.fourth),'--output',str(o)],check=True)
 s=pd.read_csv(ROOT/'results/v1_2_2/sensitivity_summary.csv');names={'milp':'HiGHS','heuristic':'Greedy','local_search':'Local search','classical_uniform_onehot':'Uniform one-hot','classical_uniform_full':'Uniform full'}
 rows=[[r.scenario,'P/D' if r.scenario=='Demand change' else 'P',num(r.useful,1),num(r.q_excess_j),num(r.keep_excess_j),names[r.best_classical],num(r.best_classical_excess_j)] for r in s.itertuples()]
 cross=pd.read_csv(a.input/'crossed_overheads.csv')
 for r in cross[cross.qpu_w==10000].itertuples():rows.append([f'10 kW + overheads, $u={r.utilization:g}$','P',num(100*r.useful,1),num(r.excess_j),num(s[s.scenario=='Marginal 0.001'].keep_excess_j.iloc[0]),'Greedy',num(s[s.scenario=='Marginal 0.001'].best_classical_excess_j.iloc[0])])
 table(o,'round5_sensitivity.tex',['Scenario','Campaign','Q useful \\%','$\\Delta E_Q$ J','$\\Delta E_{\\rm keep}$ J','Best classical','$\\Delta E_C$ J'],rows)
 t=pd.read_csv(ROOT/'results/v1_2_2/template_summary.csv');t=t[(t.algorithm=='onehot_structured')&(t.boundary=='marginal')].set_index('mode');z=pd.read_csv(ROOT/'results/v1_2_2/zero_transpilation_summary.csv');z=z[(z.algorithm=='onehot_structured')&(z.boundary=='marginal')&(z.shots==64)].iloc[0]
 table(o,'round5_marginal_map.tex',['Implementation','Timing','$\\Delta E$ J'],[['Numeric','Primary',num(s[s.scenario=='Marginal 0.001'].q_excess_j.iloc[0])],['Numeric','Template',num(t.loc['numeric','excess_j'])],['Symbolic binding','Template',num(t.loc['template','excess_j'])],['Zero compile (counterfactual)','Primary',num(z.excess_j)]])
 tr=pd.read_csv(ROOT/'results/v1_2_1/training_diagnosis.csv');tr=tr[(tr.workload=='all_on')&(tr.split=='test')&(tr.depth==1)];med=pd.read_csv(a.input/'test_medians.csv');rows=[]
 for r in med.itertuples():
  g=tr[tr['size']==r.size];rows.append([r.size,num(r.uniform_mean,3),num(r.transfer_mean,3),num(r.uniform_median,3),num(r.transfer_median,3),num((g.transferred-g.uniform).median(),3),f'{int((g.transferred<g.uniform-1e-12).sum())}/15'])
 table(o,'round5_training.tex',['$n$','Uniform mean','Transfer mean','Uniform median','Transfer median','Paired median $\\Delta$','Below'],rows,True)
 l=pd.read_csv(a.input/'larger_requirements_summary.csv');r=l[l['size']==16].iloc[0];vals=dict(QShots=r.q_median,UShots=r.u_median,QShotsMax=r.q_max,UShotsMax=r.u_max,ServiceMedian=r.service_median_j)
 table(o,'round5_larger.tex',['$n$','Q shots med/max','Uniform med/max','Q service median J','Q service max J'],[[int(r['size']),f'{r.q_median:.0f}/{r.q_max:.0f}',f'{r.u_median:.0f}/{r.u_max:.0f}',num(r.service_median_j),num(r.service_max_j)] for _,r in l.iterrows()],True)
 ordering=pd.read_csv(a.input/'ordering_instances.csv');rows=[]
 for mode,g in ordering.groupby('mode'):rows.append([mode,num(g.p_quality.mean(),3),num(g.uniform.mean(),3),f'{int((g.p_quality<g.uniform).sum())}/15'])
 table(o,'round5_ordering.tex',['Mixer order','Transfer mass','Uniform mass','Below uniform'],rows,True)
 for name,caption,label in [('round5_training','Held-out useful mass and paired changes under original circular transfer','sup:test-medians'),('round5_larger','Per-instance shot requirements for a 90-percent useful-state hit target. Service-only energy excludes host costs and deadline gating','sup:larger-shots'),('round5_ordering','Six-variable order sensitivity with matched training-only fitting budgets','sup:ordering')]:
  p=o/(name+'.tex');v=p.read_text();head=v.split('\n',1)[1].split(r'\endhead')[0];v=v.replace(r'\endhead',r'\endfirsthead'+'\n'+r'\caption[]{'+caption+r' (continued)}\\'+'\n'+head+r'\endhead',1);v=re.sub(r'(\\begin\{longtable\}\{[^\n]*\})',lambda x:x[0]+'\n'+r'\caption{'+caption+'}'+r'\label{'+label+r'}\\',v,count=1);p.write_text(v)
 diag=json.loads((a.input/'diagnostics.json').read_text());gs=ordering.groupby('mode').p_quality.mean()
 text=r'''The bounded ordering diagnostic reverses the ring-edge sequence within each
function, retaining the same normalized expectation objective, seven-by-seven
grid, circular aggregation, and at most 40 Nelder--Mead iterations. Each
training fit is capped at that instance's historical evaluation count.
Both orders use the same training and held-out split. Forward transfer
reproduces the stored baseline mean. Reversal changes held-out mean mass
from '''+num(gs['forward'],3)+' to '+num(gs['reverse'],3)+r''', still below uniform's 0.341.
Both ordered implementations remain below uniform.
The diagnostic simulates probabilities only and makes no new compiled-energy
claim. Fitting times and budgets are recorded separately.

The 1.268-J largest Primary HiGHS charge belongs to nine variables, seed 29.
Repeated-campaign round means range from '''+num(diag['milp_round_mean_min_j'],3)+' to '+num(diag['milp_round_mean_max_j'],3)+r'''~J.
The 0.96-J Demand mean lies outside that observed range. The range is
not a confidence bound for every future host campaign.
Server~0 carries the largest demand (ties included) in '''+str(diag['server0_max_demand'])+r'''/60 returned Primary MILP optima, and the largest utilization in '''+str(diag['server0_max_utilization'])+r'''/60. Enlarging its capacity therefore does not force it to be the most-loaded
server in every optimum. Co-optimal placements can have different assignments.
'''
 (o/'round5_diagnostics_text.tex').write_text(text)
 (o/'round5_numbers.tex').write_text(''.join('\\newcommand{\\F'+k+'}{'+num(v,2 if k=='ServiceMedian' else 0)+'}\n' for k,v in vals.items()))
 # Remove the standard transferred-X row from the concise main comparison only.
 p=o/'v12_principal.tex';v=p.read_text();v='\n'.join(line for line in v.splitlines() if not line.startswith(('Transferred QAOA &','Standard-X transfer &')))+'\n';p.write_text(v)
 # Correct the wording at the generated source, including its captioned supplement copy.
 for base in [o,o/'round4_supp']:
  for p in base.glob('*timing_text.tex'):p.write_text(p.read_text().replace('10 of ten timing rounds','all ten timing rounds'))
 for p in o.glob('*.tex'):p.write_text(fix_signs(p.read_text()))
 if m:
  (m/'tables').mkdir(parents=True,exist_ok=True)
  for p in o.glob('*.tex'):
   if p.stem not in {'solver_summary','clearaccept_numbers','review_numbers','tnsm_numbers'}:shutil.copy2(p,m/'tables'/p.name)
  if (o/'round4_supp').exists():shutil.copytree(o/'round4_supp',m/'tables/round4_supp',dirs_exist_ok=True)
  (m/'figures').mkdir(exist_ok=True)
  for p in o.glob('*.pdf'):shutil.copy2(p,m/'figures'/p.name)
if __name__=='__main__':main()
