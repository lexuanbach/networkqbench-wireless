#!/usr/bin/env python3
"""Check per-instance thresholds, mixer-order simulation and new cross-sensitivities."""
from artifact_paths import validation_path
import json
from pathlib import Path
import numpy as np,pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import DiagonalGate
import energy_v12 as E
from networkqbench import _normalized_cost
from analyze_round5 import ordering_probs
from analyze_v12 import raw_evaluate,DEFAULT
from dataclasses import replace
ROOT=Path(__file__).resolve().parent;D=ROOT/'results/round5_20261019'

def main():
 for seed in [0,14,15,29]:
  i=E.workload(6,seed);theta=[.73,.41];assert np.allclose(ordering_probs(i,theta),E.onehot_probs(i,theta),atol=1e-12)
  _,states=E.assignment_basis(i);q=QuantumCircuit(6);v=np.zeros(64,complex);v[states]=1/np.sqrt(len(states));q.initialize(v);q.append(DiagonalGate(np.exp(-1j*theta[0]*_normalized_cost(i.costs))),range(6))
  for f in range(2):
   for j in [2,1,0]:q.rxx(theta[1],f*3+j,f*3+(j+1)%3);q.ryy(theta[1],f*3+j,f*3+(j+1)%3)
  assert np.allclose(Statevector.from_instruction(q).probabilities(),ordering_probs(i,theta,True),atol=1e-12)
 fits=json.loads((D/'ordering_fits.json').read_text());assert len(fits)==30;assert all(r['seed']<15 and r['calls']<=r['cap'] and r['iterations']<=40 for r in fits)
 p=pd.read_csv(D/'ordering_instances.csv');assert len(p)==30 and set(p.seed)==set(range(15,30));assert np.isclose(p[p['mode']=='forward'].p_quality.mean(),.15798760124572203)
 larger=pd.read_csv(D/'larger_requirements.csv')
 for method,column in [('uniform','uniform_p_quality'),('structured','structured_p_quality')]:
  probs=larger[column].to_numpy();s=larger[method+'_required'].to_numpy();assert ((1-probs)**s<=.1+1e-12).all();assert ((1-probs[s>1])**(s[s>1]-1)>.1-1e-12).all()
 assert np.allclose(larger.allocated_service_j,larger.structured_required*larger.shot_s*20000)
 raw=pd.read_csv(ROOT/'results/v1_2/all_on/outcomes.csv');raw=raw[(raw.seed>=15)&(raw.algorithm=='onehot_structured')&(raw.shots==64)].reset_index(drop=True);cache={};replays=0
 for watts,u in [(10000,1),(10000,.5),(18691.2,.5)]:
  sc=replace(DEFAULT,qpu_w=watts,utilization=u,repetition_s=.0002,setup_s=.1);v,ready,_=raw_evaluate(raw,sc)
  for j,r in enumerate(raw.itertuples()):
   k=r.size,r.seed
   if k not in cache:cache[k]=E.workload(r.size,r.seed,incumbent=int(r.incumbent))
   e=E.event_energy(cache[k],[(ready[j],int(r.solution),'crossed')],sc);assert np.isclose(e['network_j'],v['network_j'][j],atol=1e-7);assert e['transition_j']==v['transition_j'][j];replays+=1
 # Verify every row of the four-way comparison selects the intended budget.
 from render_round5 import main as render
 import sys,tempfile
 with tempfile.TemporaryDirectory() as tmp:
  tmp=Path(tmp);(tmp/'tables').mkdir();(tmp/'v12_principal.tex').write_text('')
  argv=sys.argv;sys.argv=['render_round5.py','--input',str(D),'--output',str(tmp),'--manuscript',str(tmp),'--skip-legacy']
  try:render()
  finally:sys.argv=argv
  table=(tmp/'round5_marginal_map.tex').read_text()
  assert all(r'\ensuremath{'+x+'}' in table for x in ['10.67','9.93','9.05','8.96'])
 out=dict(ordering_forward_checks=4,independent_reverse_circuit_checks=4,training_fits=len(fits),train_test_separation=True,budget_caps=True,larger_per_instance_thresholds=len(larger)*2,independent_crossed_event_replays=replays)
 validation_path('test_round5').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
