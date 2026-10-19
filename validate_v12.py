#!/usr/bin/env python3
"""Audit stored experiments independently of the vectorized analysis."""
from pathlib import Path
from dataclasses import replace
from artifact_paths import validation_path
import json,hashlib
import numpy as np,pandas as pd
from energy_v12 import *
from analyze_v12 import raw_evaluate,DEFAULT
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/v1_2'

def main():
 report=dict(schema_version='1.2',workload_cases=0,sample_batches=0,scalar_event_replays=0,qubo_audits=0,training_only=True)
 for name,sleep in [('all_on',None),('sleep_0',0.),('sleep_10',10.),('sleep_50',50.)]:
  path=OUT/name;ins=pd.read_csv(path/'instances.csv');cl=pd.read_csv(path/'classical.csv');raw=pd.read_csv(path/'samples.csv');data=pd.read_csv(path/'outcomes.csv',low_memory=False);pi=pd.read_csv(path/'primary_instances.csv')
  assert len(ins)==120
  instances={}
  for r in ins.itertuples():
   i=workload(r.size,r.seed,sleep,incumbent=r.incumbent);instances[r.size,r.seed]=i
   assert score(i,r.incumbent)[0]
   np.testing.assert_allclose(r.optimum_w,i.metadata['power_w'][i.feasible].min(),atol=1e-8)
   assert r.onehot_states==i.metadata['nodes']**i.metadata['functions']
   assert r.cooptimal_states==np.count_nonzero(i.feasible&(abs(i.metadata['power_w']-r.optimum_w)<1e-8))
  for r in cl[cl.algorithm=='milp'].itertuples():
   assert instances[r.size,r.seed].feasible[r.solution];np.testing.assert_allclose(r.power_w,r.optimum_w,atol=1e-8)
  for (n,seed),g in raw.groupby(['size','seed']):
   i=instances[n,seed];states=g.solution.to_numpy(int)
   np.testing.assert_array_equal(g.feasible,i.feasible[states]);np.testing.assert_allclose(g.power_w,i.metadata['power_w'][states],atol=1e-8)
  for space in ['full','onehot']:
   a=raw[raw.algorithm=='classical_uniform_'+space].sort_values(['size','seed','shots','repeat']);b=raw[raw.algorithm=='quantum_uniform_'+space].sort_values(['size','seed','shots','repeat'])
   np.testing.assert_array_equal(a.solution,b.solution)
  a=raw[raw.algorithm=='onehot_transfer'].sort_values(['size','seed','shots','repeat']);b=raw[raw.algorithm=='onehot_structured'].sort_values(['size','seed','shots','repeat']);np.testing.assert_array_equal(a.solution,b.solution)
  for t in json.loads((path/'transfer.json').read_text()):assert t['training_seeds']==list(range(15))
  for fit in json.loads((path/'fits.json').read_text()):assert fit['seed']<15
  assert (pd.read_csv(path/'qubo_audit.csv').verified==True).all()
  values,ready,_=raw_evaluate(data)
  for j,r in enumerate(data.itertuples()):
   i=instances[r.size,r.seed];e=event_energy(i,[(ready[j],r.solution,r.algorithm)])
   for key in ['network_j','transition_j','delay_loss_j','quality_loss_j','fallback_loss_j']:np.testing.assert_allclose(e[key],values[key][j],atol=1e-7,rtol=1e-12)
  # Transition-rich sensitivity checks, including nonzero wake and migration delay.
  subset=data.iloc[::max(1,len(data)//200)]
  scenario=replace(DEFAULT,horizon=1.,migration_j=100.,migration_s=.01,wake_j=100,wake_s=.1)
  v,t,_=raw_evaluate(subset,scenario)
  for j,r in enumerate(subset.itertuples()):
   e=event_energy(instances[r.size,r.seed],[(t[j],r.solution,r.algorithm)],scenario)
   np.testing.assert_allclose(e['network_j']+e['transition_j'],v['network_j'][j]+v['transition_j'][j],atol=1e-7)
  np.testing.assert_allclose(pi.compute_j+pi.network_j+pi.transition_j,pi.total_j,atol=1e-7)
  np.testing.assert_allclose(pi[['preparation_j','optimization_j','communication_j','postprocessing_j','offline_j','service_j']].sum(axis=1),pi.compute_j,atol=1e-7)
  v1,_,_=raw_evaluate(subset,replace(DEFAULT,reuse=1));v100,_,_=raw_evaluate(subset,replace(DEFAULT,reuse=100))
  np.testing.assert_allclose(v1['compute_j']-v100['compute_j'],v1['offline_j']-v100['offline_j'],atol=1e-8)
  dec=pd.read_csv(path/'decomposition.csv');np.testing.assert_allclose(dec[['compute_j','transition_j','delay_loss_j','quality_loss_j','fallback_loss_j']].sum(axis=1),dec.excess_j,atol=1e-7)
  pol=pd.read_csv(path/'policies.csv');assert set(pol.seed)==set(range(15,30));assert not pol[pol.policy=='energy_aware'].invoked_quantum.any() if name=='all_on' else True
  ev=pd.read_csv(path/'admission_events.csv')
  assert (ev[ev.rejection_reason!='accepted'].transition_j==0).all()
  assert not ev.duplicated(['transition_id']).any()
  event_sums={}
  for (n,seed,policy,outcome),g in ev.groupby(['size','seed','policy','outcome_id']):
   inst=instances[n,seed];state=int(g.incumbent_id.iloc[0]);clock=0.;network=0.;transition=0.
   for r in g.sort_values('event_id').itertuples():
    if r.rejection_reason=='accepted':
     assert r.arrival_s<=inst.deadline_s and r.arrival_s<60
     network+=(r.arrival_s-clock)*score(inst,state)[1];clock=r.arrival_s
     state=int(r.computed_action);transition+=r.transition_j
    assert int(r.applied_action)==state
   network+=(60-clock)*score(inst,state)[1]
   key=(n,seed,policy);v=event_sums.setdefault(key,np.zeros(2));v+=float(g.solution_probability.iloc[0])*np.array([network,transition])
  for r in pol.itertuples():
   np.testing.assert_allclose(event_sums[r.size,r.seed,r.policy],[r.network_j,r.transition_j],atol=1e-7)
   np.testing.assert_allclose(r.compute_j+r.network_j+r.transition_j,r.total_j,atol=1e-7)

  report['workload_cases']+=len(ins);report['sample_batches']+=len(raw);report['scalar_event_replays']+=len(data);report['qubo_audits']+=120
  print('audit',name,'passed',flush=True)
 grid=pd.read_csv(OUT/'all_on/sensitivity.csv');assert grid.scenario_id.nunique()==8640
 report['base_scenarios']=8640
 if (OUT/'noise/manifest.json').exists():
  m=json.loads((OUT/'noise/manifest.json').read_text());assert m['conditions']==240
  p=pd.read_csv(OUT/'noise/probabilities.csv');assert p.compilation_probability_error.max()<1e-6
  assert p.target.eq('all_to_all').all();assert p.physical_width.max()<=12
  noise_samples=pd.read_csv(OUT/'noise/samples.csv');assert len(noise_samples)==28800
  for (n,seed),g in noise_samples.groupby(['size','seed']):
   inst=workload(n,seed);states=g.solution.to_numpy(int);power=inst.metadata['power_w'];opt=power[inst.feasible].min()
   np.testing.assert_array_equal(g.feasible,inst.feasible[states]);np.testing.assert_allclose(g.power_w,power[states],atol=1e-8)
   np.testing.assert_array_equal(g.quality,inst.feasible[states]&((power[states]-opt)/(opt+1)<=.05))
  for r in p.itertuples():
   inst=workload(r.size,r.seed);v=np.load(OUT/'noise'/f'n{r.size}_s{r.seed}_{r.algorithm}_r{r.cx_depolarizing_parameter:g}.npz')['probability']
   assert np.min(v)>=-1e-12 and abs(v.sum()-1)<1e-9
   power=inst.metadata['power_w'];opt=power[inst.feasible].min();good=inst.feasible&((power-opt)/(opt+1)<=.05)
   np.testing.assert_allclose(v[good].sum(),r.p_quality,atol=1e-10)
  report['noise_conditions']=240;report['noise_sampling_batches']=m['sampling_batches']
 else:report['noise_status']='pending'
 report['status']='passed' if 'noise_status' not in report else 'partial; noise pending'
 validation_path('validate_v12').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
