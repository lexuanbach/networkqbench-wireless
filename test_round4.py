#!/usr/bin/env python3
"""Audit local additions without remeasuring the scientific campaigns."""
from pathlib import Path
from artifact_paths import validation_path
import json, hashlib, itertools
from dataclasses import replace
import numpy as np
import pandas as pd
import energy_v12 as E
from energy_model import assignment_state
from round4 import deadline_milp, deploy_sparse, fit_family, FAMILIES
from revision_v121 import sparse_workload, paired_interval
from analyze_v12 import raw_evaluate, DEFAULT
from networkqbench import _normalized_cost
ROOT=Path(__file__).resolve().parent;D=ROOT/'results/round4_20261019'

def main():
 # Exhaustive physical scoring independently certifies small sparse MILPs.
 small=0
 for n,f in [(3,2),(4,2),(3,3)]:
  for seed in [0,3]:
   inst,_=sparse_workload(n,f,seed);powers=[]
   for x in itertools.product(range(n),repeat=f):
    state=assignment_state(x,n);ok,p=E.score(inst,state)
    loads=np.bincount(x,weights=inst.metadata['demands'],minlength=n)
    valid=bool(np.all(loads<=inst.metadata['capacities']+1e-10) and sum(inst.metadata['latency'][x[k],x[k+1]] for k in range(f-1))<=4+1e-10)
    assert ok==valid
    if valid:powers.append(p)
   r=deadline_milp(inst);assert r['exact'];assert np.isclose(r['power_w'],min(powers),atol=1e-7);small+=1
 inst,_=sparse_workload(3,2,0);r=deadline_milp(inst,deadline=0);assert not r['solver_called'] and r['solution'] is None
 fallback=deploy_sparse(inst,None,.1,5);assert not fallback['accepted'] and fallback['transition_j']==0 and fallback['total_j']==60*fallback['incumbent_w']+5
 no=deploy_sparse(inst,inst.metadata['incumbent'],0,0);assert no['compute_j']==no['transition_j']==0
 best=deadline_milp(inst)['solution'];late=deploy_sparse(inst,best,inst.deadline_s+1,10);assert not late['accepted'] and late['compute_j']==10
 # Validate fitting membership, accounting, caps, and the stored objective itself.
 fits=0;replays=0
 for folder in sorted((D/'transfer').iterdir()):
  if not folder.is_dir():continue
  for r in json.loads((folder/'fits.json').read_text()):
   assert r['family_evaluations']<=min(r['historical_per_seed_evaluations'].values());assert set(map(int,r['historical_per_seed_evaluations']))==set(range(15));assert r['training_seeds']==list(range(15));assert r['instance_evaluations']==15*r['family_evaluations']<=r['historical_instance_budget'];assert r['iterations']<=40
   sleep=None if r['workload']=='all_on' else float(r['workload'].split('_')[1]);values=[]
   for seed in range(15):
    i=E.workload(r['size'],seed,sleep);p=E.onehot_probs(i,r['theta']);opt=i.metadata['power_w'][i.feasible].min()
    mask=i.feasible & ((i.metadata['power_w']-opt)/(opt+1)<=.05)
    values.append(p@_normalized_cost(i.costs) if r['objective']=='expectation' else -p[mask].sum())
   assert np.isclose(np.mean(values),r['objective_value'],atol=1e-10);fits+=1
  raw=pd.read_csv(folder/'outcomes.csv');assert len(raw)==7200 and set(raw.seed)==set(range(15,30));assert set(raw.shots)=={16,64,256,1024}
  assert raw.groupby(['size','seed','algorithm','shots']).weight.sum().sub(1).abs().max()<1e-10
  # One sample from every instance/method/budget plus every first-size batch.
  selected=raw[(raw.repeat==0)|(raw.seed==15)].reset_index(drop=True);v,ready,_=raw_evaluate(selected);cache={}
  for j,r in enumerate(selected.itertuples()):
   key=(r.size,r.seed)
   if key not in cache:cache[key]=E.workload(r.size,r.seed,None if r.workload=='all_on' else float(r.workload.split('_')[1]),incumbent=int(r.incumbent))
   e=E.event_energy(cache[key],[(ready[j],int(r.solution),'check')],DEFAULT)
   assert np.isclose(e['network_j'],v['network_j'][j],atol=1e-7);assert np.isclose(e['transition_j'],v['transition_j'][j],atol=1e-8);replays+=1
 # Shared profiles and circular controls reproduce frozen candidate outcomes.
 pi=pd.read_csv(D/'transfer_instances.csv');common=pi[pi.profile=='common'];assert common.groupby(['workload','size','seed','shots','boundary']).compute_j.agg(lambda v:np.ptp(v)).max()<1e-8
 frozen=pd.read_csv(ROOT/'results/v1_2/all_on/primary_instances.csv').query("algorithm=='onehot_structured' and seed>=15").set_index(['size','seed','shots'])
 current=common.query("workload=='all_on' and algorithm=='circular_expectation' and boundary=='allocated'").set_index(['size','seed','shots']).loc[frozen.index]
 for c in ['quality','accepted','total_j']:assert np.allclose(current[c],frozen[c])
 # Replay all 600 sparse records with a separately expressed event integration.
 runs=pd.read_csv(D/'deadline_instances.csv',dtype={'solution':str,'applied_action':str});assert len(runs)==600;assert len(runs[['nodes','functions','seed']].drop_duplicates())==100
 for r in runs.itertuples():
  inst,_=sparse_workload(r.nodes,r.functions,r.seed);m=inst.metadata;inc=m['incumbent'];p0=E.score(inst,inc)[1]
  state=None if pd.isna(r.solution) else int(r.solution);valid,p=E.score(inst,state) if state is not None else (False,None)
  accept=False;trans=0;segments=[(0,60,p0)]
  if valid and r.runtime_s<=r.deadline_s and r.runtime_s<60:
   moves=np.count_nonzero(E.decode(inst,inc)!=E.decode(inst,state));cost=10*moves
   if (60-r.runtime_s)*(p0-p)>cost+1e-12:accept=True;trans=cost;segments=[(0,r.runtime_s,p0),(r.runtime_s,60,p)]
  network=sum((end-start)*power for start,end,power in segments)
  assert accept==r.accepted;assert np.isclose(network,r.network_j);assert trans==r.transition_j;assert np.isclose(network+trans+50*r.runtime_s,r.total_j)
  if not r.solver_called:assert r.algorithm=='milp' and r.solve_s==0 and r.construction_s>=r.deadline_s
  if r.runtime_s>r.deadline_s:assert not r.accepted
  if pd.isna(r.reference_bound_w):assert pd.isna(r.bound_gap_w)
  if not r.reference_exact:assert pd.isna(r.exact_gap_w)
 assert (~runs.solver_called).any() and (runs.overrun_s>0).any()
 # Pair round-specific event results first, then resample instance averages.
 rounds=pd.read_csv(D/'template_round_instances.csv');components=['preparation_j','optimization_j','communication_j','postprocessing_j','offline_j','service_j','transition_j','network_j']
 assert np.allclose(rounds[components].sum(axis=1),rounds.total_j)
 avg=pd.read_csv(D/'template_paired_instances.csv');summary=pd.read_csv(D/'template_paired_summary.csv')
 for r in summary.itertuples():
  g=avg[avg.boundary==r.boundary];low,high=paired_interval(g,r.metric);assert np.isclose(g[r.metric].mean(),r.mean);assert np.allclose([low,high],[r.low,r.high])
 # Campaign source snapshots preserve the exact measured implementation.
 for name,snapshot in [('transfer','round4_transfer'),('scaling','round4_scaling')]:
  manifest=json.loads((D/(name+'_manifest.json')).read_text())
  for filename,digest in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/'source_snapshots'/snapshot/filename).read_bytes()).hexdigest()==digest
 report=dict(sparse_exhaustive_cases=small,direct_fits= fits,training_separation=True,evaluation_caps=True,independent_quantum_event_replays=replays,sparse_event_replays=len(runs),deadline_exhaustion_and_overruns=True,zero_compute_fallback=True,missing_bounds_preserved=True,round_pairing=True,common_profiles=True,campaign_source_hashes=True)
 validation_path('test_round4').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
