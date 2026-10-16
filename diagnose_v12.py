#!/usr/bin/env python3
"""Additional reproducible diagnostics; no new physical measurements."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np,pandas as pd
from energy_v12 import *
from analyze_v12 import prepare,per_instance,summary,raw_evaluate,DEFAULT,OUT
from energy_model import minimum_shots
ROOT=Path(__file__).resolve().parent

def main():
 folder=OUT/'all_on';data=pd.read_csv(folder/'outcomes.csv');test=data[data.seed>=15]
 # Utilization distribution over physical servers; instance identity retained.
 rows=[]
 for name in ['all_on','sleep_0','sleep_10','sleep_50']:
  p=OUT/name;ins=pd.read_csv(p/'instances.csv');cl=pd.read_csv(p/'classical.csv')
  for r in ins.itertuples():
   i=workload(r.size,r.seed,None if name=='all_on' else float(name.split('_')[1]));opt=int(cl[(cl['size']==r.size)&(cl.seed==r.seed)&(cl.algorithm=='milp')].solution.iloc[0])
   for stage,state in [('previous',r.incumbent),('optimum',opt)]:
    for server,u in enumerate(i.metadata['utilizations'][int(state)]):rows.append(dict(workload=name,size=r.size,seed=r.seed,stage=stage,server=server,utilization=u))
 pd.DataFrame(rows).to_csv(OUT/'utilizations.csv',index=False)
 # Component rows carry explicit boundary, evidence, and scenario provenance.
 ledger=[]
 for family in ['all_on','sleep_0','sleep_10','sleep_50']:
  pi=pd.read_csv(OUT/family/'primary_instances.csv')
  for component in ['preparation','optimization','communication','postprocessing','offline','service','network','transition']:
   part=pi[['workload','size','seed','algorithm','shots']].copy();part['component']=component;part['joules']=pi[component+'_j']
   part['schema_version']='1.2';part['accounting_boundary']='allocated operational';part['evidence_kind']='modeled_energy'
   part['source']={'preparation':'measured representation/build/compile time x host watts','optimization':'measured local solve or declared online host-update duration x host watts','communication':'declared submission/retrieval client duration x host watts','postprocessing':'measured selection-time profile x host watts','offline':'recorded simulator-training time x host watts / reuse','service':'dependency-scheduled occupied duration x declared envelope','network':'SPEC server curve and event integral','transition':'illustrative move/wake coefficient x accepted transitions'}[component]
   part['scenario']='primary_previous_placement_benefit_gate';part['amortization_count']=100 if component=='offline' else 1;ledger.append(part)
 pd.concat(ledger).to_csv(OUT/'energy_ledger.csv',index=False)
 # Source-grounded component model: 16.1 kW cooling, 1.1 kW cluster,
 # 0.8 kW host, and 172.8 W per feed line serving up to 5 qubits.
 power_rows=[]
 for n in [6,8,9,12,16,50]:power_rows.append(dict(qubits=n,feed_lines=int(np.ceil(n/5)),cooling_w=16100,cluster_w=1100,host_w=800,per_line_w=172.8,total_w=18000+172.8*np.ceil(n/5)))
 pd.DataFrame(power_rows).to_csv(OUT/'power_components.csv',index=False)
 model=summary(per_instance(test,replace(DEFAULT,qpu_w=18691.2)))
 model.to_csv(OUT/'component_power_sensitivity.csv',index=False)
 # Preserve and decompose the original stress-case contrast without changing its measurements.
 old=pd.read_csv(ROOT/'results/sustainability/primary_decisions.csv');old=old[old.seed>=15]
 parts=[]
 for r in old.itertuples():
  p0=r.incumbent_w;pc=r.optimum_w;T=r.delay_s;H=60
  accepted=r.on_time*r.feasible_rate
  delay=accepted*T*(p0-pc)
  quality=(H-T)*r.on_time*(r.feasible_power-r.feasible_rate*pc)
  fallback=(1-accepted)*H*(p0-pc)
  parts.append(dict(size=r.size,seed=r.seed,algorithm=r.algorithm,shots=r.shots,compute_j=r.compute_j,transition_j=r.transition_j,delay_loss_j=delay,quality_loss_j=quality,fallback_loss_j=fallback,total_j=r.total_j))
 parts=pd.DataFrame(parts);base=parts[parts.algorithm=='milp'].set_index(['size','seed'])
 for c in ['compute_j','transition_j','delay_loss_j','quality_loss_j','fallback_loss_j','total_j']:parts[c]-=base.loc[list(zip(parts['size'],parts.seed)),c].to_numpy()
 assert np.allclose(parts[['compute_j','transition_j','delay_loss_j','quality_loss_j','fallback_loss_j']].sum(axis=1),parts.total_j,atol=1e-7)
 parts.groupby(['algorithm','shots']).mean(numeric_only=True).to_csv(OUT/'stress_decomposition.csv')
 # Frozen-overhead frontier: exact per-case probability; postprocessing is held
 # at its measured 64-shot reference (not a general hardware guarantee).
 rows=[]
 for (n,seed,alg),g in test[test.shots==64].groupby(['size','seed','algorithm']):
  if not g.is_quantum.iloc[0]:continue
  vals,ready,arrival=raw_evaluate(g);p=float(g.p_quality.iloc[0]);needed=minimum_shots(p)
  shot=float(g.shot_s.iloc[0]);R=float(g.evals.iloc[0]);t0=float(ready[0])-R*64*shot
  e0=float(vals['compute_j'][0])-DEFAULT.qpu_w*R*64*shot
  total_t=t0+needed*R*shot;energy=e0+needed*R*shot*DEFAULT.qpu_w
  rows.append(dict(size=n,seed=seed,algorithm=alg,p_quality=p,required_shots=needed,required_s=total_t,required_j=energy,deadline_s=float(g.deadline_s.iloc[0]),time_ratio=total_t/g.deadline_s.iloc[0],admissible=total_t<=g.deadline_s.iloc[0] and total_t<60))
 pd.DataFrame(rows).to_csv(OUT/'frontier.csv',index=False)
 # Recheck the reviewed 13.23-J corner under both actuation rules, using
 # the present matched phase/sample records and concentrated incumbent.
 stress=test.copy()
 for (n,seed),ix in stress.groupby(['size','seed']).groups.items():
  inst=workload(n,seed);old_state=inst.metadata['incumbent'];old=decode(inst,old_state)
  stress.loc[ix,'incumbent']=old_state;stress.loc[ix,'incumbent_w']=score(inst,old_state)[1]
  stress.loc[ix,'moves']=[0 if decode(inst,int(v)) is None else np.count_nonzero(decode(inst,int(v))!=old) for v in stress.loc[ix,'solution']]
 corners=[]
 for gate in [False,True]:
  corner=replace(DEFAULT,local_w=10,qpu_w=10000,reuse=1000,horizon=1,migration_j=100,boundary='marginal',marginal_fraction=.01,benefit_gate=gate)
  a=summary(per_instance(stress,corner));a['benefit_gate']=gate;corners.append(a)
 pd.concat(corners).to_csv(OUT/'reviewed_stress_corner.csv',index=False)
 # Main lifetime curves get conditional paired instance uncertainty, not scenario resampling.
 horizons=[]
 for H in [1,10,60,300,3600]:
  result=summary(per_instance(test,replace(DEFAULT,horizon=H)));result['horizon']=H;horizons.append(result)
 pd.concat(horizons).to_csv(OUT/'horizon_intervals.csv',index=False)
 # Recompute every acceptance breakpoint under a hypothetical service-speed scale.
 speed=[]
 for factor in [0.,.001,.01,.1,.25,.5,1.]:
  scaled=test.copy();scaled['shot_s']=scaled.shot_s*factor
  result=summary(per_instance(scaled));result['service_duration_factor']=factor;speed.append(result)
 pd.concat(speed).to_csv(OUT/'service_compression.csv',index=False)
 # Illustrative carbon conversion; no location or lifecycle footprint implied.
 s=pd.read_csv(folder/'primary_summary.csv');carbon=[]
 for r in s.itertuples():
  for intensity in [50,400,800]:carbon.append(dict(algorithm=r.algorithm,shots=r.shots,gCO2e_per_kWh=intensity,total_gCO2e=r.total_j/3.6e6*intensity,excess_gCO2e=r.excess_j/3.6e6*intensity))
 pd.DataFrame(carbon).to_csv(OUT/'illustrative_carbon.csv',index=False)
 # Noise probabilities and outcomes belong only to their actual all-to-all target.
 noise=OUT/'noise'
 if (noise/'manifest.json').exists():
  prob=pd.read_csv(noise/'probabilities.csv');raw=pd.read_csv(noise/'samples.csv');cir=pd.read_csv(folder/'circuits.csv');ins=pd.read_csv(folder/'instances.csv');cl=pd.read_csv(folder/'classical.csv')
  nr=[]
  for (n,seed,alg,rate,budget),g in raw.groupby(['size','seed','algorithm','noise','shots']):
   cfg=prob[(prob['size']==n)&(prob.seed==seed)&(prob.algorithm==alg)&(prob.cx_depolarizing_parameter==rate)].iloc[0]
   r=cir[(cir['size']==n)&(cir.seed==seed)&(cir.algorithm==alg)].iloc[0];ref=ins[(ins['size']==n)&(ins.seed==seed)].iloc[0]
   inst=workload(n,seed,incumbent=int(ref.incumbent));post=float(data[(data['size']==n)&(data.seed==seed)&(data.algorithm==alg)&(data.shots==budget)].postprocessing_s.iloc[0])
   # Matched target compilation duration; no grid duration or grid compile time reused.
   host=ref.dense_preparation_s+r.circuit_build_s+cfg.compile_s+post+.0025
   ready=host+.01+budget*cfg.shot_s;compute=50*host+20000*budget*cfg.shot_s+50*r.offline_s/100
   totals=[]
   for state in g.solution:
    ev=event_energy(inst,[(ready,int(state),alg)]);totals.append(compute+ev['network_j']+ev['transition_j'])
   baseline=cl[(cl['size']==n)&(cl.seed==seed)&(cl.algorithm=='milp')].iloc[0];ev=event_energy(inst,[(baseline.runtime_s,int(baseline.solution),'milp')]);base=50*baseline.runtime_s+ev['network_j']+ev['transition_j']
   nr.append(dict(size=n,seed=seed,algorithm=alg,noise=rate,shots=budget,p_quality=cfg.p_quality,required_shots=minimum_shots(np.clip(cfg.p_quality,0,1)),quality=g.quality.mean(),timely=ready<=inst.deadline_s and ready<60,compute_j=compute,total_j=np.mean(totals),excess_j=np.mean(totals)-base,cx=cfg.cx,shot_s=cfg.shot_s,target='all_to_all'))
  pd.DataFrame(nr).to_csv(OUT/'noise_instances.csv',index=False)
  pd.DataFrame(nr).groupby(['size','algorithm','noise','shots']).mean(numeric_only=True).to_csv(OUT/'noise_summary.csv')
 # Research-execution durations are retained separately; no power meter was available.
 research=[]
 for family in ['all_on','sleep_0','sleep_10','sleep_50']:
  for f in json.loads((OUT/family/'fits.json').read_text()):
   research.append(dict(workload=family,size=f['size'],seed=f['seed'],phase='offline simulator fitting',depth=f['depth'],elapsed_s=f['simulation_s'],joules=None,evidence_kind='measured_local_time; energy unknown',source=family+'/fits.json'))
 if (noise/'manifest.json').exists():
  for r in pd.read_csv(noise/'probabilities.csv').itertuples():research.append(dict(workload='all_on',size=r.size,seed=r.seed,phase='exact noisy research simulation',depth=1,elapsed_s=r.simulation_s,joules=None,evidence_kind='measured_local_time; energy unknown',source='noise/probabilities.csv',algorithm=r.algorithm,noise=r.cx_depolarizing_parameter))
 pd.DataFrame(research).to_csv(OUT/'research_execution.csv',index=False)
 print('diagnostics complete',flush=True)
if __name__=='__main__':main()
