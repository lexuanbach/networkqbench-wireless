#!/usr/bin/env python3
"""Analyze frozen 1.2 distributions through common physical accounting."""
from dataclasses import asdict,replace
from pathlib import Path
import argparse,itertools,json,time,hashlib
import numpy as np
import pandas as pd
from energy_v12 import *
from energy_model import PROFILES,minimum_shots
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/v1_2'
DEFAULT=Scenario()
LABELS={'milp':'HiGHS MILP','local_search':'Local search','heuristic':'Greedy placement','classical_uniform_full':'Classical uniform full','classical_uniform_onehot':'Classical uniform one-hot','quantum_uniform_full':'Quantum uniform full','quantum_uniform_onehot':'Quantum uniform one-hot','qaoa_online':'Online QAOA','qaoa_transfer':'Standard-X transfer','onehot_transfer':'Dense QAOA','onehot_structured':'Structured QAOA','onehot_p2':'Structured QAOA p=2'}
KEY=['workload','size','seed','algorithm','shots']
METRICS=['compute_j','preparation_j','optimization_j','communication_j','postprocessing_j','offline_j','service_j','network_j','transition_j','total_j','quality','optimal','feasible','timely','useful','accepted','applied_quality','delay_loss_j','quality_loss_j','fallback_loss_j']


def prepare(folder):
 ins=pd.read_csv(folder/'instances.csv');cir=pd.read_csv(folder/'circuits.csv');s=pd.read_csv(folder/'samples.csv');cl=pd.read_csv(folder/'classical.csv')
 timing=s.groupby(KEY,as_index=False)[['postprocessing_s','sampling_s']].mean()
 g=s.groupby(KEY+['solution'],as_index=False).agg(count=('repeat','size'),quality=('quality','first'),optimal=('optimal','first'),feasible=('feasible','first'),power_w=('power_w','first'),p_quality=('p_quality','first'),p_optimal=('p_optimal','first'))
 g['weight']=g['count']/30;g=g.merge(timing,on=KEY).merge(ins,on=['workload','size','seed']).merge(cir,on=['workload','size','seed','algorithm']);g['local_s']=g.sampling_s
 for k,v in dict(shots=0,weight=1.,quality=True,optimal=True,feasible=True,p_quality=1.,p_optimal=1.,postprocessing_s=0.,sampling_s=0.,is_quantum=False,offline_s=0.,evals=0,compile_s=0.,circuit_build_s=0.,shot_s=0.,qaoa_depth=0,encoding='classical',mixer='none',cx=0,target='local').items():cl[k]=v
 cl['local_s']=cl.runtime_s;cl['optimal']=np.isclose(cl.power_w,cl.optimum_w,atol=1e-8);cl['quality']=(cl.power_w-cl.optimum_w)/(cl.optimum_w+1)<=.05;cl['p_quality']=cl.quality.astype(float);cl['p_optimal']=cl.optimal.astype(float)
 data=pd.concat([g,cl],ignore_index=True);data['moves']=0.;data['wakes']=0.
 sleep=None if folder.name=='all_on' else float(folder.name.split('_')[1])
 for (n,seed),ix in data.groupby(['size','seed']).groups.items():
  i=workload(n,seed,sleep);rows=data.loc[ix];old=decode(i,int(rows.incumbent.iloc[0]));new=[decode(i,int(x)) for x in rows.solution]
  data.loc[ix,'moves']=[0 if a is None else int(np.count_nonzero(a!=old)) for a in new]
  data.loc[ix,'wakes']=[0 if a is None or sleep is None else len(set(a)-set(old)) for a in new]
 for c in ['p_quality','p_optimal']:
  assert data[c].between(-1e-12,1+1e-12).all()
  data[c]=data[c].clip(0,1)
 data['lookup_s']=0.
 return data


def raw_evaluate(d,s=DEFAULT):
 q=d.is_quantum.to_numpy(bool);queue,submit,roundtrip=PROFILES[s.profile]
 prep=np.where(q,d.dense_preparation_s+d.circuit_build_s+d.compile_s,0.)
 opt=np.where(q,np.maximum(d.evals-1,0)*.0001,d.local_s)
 comm=np.where(q,d.evals*(submit+roundtrip),0.)
 service=np.where(q,d.evals*(d.shots*(d.shot_s+s.repetition_s)+s.setup_s),0.)
 ready=prep+opt+comm+service+d.postprocessing_s+np.where(q,queue,0.)+d.lookup_s
 quantum_power=s.qpu_w*(s.marginal_fraction if s.boundary=='marginal' else 1/s.utilization)
 result=dict(preparation_j=np.asarray(prep*s.local_w),optimization_j=np.asarray((opt+d.lookup_s)*s.local_w),communication_j=np.asarray(comm*s.local_w),postprocessing_j=np.asarray(d.postprocessing_s*s.local_w),offline_j=np.asarray(d.offline_s*s.local_w/s.reuse),service_j=np.asarray(service*quantum_power))
 compute=sum(result.values());cost=np.asarray(d.moves*s.migration_j+d.wakes*s.wake_j)
 arrival=np.asarray(ready+d.moves*s.migration_s+(d.wakes>0)*s.wake_s)
 timely=(arrival<=d.deadline_s)&(arrival<s.horizon)
 accepted=np.asarray(d.feasible&timely)
 if s.benefit_gate:accepted=accepted & np.asarray((s.horizon-arrival)*(d.incumbent_w-d.power_w)>cost+1e-12)
 network=np.asarray(d.incumbent_w*s.horizon+accepted*(s.horizon-arrival)*(d.power_w-d.incumbent_w))
 transition=accepted*cost
 result.update(compute_j=np.asarray(compute),network_j=network,transition_j=transition,total_j=compute+network+transition,quality=d.quality.to_numpy(float),optimal=d.optimal.to_numpy(float),feasible=d.feasible.to_numpy(float),timely=np.asarray(timely,float),useful=np.asarray(timely*d.quality,float),accepted=accepted.astype(float),applied_quality=np.where(accepted,d.quality,(d.incumbent_w-d.optimum_w)/(d.optimum_w+1)<=.05).astype(float),delay_loss_j=np.asarray(accepted*arrival*(d.incumbent_w-d.optimum_w)),quality_loss_j=np.asarray(accepted*(s.horizon-arrival)*(d.power_w-d.optimum_w)),fallback_loss_j=np.asarray((~accepted)*s.horizon*(d.incumbent_w-d.optimum_w)))
 return result,np.asarray(ready),arrival


def per_instance(d,s=DEFAULT):
 values,ready,arrival=raw_evaluate(d,s);out=d[KEY+['p_quality','p_optimal','weight']].copy()
 for k,v in values.items():out[k]=v*d.weight
 out['ready_s']=ready*d.weight;out['arrival_s']=arrival*d.weight
 for k in ['p_quality','p_optimal']:out[k]*=out.weight
 out=out.groupby(KEY,as_index=False).sum(numeric_only=True)
 base=out[out.algorithm=='milp'][['workload','size','seed','total_j']].rename(columns={'total_j':'milp_j'})
 out=out.merge(base,on=['workload','size','seed'],validate='many_to_one');out['excess_j']=out.total_j-out.milp_j
 return out


def summary(pi):
 keys=['algorithm','shots'];out=pi.groupby(keys,as_index=False)[METRICS+['excess_j','p_quality','p_optimal','ready_s','arrival_s']].mean()
 extra=pi.groupby(keys).agg(instances=('seed','size'),p_quality_median=('p_quality','median'),wins=('excess_j',lambda v:int((v< -1e-8).sum())),ties=('excess_j',lambda v:int((abs(v)<=1e-8).sum())))
 out=out.merge(extra,on=keys);out['losses']=out.instances-out.wins-out.ties
 out['compute_per_useful_j']=np.divide(out.compute_j.to_numpy(),out.useful.to_numpy(),out=np.full(len(out),np.inf),where=out.useful.to_numpy()>0)
 rng=np.random.default_rng(20261016)
 for index,r in out.iterrows():
  g=pi[(pi.algorithm==r.algorithm)&(pi.shots==r.shots)];means=[]
  for _,sg in g.groupby('size'):
   x=sg.excess_j.to_numpy();means.append(x[rng.integers(len(x),size=(20000,len(x)))].mean(axis=1))
  z=np.mean(means,axis=0);out.loc[index,'ci_low_j'],out.loc[index,'ci_high_j']=np.quantile(z,[.025,.975])
  quality_rng=np.random.default_rng(20261017)
  for metric in ['quality','useful']:
   strata=[]
   for _,sg in g.groupby('size'):
    x=sg[metric].to_numpy();strata.append(x[quality_rng.integers(len(x),size=(20000,len(x)))].mean(axis=1))
   out.loc[index,metric+'_ci_low'],out.loc[index,metric+'_ci_high']=np.quantile(np.mean(strata,axis=0),[.025,.975])
 return out


def sweep(d,scenarios,path):
 groups=d[['algorithm','shots']].apply(tuple,axis=1);codes,unique=pd.factorize(groups,sort=True);weight=d.weight.to_numpy();den=np.bincount(codes,weights=weight)
 rows=[]
 for sid,s in enumerate(scenarios):
  values,_,_=raw_evaluate(d,s);means={k:np.bincount(codes,weights=v*weight)/den for k,v in values.items() if k in ['total_j','compute_j','network_j','transition_j','quality','timely','useful','accepted','applied_quality']}
  bi=list(unique).index(('milp',0));baseline=means['total_j'][bi]
  for j,(alg,shots) in enumerate(unique):rows.append(dict(scenario_id=sid,algorithm=alg,shots=shots,**asdict(s),**{k:v[j] for k,v in means.items()},excess_j=means['total_j'][j]-baseline))
  if sid%1000==0:print(path.name,sid,flush=True)
 out=pd.DataFrame(rows);out.to_csv(path,index=False);return out


def diagnostics(d,pi,folder):
 # Conditional power crossing holds arrival fixed. Duration crossing is piecewise
 # and is reported as the maximum grid-tested ready time in a separate frontier.
 rows=[]
 for r in pi.itertuples():
  if r.algorithm in ['milp','local_search','heuristic'] or r.algorithm.startswith('classical'):continue
  source=d[(d['size']==r.size)&(d.seed==r.seed)&(d.algorithm==r.algorithm)&(d.shots==r.shots)]
  service_seconds=float(np.average(source.evals*source.shots*source.shot_s,weights=source.weight))
  nonq=r.total_j-r.service_j;headroom=r.milp_j-nonq
  threshold=headroom/service_seconds if service_seconds>0 and headroom>0 else np.nan
  no_offline=r.total_j-r.offline_j;offline=r.offline_j*DEFAULT.reuse;room=r.milp_j-no_offline
  minimum=math.floor(offline/room)+1 if room>0 else np.nan
  rows.append(dict(size=r.size,seed=r.seed,algorithm=r.algorithm,shots=r.shots,qpu_power_threshold_w=threshold,headroom_j=headroom,minimum_reuse=minimum,reason='positive fixed-arrival headroom' if headroom>0 else 'non-QPU terms already exceed MILP'))
 pd.DataFrame(rows).to_csv(folder/'thresholds.csv',index=False)
 probs=d[KEY+['p_quality','p_optimal']].drop_duplicates(KEY);rows=[]
 for r in probs.itertuples():
  for epsilon,p in [(0.,r.p_optimal),(.05,r.p_quality)]:
   for alpha in [.9,.95]:rows.append(dict(size=r.size,seed=r.seed,algorithm=r.algorithm,shots=r.shots,epsilon=epsilon,alpha=alpha,p=p,required_shots=minimum_shots(p,alpha)))
 pd.DataFrame(rows).drop_duplicates(['size','seed','algorithm','epsilon','alpha']).to_csv(folder/'shot_requirements.csv',index=False)
 # Exact additive decomposition against the paired MILP, including transition.
 parts=['compute_j','transition_j','delay_loss_j','quality_loss_j','fallback_loss_j']
 base=pi[pi.algorithm=='milp'].set_index(['size','seed'])
 decomp=pi[KEY+['excess_j']].copy()
 for k in parts:decomp[k]=pi[k].to_numpy()-base.loc[list(zip(pi['size'],pi.seed)),k].to_numpy()
 assert np.allclose(decomp[parts].sum(axis=1),decomp.excess_j,atol=1e-7)
 decomp.to_csv(folder/'decomposition.csv',index=False)


def policies(data,pi,folder):
 train=pi[pi.seed<15];test=pi[pi.seed>=15]
 timingfile=folder/'policy_timings.json';timings=json.loads(timingfile.read_text()) if timingfile.exists() else {}
 calibration_start=time.perf_counter()
 cal=train.groupby(['size','algorithm','shots'],as_index=False).agg(predicted_quality=('quality','mean'),predicted_delay=('arrival_s',lambda v:v.quantile(.9)),predicted_total=('total_j','mean'))
 calibration_s=timings.get('calibration_s',time.perf_counter()-calibration_start);timings['calibration_s']=calibration_s
 cal.to_csv(folder/'admission_calibration.csv',index=False)
 fits=json.loads((folder/'fits.json').read_text());cir=pd.read_csv(folder/'circuits.csv');cl=pd.read_csv(folder/'classical.csv');ins=pd.read_csv(folder/'instances.csv');oldfits=pd.read_csv(ROOT/'results/sustainability/fits.csv')
 offline={}
 for n in SIZES:
  offline[n]=calibration_s+sum(f['simulation_s'] for f in fits if f['size']==n)+float(cir[(cir['size']==n)&(cir.seed<15)][['circuit_build_s','compile_s']].sum().sum())+float(cl[(cl['size']==n)&(cl.seed<15)].runtime_s.sum())+float(ins[(ins['size']==n)&(ins.seed<15)].dense_preparation_s.sum())
  acquisition=data[(data['size']==n)&(data.seed<15)].groupby(KEY)[['postprocessing_s','sampling_s']].first()
  offline[n]+=30*float(acquisition.sum().sum())
  if folder.name=='all_on':offline[n]+=float(oldfits[(oldfits['size']==n)&(oldfits.seed<15)&(oldfits.mixer=='x')].simulation_s.sum())
 records=[];events=[]
 sleep=None if folder.name=='all_on' else float(folder.name.split('_')[1])
 for (n,seed),g in data[data.seed>=15].groupby(['size','seed']):
  inst=workload(n,seed,sleep,incumbent=int(g.incumbent.iloc[0]));model=cal[cal['size']==n]
  classicalrow=g[g.algorithm=='milp'].iloc[0];base_ready=float(classicalrow.local_s)
  for policy in ['always_classical','always_quantum_selected','deadline_only','energy_aware','quantum_with_backup','fixed_quantum_1024']:
   if policy=='fixed_quantum_1024' and folder.name!='all_on':continue
   start=time.perf_counter();eligible=model[(model.predicted_quality>=.9)&(model.predicted_delay<=inst.deadline_s)]
   quant=model[~model.algorithm.isin(['milp','local_search','heuristic'])&~model.algorithm.str.startswith('classical')];qe=eligible[eligible.algorithm.isin(quant.algorithm)]
   if policy=='always_classical':alg,shots='milp',0;reason='fixed classical reference'
   elif policy=='fixed_quantum_1024':alg,shots='qaoa_transfer',1024;reason='fixed stress comparator'
   elif policy=='energy_aware':
    choice=eligible.sort_values(['predicted_total','algorithm','shots']).iloc[0] if len(eligible) else model[(model.algorithm=='milp')].iloc[0];alg,shots=choice.algorithm,int(choice.shots);reason='training minimum total energy' if len(eligible) else 'no qualifying configuration; classical fallback'
   elif policy=='deadline_only':
    choice=qe.sort_values(['predicted_delay','algorithm','shots']).iloc[0] if len(qe) else model[model.algorithm=='milp'].iloc[0];alg,shots=choice.algorithm,int(choice.shots);reason='training quality and delay' if len(qe) else 'no qualifying quantum configuration'
   else:
    pool=qe if len(qe) else quant;choice=pool.sort_values(['predicted_total','algorithm','shots']).iloc[0];alg,shots=choice.algorithm,int(choice.shots);reason='training-selected quantum' if len(qe) else 'quantum forced; training targets fail'
   key=f'{n}:{seed}:{policy}';lookup=timings.get(key,time.perf_counter()-start);timings[key]=lookup
   chosen=g[(g.algorithm==alg)&(g.shots==shots)].copy();chosen['lookup_s']=lookup
   vals,ready,arrival=raw_evaluate(chosen)
   trained=policy not in ['always_classical','fixed_quantum_1024'];training_j=offline[n]*DEFAULT.local_w/DEFAULT.reuse if trained else 0.
   pred=model[(model.algorithm==alg)&(model.shots==shots)].iloc[0]
   costs=[];network=[];trans=[];useful=[];accepted=[];aq=[]
   for j,r in enumerate(chosen.itertuples()):
    cand=[(ready[j],r.solution,alg)]
    compute=vals['compute_j'][j]
    if trained:compute+=training_j-vals['offline_j'][j]
    if policy=='quantum_with_backup':
     cand.append((base_ready+lookup,int(classicalrow.solution),'milp_backup'));compute+=DEFAULT.local_w*base_ready
    applied=event_energy(inst,cand)
    costs.append(compute);network.append(applied['network_j']);trans.append(applied['transition_j']);accepted.append(applied['accepted']);aq.append((score(inst,applied['applied_action'])[1]-r.optimum_w)/(r.optimum_w+1)<=.05)
    selected_event=next(e for e in applied['events'] if e['solver']==alg)
    useful.append(bool(r.quality and selected_event['arrival_s']<=r.deadline_s and selected_event['arrival_s']<DEFAULT.horizon))
    for k,e in enumerate(applied['events']):events.append(dict(workload=folder.name,size=n,seed=seed,policy=policy,selected_solver=alg,shots=shots,solution_probability=r.weight,outcome_id=int(r.solution),controller_id=policy,incumbent_id=int(r.incumbent),encoding=r.encoding,depth=int(r.qaoa_depth),noise_id='ideal',transition_id=f'{folder.name}:{n}:{seed}:{policy}:{int(r.solution)}:{k}',event_id=k,predicted_quality=pred.predicted_quality,predicted_delay=pred.predicted_delay,predicted_total=pred.predicted_total,selection_reason=reason,**e))
   w=chosen.weight.to_numpy();total=np.average(np.array(costs)+network+trans,weights=w);baseline=float(test[(test['size']==n)&(test.seed==seed)&(test.algorithm=='milp')].total_j.iloc[0])
   records.append(dict(workload=folder.name,size=n,seed=seed,policy=policy,algorithm=alg,shots=shots,invoked_quantum=bool(chosen.is_quantum.iloc[0]),useful=np.average(useful,weights=w),accepted=np.average(accepted,weights=w),applied_quality=np.average(aq,weights=w),compute_j=np.average(costs,weights=w),network_j=np.average(network,weights=w),transition_j=np.average(trans,weights=w),total_j=total,excess_j=total-baseline,training_targets_pass=pred.predicted_quality>=.9 and pred.predicted_delay<=inst.deadline_s))
 timingfile.write_text(json.dumps(timings,indent=2));pd.DataFrame(events).to_csv(folder/'admission_events.csv',index=False);result=pd.DataFrame(records);result.to_csv(folder/'policies.csv',index=False)
 def policy_summary(frame):
  out=frame.groupby('policy').mean(numeric_only=True)
  for name,g in frame.groupby('policy'):
   rng=np.random.default_rng(20261016);strata=[]
   for _,sg in g.groupby('size'):
    x=sg.excess_j.to_numpy();strata.append(x[rng.integers(len(x),size=(20000,len(x)))].mean(axis=1))
   out.loc[name,'ci_low_j'],out.loc[name,'ci_high_j']=np.quantile(np.mean(strata,axis=0),[.025,.975])
  return out
 policy_summary(result).to_csv(folder/'policy_summary.csv')
 pd.concat([policy_summary(g).assign(size=n) for n,g in result.groupby('size')]).to_csv(folder/'policy_per_size.csv')
 return result


def main():
 global OUT
 from artifact_paths import output_directory
 import shutil

 ap=argparse.ArgumentParser();ap.add_argument('--skip-sweep',action='store_true');ap.add_argument('--input',type=Path,default=OUT);ap.add_argument('--output',type=Path,default=ROOT/'output/analysis/v12');a=ap.parse_args()
 OUT=output_directory(a.output)
 if OUT == a.input.resolve():raise ValueError('Input and output must differ')
 shutil.copytree(a.input,OUT,dirs_exist_ok=True)
 for name in ['all_on','sleep_0','sleep_10','sleep_50']:
  folder=OUT/name
  if not (folder/'complete.json').exists():raise RuntimeError('incomplete experiment '+name)
  data=prepare(folder);data.to_csv(folder/'outcomes.csv',index=False);pi=per_instance(data);pi.to_csv(folder/'primary_instances.csv',index=False);test=pi[pi.seed>=15]
  summary(test).to_csv(folder/'primary_summary.csv',index=False)
  bysize=[]
  for n,g in test.groupby('size'):
   s=summary(g);s['size']=n;bysize.append(s)
  pd.concat(bysize).to_csv(folder/'per_size.csv',index=False)
  diagnostics(data[data.seed>=15],test,folder);policies(data,pi,folder)
  testdata=data[data.seed>=15]
  if name=='all_on':
   if not a.skip_sweep:
    grid=[]
    for profile,lw,qw,reuse,h,mig in itertools.product(PROFILES,[10.,50.,100.],[10000.,20000.,40000.],[1,10,100,1000],[1.,10.,60.,300.,3600.],[0.,1.,10.,100.]):
     for boundary,eta in [('allocated',.01),('marginal',.001),('marginal',.01),('marginal',.1)]:grid.append(Scenario(profile=profile,local_w=lw,qpu_w=qw,reuse=reuse,horizon=h,migration_j=mig,boundary=boundary,marginal_fraction=eta))
    assert len(grid)==8640;sweep(testdata,grid,folder/'sensitivity.csv')
   overhead=[replace(DEFAULT,repetition_s=r,setup_s=s,utilization=u) for r,s,u in itertools.product([0,50e-6,200e-6],[0,.01,.1],[1,.5])]
   sweep(testdata,overhead,folder/'overheads.csv')
   migrations=[replace(DEFAULT,horizon=h,migration_j=m,migration_s=t) for h,m,t in itertools.product([1,10,60,300,3600],[0,1,10,100],[0,.01,.1])]
   sweep(testdata,migrations,folder/'migration_sensitivity.csv')
   # Original concentrated incumbent, with both acceptance semantics, using current timing profiles.
   stress=testdata.copy()
   for (n,seed),ix in stress.groupby(['size','seed']).groups.items():
    inst=workload(n,seed);old=decode(inst,inst.metadata['incumbent']);stress.loc[ix,'incumbent']=inst.metadata['incumbent'];stress.loc[ix,'incumbent_w']=score(inst,inst.metadata['incumbent'])[1]
    stress.loc[ix,'moves']=[0 if decode(inst,int(s)) is None else np.count_nonzero(decode(inst,int(s))!=old) for s in stress.loc[ix,'solution']]
   sc=[replace(DEFAULT,horizon=h,migration_j=m,benefit_gate=b) for h,m,b in itertools.product([1,10,60,300,3600],[0,1,10,100],[False,True])]
   sweep(stress,sc,folder/'concentrated_incumbent.csv')
  else:
   grid=[replace(DEFAULT,wake_j=w,wake_s=t,horizon=h,migration_j=m) for w,t,h,m in itertools.product([0,10,100],[0,.01,.1],[1,10,60,300,3600],[0,1,10,100])]
   sweep(testdata,grid,folder/'sleep_sensitivity.csv')
  print('analyzed',name,flush=True)
 (OUT/'analysis_manifest.json').write_text(json.dumps(dict(schema_version='1.2',primary=asdict(DEFAULT),main_grid=8640,statistical_unit='held-out instance',resamples=20000,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':main()
