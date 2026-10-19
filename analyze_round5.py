#!/usr/bin/env python3
"""Read-only round-five diagnostics and bounded six-variable ordering sensitivity."""
import argparse,json,hashlib,platform,time,shutil
from artifact_paths import output_directory
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
import numpy as np,pandas as pd
from scipy.optimize import minimize
import energy_v12 as E
from networkqbench import _normalized_cost
from analyze_v12 import per_instance,DEFAULT
from revision_v121 import paired_interval
ROOT=Path(__file__).resolve().parent

def ordering_probs(inst,theta,reverse=False):
 a,states=E.assignment_basis(inst);m=inst.metadata;v=np.ones(len(states),complex)/np.sqrt(len(states))
 v*=np.exp(-1j*theta[0]*_normalized_cost(inst.costs)[states])
 edges=list(range(m['nodes']))
 if reverse:edges.reverse()
 for f in range(m['functions']):
  for j in edges:
   left=np.flatnonzero(a[:,f]==j);b=a[left].copy();b[:,f]=(j+1)%m['nodes'];right=np.ravel_multi_index(b.T,(m['nodes'],)*m['functions'])
   x,y=v[left].copy(),v[right].copy();c=np.cos(theta[1]);s=-1j*np.sin(theta[1]);v[left]=c*x+s*y;v[right]=s*x+c*y
 out=np.zeros(1<<inst.n_vars);out[states]=abs(v)**2;return out/out.sum()

def diagnostic(out, inputs):
 fits=json.loads((inputs/'v1_2/all_on/fits.json').read_text());old={r['seed']:r for r in fits if r['size']==6 and r['depth']==1}
 records=[];angles={};started=datetime.now(timezone.utc).isoformat()
 for reverse in [False,True]:
  mode='reverse' if reverse else 'forward';ts=[]
  for seed in range(15):
   inst=E.workload(6,seed);cost=_normalized_cost(inst.costs);calls=0;t0=time.perf_counter()
   def obj(theta):
    nonlocal calls
    calls+=1;return float(ordering_probs(inst,theta,reverse)@cost)
   grid=[np.array([g,b]) for g in np.linspace(0,2*np.pi,7,endpoint=False) for b in np.linspace(0,np.pi,7,endpoint=False)]
   values=[obj(t) for t in grid];best=grid[int(np.argmin(values))];val=min(values)
   fit=minimize(obj,best,method='Nelder-Mead',options={'maxiter':40,'maxfev':old[seed]['evals']-calls,'xatol':1e-3,'fatol':1e-5})
   if fit.fun<val:best,val=fit.x,float(fit.fun)
   assert calls<=old[seed]['evals'];ts.append(best)
   records.append(dict(mode=mode,seed=seed,theta=best.tolist(),calls=calls,cap=old[seed]['evals'],iterations=int(fit.nit),reason=str(fit.message),simulation_s=time.perf_counter()-t0))
  ts=np.array(ts);period=np.array([2*np.pi,np.pi]);angles[mode]=((np.angle(np.exp(2j*np.pi*ts/period).mean(axis=0))%(2*np.pi))*period/(2*np.pi)).tolist()
 rows=[]
 for seed in range(15,30):
  inst=E.workload(6,seed);_,states=E.assignment_basis(inst);opt=inst.metadata['power_w'][inst.feasible].min();mask=inst.feasible & ((inst.metadata['power_w']-opt)/(opt+1)<=.05)
  for mode in angles:
   p=ordering_probs(inst,angles[mode],mode=='reverse')
   rows.append(dict(size=6,seed=seed,mode=mode,p_quality=p[mask].sum(),uniform=mask[states].mean()))
 pd.DataFrame(rows).to_csv(out/'ordering_instances.csv',index=False)
 (out/'ordering_fits.json').write_text(json.dumps(records,indent=2)+'\n')
 (out/'ordering_manifest.json').write_text(json.dumps(dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),campaign='Round5Ordering',training=list(range(15)),heldout=list(range(15,30)),angles=angles,objective='normalized expectation',depth=1,scope='distribution-only exploratory diagnostic; not a new energy or hardware comparison',budget='same 7x7 grid; max 40 iterations; each fit capped at historical per-instance actual evaluations',environment=dict(python=platform.python_version(),numpy=np.__version__)),indent=2)+'\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=ROOT/'results');ap.add_argument('--output',type=Path,default=ROOT/'output/analysis/round5');ap.add_argument('--campaign',type=Path,default=ROOT/'results/round5_20261019');ap.add_argument('--run-ordering',action='store_true');a=ap.parse_args();o=output_directory(a.output)
 if a.run_ordering:diagnostic(o,a.input)
 else:
  for name in ['ordering_instances.csv','ordering_fits.json','ordering_manifest.json']:
   shutil.copy2(a.campaign/name,o/name)
 raw=pd.read_csv(a.input/'v1_2/all_on/outcomes.csv');raw=raw[raw.seed>=15]
 cross=[]
 for watts,u in [(10000,1),(10000,.5),(18691.2,.5)]:
  pi=per_instance(raw,replace(DEFAULT,qpu_w=watts,utilization=u,repetition_s=.0002,setup_s=.1));g=pi[(pi.algorithm=='onehot_structured')&(pi.shots==64)];lo,hi=paired_interval(g,'excess_j');cross.append(dict(qpu_w=watts,utilization=u,useful=g.useful.mean(),compute_j=g.compute_j.mean(),excess_j=g.excess_j.mean(),low=lo,high=hi))
 pd.DataFrame(cross).to_csv(o/'crossed_overheads.csv',index=False)
 t=pd.read_csv(a.input/'v1_2_1/training_diagnosis.csv');t=t[(t.workload=='all_on')&(t.split=='test')&(t.depth==1)]
 t.groupby('size').agg(uniform_mean=('uniform','mean'),uniform_median=('uniform','median'),transfer_mean=('transferred','mean'),transfer_median=('transferred','median')).reset_index().to_csv(o/'test_medians.csv',index=False)
 l=pd.read_csv(a.input/'v1_2_1/larger_quality.csv').drop_duplicates(['size','seed']);res=pd.read_csv(a.input/'v1_2_1/larger_resources.csv');l=l.merge(res,on=['size','seed'],validate='one_to_one')
 for name,col in [('uniform','uniform_p_quality'),('structured','structured_p_quality')]:
  p=l[col];l[name+'_required']=np.where(p==0,np.inf,np.where(p>=1,1,np.ceil(np.log(.1)/np.log1p(-p))))
 l['allocated_service_j']=l.structured_required*l.shot_s*20000;l.to_csv(o/'larger_requirements.csv',index=False)
 s=l.groupby('size').agg(q_median=('structured_required','median'),q_max=('structured_required','max'),u_median=('uniform_required','median'),u_max=('uniform_required','max'),service_median_j=('allocated_service_j','median'),service_max_j=('allocated_service_j','max')).reset_index();s.to_csv(o/'larger_requirements_summary.csv',index=False)
 timing=pd.read_csv(a.input/'v1_2_1/timing.csv');rt=timing[timing.algorithm=='milp'].groupby('round').compute_j.mean()
 largest=raw[raw.algorithm=='milp'].sort_values('local_s').iloc[-1]
 d=pd.read_csv(a.input/'round4_20261019/deadline_instances.csv');base=d[d.deadline_s==.8];original=base[(base.algorithm=='heuristic')&(base.original)]
 h=pd.read_csv(a.input/'round4_20261019/deadline_summary.csv');winning=h[(h.algorithm=='milp')&(h.minus_greedy_j<0)]
 # Current independently returned optimum: load is demand, utilization is demand/capacity.
 loadrows=[]
 for n in [6,8,9,12]:
  for seed in range(15,30):
   inst=E.workload(n,seed);r=raw[(raw.algorithm=='milp')&(raw['size']==n)&(raw.seed==seed)].iloc[0];x=E.decode(inst,int(r.solution));loads=np.bincount(x,weights=inst.metadata['demands'],minlength=inst.metadata['nodes']);util=loads/inst.metadata['capacities'];loadrows.append(dict(size=n,seed=seed,server0_max_demand=bool(loads[0]>=loads.max()-1e-10),server0_max_utilization=bool(util[0]>=util.max()-1e-10)))
 pd.DataFrame(loadrows).to_csv(o/'server0_load.csv',index=False)
 diag=dict(milp_round_mean_min_j=float(rt.min()),milp_round_mean_max_j=float(rt.max()),primary_max_j=float(largest.local_s*50),primary_max_size=int(largest['size']),primary_max_seed=int(largest.seed),original_scaling_mean_utilization=float(original.utilization.mean()),applied_by_servers={str(int(k)):int(v) for k,v in original.groupby('nodes').accepted.sum().items()},winning_deadline_cells=winning[['nodes','functions','deadline_s','minus_greedy_j','greedy_low','greedy_high']].to_dict('records'),server0_max_demand=sum(r['server0_max_demand'] for r in loadrows),server0_max_utilization=sum(r['server0_max_utilization'] for r in loadrows))
 (o/'diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n')
 inputs=[a.input/'v1_2/all_on/outcomes.csv',a.input/'v1_2_1/larger_quality.csv',a.input/'v1_2_1/larger_resources.csv',a.input/'v1_2_1/training_diagnosis.csv',a.input/'v1_2_1/timing.csv',a.input/'round4_20261019/deadline_instances.csv',a.input/'round4_20261019/deadline_summary.csv']
 (o/'analysis_manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={str(p.relative_to(a.input)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},bootstrap_resamples=20000,bootstrap_seed=20261017,assumptions='unchanged Primary except specified crossed overhead/power/utilization',generated_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
if __name__=='__main__':main()
