#!/usr/bin/env python3
"""Reproducible NetworkQBench-Wireless 1.2 experiments; never overwrite 1.1 data."""
from pathlib import Path
import argparse,hashlib,json,time,platform
import numpy as np
import pandas as pd
from energy_v12 import *
from networkqbench import qaoa_probabilities
from run_energy_experiments import angle_mean, DURATIONS

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'/'v1_2';OLD=ROOT/'results'/'sustainability'
WORKLOADS={'all_on':None,'sleep_0':0.,'sleep_10':10.,'sleep_50':50.}

def sparse_select(inst,draw):
    best=int(draw[0]);bestp=float('inf')
    for s in np.unique(draw):
        ok,p=score(inst,int(s))
        if ok and p<bestp:best,bestp=int(s),p
    return best


def run(name,out):
    folder=out/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():print('resume',name,flush=True);return
    sleep=WORKLOADS[name];oldfit=pd.read_csv(OLD/'fits.csv');oldt=json.loads((OLD/'transfer.json').read_text())
    instances={};refs=[];cls=[];fits=[];oldcir=pd.read_csv(OLD/'circuits.csv')
    for n in SIZES:
        for seed in range(30):
            start=time.perf_counter();i=workload(n,seed,sleep);prep=time.perf_counter()-start
            prev=previous_incumbent(i);i.metadata['incumbent']=prev['incumbent'];instances[n,seed]=i
            opt=float(i.metadata['power_w'][i.feasible].min());a,states=assignment_basis(i)
            co=int(np.count_nonzero(i.feasible & np.isclose(i.metadata['power_w'],opt,atol=1e-8)))
            ref=dict(workload=name,size=n,seed=seed,split='train' if seed<15 else 'test',incumbent=prev['incumbent'],incumbent_w=score(i,prev['incumbent'])[1],optimum_w=opt,
                     deadline_s=i.deadline_s,dense_preparation_s=prep,prior_demands=json.dumps(prev['prior_demands']),perturbation=prev['perturbation'],halvings=prev['halvings'],prior_s=prev['prior_s'],
                     cooptimal_states=co,onehot_states=len(states),uniform_optimum_probability=co/len(states))
            refs.append(ref)
            for alg in ['milp','local_search','heuristic']:
                r=classical(i,alg)
                if alg=='milp':assert abs(r['power_w']-opt)<1e-7
                cls.append({**ref,'algorithm':alg,**r})
            if seed<15:
                if name=='all_on':
                    r=oldfit[(oldfit['size']==n)&(oldfit.seed==seed)&(oldfit.mixer=='onehot')].iloc[0]
                    p1=dict(theta=[float(r.gamma),float(r.beta)],simulation_s=float(r.simulation_s),evals=int(r.evals),expectation=np.nan)
                else:p1=fit_onehot(i,1)
                p2=fit_onehot(i,2,p1['theta'])
                for depth,r in [(1,p1),(2,p2)]:fits.append(dict(workload=name,size=n,seed=seed,depth=depth,**r))
        print(name,'model and fits',n,flush=True)
    pd.DataFrame(refs).to_csv(folder/'instances.csv',index=False);pd.DataFrame(cls).to_csv(folder/'classical.csv',index=False)
    (folder/'fits.json').write_text(json.dumps(fits,indent=2,allow_nan=True))
    transferred={};train=[]
    for n in SIZES:
        for depth in [1,2]:
            g=[r for r in fits if r['size']==n and r['depth']==depth];angles=np.array([r['theta'] for r in g]);periods=np.array([2*np.pi]*depth+[np.pi]*depth)
            theta=[angle_mean(angles[:,k],periods[k]) for k in range(2*depth)]
            resultant=abs(np.exp(2j*np.pi*angles/periods).mean(axis=0))
            offline=sum(r['simulation_s'] for r in g)
            transferred[n,depth]=(theta,offline)
            train.append(dict(workload=name,size=n,depth=depth,theta=theta,training_seeds=list(range(15)),offline_s=offline,circular_resultant=resultant.tolist(),circular_sd_radians=np.sqrt(-2*np.log(np.maximum(resultant,1e-15))).tolist()))
    (folder/'transfer.json').write_text(json.dumps(train,indent=2))
    resources=[];samples=[];distributions={};audits=[];refs={(r['size'],r['seed']):r for r in refs}
    for n in SIZES:
        for seed in range(30):
            i=instances[n,seed];ref=refs[n,seed];m=i.metadata;a,states=assignment_basis(i)
            norm=_normalized_cost(i.costs) if False else None
            optimal=i.feasible & np.isclose(m['power_w'],ref['optimum_w'],atol=1e-8)
            quality=i.feasible & ((m['power_w']-ref['optimum_w'])/(ref['optimum_w']+1)<=.05)
            configs=[]
            for space in ['full','onehot']:
                p=np.ones(1<<n)/(1<<n) if space=='full' else np.zeros(1<<n)
                if space=='onehot':p[states]=1/len(states)
                for service in ['classical','quantum']:configs.append((service+'_uniform_'+space,p,[],space,'none',0,0.,service=='quantum',1))
            for depth in [1,2]:
                theta,off=transferred[n,depth];p=onehot_probs(i,theta)
                encodings=['dense','structured'] if depth==1 else ['structured']
                for enc in encodings:configs.append(('onehot_transfer' if enc=='dense' else ('onehot_structured' if depth==1 else 'onehot_p2'),p,theta,'onehot',enc,depth,off,True,1))
            if name=='all_on':
                t=next(r for r in oldt if r['size']==n and r['mixer']=='x');theta=[t['gamma'],t['beta']]
                configs.append(('qaoa_transfer',qaoa_probabilities(i.costs,n,*theta),theta,'x','dense',1,t['offline_simulation_s'],True,1))
                r=oldfit[(oldfit['size']==n)&(oldfit.seed==seed)&(oldfit.mixer=='x')].iloc[0];theta=[r.gamma,r.beta]
                configs.append(('qaoa_online',qaoa_probabilities(i.costs,n,*theta),theta,'x','dense',1,0.,True,int(r.evals)+1))
            for alg,p,theta,mixer,enc,depth,off,isq,evals in configs:
                distributions[f'n{n}_s{seed}_{alg}']=p
                res=dict(compile_s=0.,circuit_build_s=0.,shot_s=0.,cx=0,depth=0,physical_width=0,target='local')
                if isq:
                    start=time.perf_counter();q=circuit(i,theta,'onehot' if mixer=='onehot' else 'x',enc,depth);build=time.perf_counter()-start
                    comp,res=compile_circuit(q);res['circuit_build_s']=build
                resources.append(dict(workload=name,size=n,seed=seed,algorithm=alg,is_quantum=isq,qaoa_depth=depth,encoding=enc,mixer=mixer,theta=json.dumps(theta),offline_s=off,evals=evals,**res))
                # Distribution-family seed gives dense/structured and classical/quantum controls common draws.
                family=0 if 'uniform_full' in alg else 1 if 'uniform_onehot' in alg else 2 if alg in ['onehot_transfer','onehot_structured'] else 3 if alg=='onehot_p2' else 4 if alg=='qaoa_transfer' else 5
                for shots in SHOTS:
                    post_s=None
                    for rep in range(30):
                        rng=np.random.default_rng(2026101600+n*10000+seed*100+family*1000000+rep)
                        start=time.perf_counter();draw=rng.choice(len(p),shots,p=p);sample_s=time.perf_counter()-start
                        if rep==0:
                            times=[]
                            for _ in range(3):
                                start=time.perf_counter();direct=sparse_select(i,draw);times.append(time.perf_counter()-start)
                            post_s=float(np.median(times))
                        valid=draw[i.feasible[draw]];chosen=int(valid[np.argmin(m['power_w'][valid])]) if len(valid) else int(draw[0])
                        if rep==0:assert (score(i,direct)[0]==bool(i.feasible[chosen])) and (not i.feasible[chosen] or abs(score(i,direct)[1]-m['power_w'][chosen])<1e-8)
                        samples.append(dict(workload=name,size=n,seed=seed,algorithm=alg,shots=shots,repeat=rep,solution=chosen,feasible=bool(i.feasible[chosen]),quality=bool(quality[chosen]),optimal=bool(optimal[chosen]),power_w=float(m['power_w'][chosen]),p_quality=float(p[quality].sum()),p_optimal=float(p[optimal].sum()),p_feasible=float(p[i.feasible].sum()),postprocessing_s=post_s,sampling_s=sample_s if not isq else 0.))
            audit=audit_qubo(i)
            q=qubo_circuit(i);width=q.num_qubits;side=math.ceil(math.sqrt(width));rows=math.ceil(width/side)
            comp,resource=compile_circuit(q,f'grid:{rows}x{side}')
            audits.append(dict(workload=name,size=n,seed=seed,**audit,**resource))
        print(name,'circuits and samples',n,flush=True)
        # Checkpoints after each size are sufficient to inspect partial progress.
        pd.DataFrame(resources).to_csv(folder/'circuits.csv',index=False)
        pd.DataFrame(audits).to_csv(folder/'qubo_audit.csv',index=False)
    pd.DataFrame(samples).to_csv(folder/'samples.csv',index=False);np.savez_compressed(folder/'distributions.npz',**distributions)
    (folder/'complete.json').write_text(json.dumps(dict(workload=name,instances=len(instances),circuits=len(resources),sampling_batches=len(samples),milp_matches=len(instances),qubo_matches=len(audits),schema_version='1.2'),indent=2))
    print('completed',name,len(samples),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workloads',nargs='+',default=list(WORKLOADS));ap.add_argument('--output',type=Path,default=OUT);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    for name in args.workloads:run(name,args.output)
    manifest=dict(schema_version='1.2',study='NetworkQBench-Wireless',python=platform.python_version(),platform=platform.platform(),workloads=WORKLOADS,sizes=list(SIZES),training_seeds=list(range(15)),test_seeds=list(range(15,30)),shots=list(SHOTS),repeats=30,operation_durations_s=DURATIONS,elapsed_s=time.perf_counter()-start,
      local_power='10/50/100 W scenarios applied to measured phase times',energy_evidence='modeled_energy',quality_evidence='exact_simulated',timing='common sparse selection measured as median of three batches on repeat 0 per case/pipeline/budget; shared across repetitions',
      incumbent='prior optimum under seeded nonnegative demand perturbation; historical preparation outside current decision boundary',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'energy_v12.py']})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
