#!/usr/bin/env python3
"""New experiments for the round-two revision; never overwrite v1.2 results."""
import argparse,json,time,platform,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from revision_v121 import *
from run_v12 import sparse_select
from run_energy_experiments import angle_mean


def training(out,objective):
    diagnosis=[];fits_out=[];transfer_out=[];sample_rows=[];resources=[];evaluated=[]
    for family in ['all_on','sleep_0','sleep_10','sleep_50']:
        folder=FROZEN/family;sleep=None if family=='all_on' else float(family.split('_')[1])
        oldfits=json.loads((folder/'fits.json').read_text());tr=json.loads((folder/'transfer.json').read_text())
        ins=pd.read_csv(folder/'instances.csv').set_index(['size','seed'])
        for n in [6,8,9,12]:
            fresh=[]
            for seed in range(30):
                ref=ins.loc[(n,seed)];inst=E.workload(n,seed,sleep,incumbent=int(ref.incumbent))
                _,states=E.assignment_basis(inst);opt=ref.optimum_w
                mask=inst.feasible & ((inst.metadata['power_w']-opt)/(opt+1)<=.05)
                uniform=float(mask[states].mean())
                for depth in [1,2]:
                    t=next(r for r in tr if r['size']==n and r['depth']==depth)
                    p=E.onehot_probs(inst,t['theta'])
                    fitted=None
                    if seed<15:
                        f=next(r for r in oldfits if r['size']==n and r['depth']==depth and r['seed']==seed)
                        fitted=float(E.onehot_probs(inst,f['theta'])[mask].sum())
                    diagnosis.append(dict(workload=family,size=n,seed=seed,split='train' if seed<15 else 'test',depth=depth,
                        uniform=uniform,fitted=fitted,transferred=float(p[mask].sum()),resultant=json.dumps(t['circular_resultant']),dispersion=json.dumps(t['circular_sd_radians'])))
                if seed<15:
                    f=fit_useful(inst,objective);fresh.append(f)
                    fits_out.append(dict(workload=family,size=n,seed=seed,split='train',**f))
            angles=np.array([x['theta'] for x in fresh]);periods=np.array([2*np.pi,np.pi])
            theta=[angle_mean(angles[:,k],periods[k]) for k in range(2)]
            resultant=abs(np.exp(2j*np.pi*angles/periods).mean(axis=0));offline=sum(x['simulation_s'] for x in fresh)
            transfer_out.append(dict(workload=family,size=n,depth=1,objective=objective,theta=theta,
                offline_s=offline,training_seeds=list(range(15)),circular_resultant=resultant.tolist()))
            for seed in range(15,30):
                ref=ins.loc[(n,seed)];start=time.perf_counter();inst=E.workload(n,seed,sleep,incumbent=int(ref.incumbent));prep=time.perf_counter()-start
                p=E.onehot_probs(inst,theta);opt=ref.optimum_w
                mask=inst.feasible & ((inst.metadata['power_w']-opt)/(opt+1)<=.05)
                start=time.perf_counter();q=E.circuit(inst,theta);build=time.perf_counter()-start
                _,res=E.compile_circuit(q);resources.append(dict(workload=family,size=n,seed=seed,preparation_s=prep,circuit_build_s=build,offline_s=offline,**res))
                for shots in SHOTS:
                    post=None
                    for rep in range(30):
                        # Same uniform random stream as the old structured/dense family.
                        rng=np.random.default_rng(2026101600+n*10000+seed*100+2*1000000+rep)
                        draw=rng.choice(len(p),shots,p=p)
                        if rep==0:
                            times=[]
                            for _ in range(3):
                                tick=time.perf_counter();sparse_select(inst,draw);times.append(time.perf_counter()-tick)
                            post=float(np.median(times))
                        valid=draw[inst.feasible[draw]];chosen=int(valid[np.argmin(inst.metadata['power_w'][valid])]) if len(valid) else int(draw[0])
                        ready=prep+build+res['compile_s']+post+.0125+shots*res['shot_s']
                        compute=50*(prep+build+res['compile_s']+post+.0025+offline/100)+20000*shots*res['shot_s']
                        row=candidate_record(inst,ref,chosen,ready,compute,'useful_mass_transfer',shots)
                        row.update(workload=family,repeat=rep,p_quality=float(p[mask].sum()),offline_j=50*offline/100)
                        sample_rows.append(row)
            print('training',family,n,flush=True)
    pd.DataFrame(diagnosis).to_csv(out/'training_diagnosis.csv',index=False)
    (out/'aligned_fits.json').write_text(json.dumps(fits_out,indent=2))
    (out/'aligned_transfer.json').write_text(json.dumps(transfer_out,indent=2))
    pd.DataFrame(resources).to_csv(out/'aligned_resources.csv',index=False)
    pd.DataFrame(sample_rows).to_csv(out/'aligned_samples.csv',index=False)


def demand(out,low,high):
    data=pd.read_csv(FROZEN/'all_on/outcomes.csv');data=data[data.seed>=15].copy()
    rows=[]
    for (n,seed),ix in data.groupby(['size','seed']).groups.items():
        inst=E.workload(n,seed);m=inst.metadata;rng=np.random.default_rng(202610160+n*100+seed)
        perturb=rng.uniform(low,high,len(m['demands']))
        for step in range(12):
            factor=2.**(-step) if step<11 else 0.
            prior=E.workload(n,seed,demands=m['demands']*(1+factor*perturb));r=E.solve_milp(prior)
            if r is not None:break
        state=int(r['solution']);assert E.score(inst,state)[0];m['incumbent']=state
        rows.append(dict(size=n,seed=seed,requested_min=low,requested_max=high,
            realized_min=float((factor*perturb).min()),realized_max=float((factor*perturb).max()),halvings=step,
            incumbent=state,incumbent_w=E.score(inst,state)[1],optimum_w=float(m['power_w'][inst.feasible].min()),prior_s=r['runtime_s']))
        old=E.decode(inst,state);data.loc[ix,'incumbent']=state;data.loc[ix,'incumbent_w']=E.score(inst,state)[1]
        for algorithm in ['milp','heuristic','local_search']:
            r=E.classical(inst,algorithm);sel=ix[data.loc[ix,'algorithm'].to_numpy()==algorithm]
            data.loc[sel,'solution']=r['solution'];data.loc[sel,'power_w']=r['power_w'];data.loc[sel,'local_s']=r['runtime_s']
            data.loc[sel,'runtime_s']=r['runtime_s'];q=(r['power_w']-rows[-1]['optimum_w'])/(rows[-1]['optimum_w']+1)<=.05
            data.loc[sel,'quality']=q;data.loc[sel,'p_quality']=float(q)
        data.loc[ix,'moves']=[0 if E.decode(inst,int(v)) is None else int(np.count_nonzero(E.decode(inst,int(v))!=old)) for v in data.loc[ix,'solution']]
    pi=per_instance(data);pi.to_csv(out/'demand_instances.csv',index=False);summary(pi).to_csv(out/'demand_summary.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'demand_incumbents.csv',index=False)
    data.to_csv(out/'demand_outcomes.csv',index=False)
    print('demand complete',flush=True)


def larger(out):
    fits=[];rows=[];resources=[]
    for n in [14,16]:
        fs=[]
        for seed in range(15):
            inst=E.workload(n,seed);fit=E.fit_onehot(inst,1);fs.append(fit);fits.append(dict(size=n,seed=seed,**fit))
        theta=[angle_mean(np.array([f['theta'][k] for f in fs]),[2*np.pi,np.pi][k]) for k in range(2)]
        for seed in range(15,30):
            inst=E.workload(n,seed);_,states=E.assignment_basis(inst);opt=inst.metadata['power_w'][inst.feasible].min()
            quality=inst.feasible & ((inst.metadata['power_w']-opt)/(opt+1)<=.05)
            p=float(quality[states].mean());pq=float(E.onehot_probs(inst,theta)[quality].sum())
            _,res=E.compile_circuit(E.circuit(inst,theta));resources.append(dict(size=n,seed=seed,**res))
            for shots in SHOTS:
                successes=[]
                for rep in range(30):
                    rng=np.random.default_rng(2026101600+n*10000+seed*100+1000000+rep)
                    draw=rng.choice(states,shots);successes.append(bool(quality[draw].any()))
                rows.append(dict(size=n,seed=seed,shots=shots,onehot_states=len(states),uniform_p_quality=p,
                    exact_batch_quality=1-(1-p)**shots,sampled_quality=np.mean(successes),structured_p_quality=pq))
        print('larger',n,flush=True)
    (out/'larger_fits.json').write_text(json.dumps(fits,indent=2));pd.DataFrame(rows).to_csv(out/'larger_quality.csv',index=False)
    pd.DataFrame(resources).to_csv(out/'larger_resources.csv',index=False)


def scaling(out,nodes,functions,seeds,limit):
    path=out/'scaling.csv';records=pd.read_csv(path).to_dict('records') if path.exists() else []
    done={(int(r['nodes']),int(r['functions']),int(r['seed'])) for r in records}
    for n in nodes:
        for f in functions:
            for seed in seeds:
                if (n,f,seed) in done:continue
                inst,generation=sparse_workload(n,f,seed)
                # Alternating solver order avoids one fixed cache/order advantage.
                if seed%2:g=E.classical(inst,'heuristic');r=sparse_milp(inst,limit)
                else:r=sparse_milp(inst,limit);g=E.classical(inst,'heuristic')
                record=dict(nodes=n,functions=f,seed=seed,generation_s=generation,**r,
                    greedy_solution=str(g['solution']),greedy_power_w=g['power_w'],greedy_s=g['runtime_s'],
                    milp_compute_j=50*r['runtime_s'],greedy_compute_j=50*g['runtime_s'],time_limit_s=limit)
                record['solution']=str(record['solution']) if record['solution'] is not None else None
                records.append(record);pd.DataFrame(records).to_csv(path,index=False)
                print('scaling',n,f,seed,'status',r['status'],'seconds',round(r['runtime_s'],2),flush=True)


def timing(out):
    ins=pd.read_csv(FROZEN/'all_on/instances.csv');ins=ins[ins.seed>=15]
    cases={}
    for ref in ins.to_dict('records'):
        i=E.workload(ref['size'],ref['seed'],incumbent=int(ref['incumbent']));cases[(i.n_vars,i.seed)]=(i,ref)
    # Warm all relevant code paths, outside recorded rounds.
    for i,ref in cases.values():
        E.classical(i,'milp');E.classical(i,'heuristic');_,states=E.assignment_basis(i);sparse_select(i,states[:min(64,len(states))])
    rows=[]
    jobs=[(n,seed,alg,s) for n,seed in cases for alg,s in [('milp',0),('heuristic',0),('classical_uniform_onehot',64),('classical_uniform_onehot',256)]]
    rng=np.random.default_rng(2026101701)
    for rep in range(10):
        for j in rng.permutation(len(jobs)):
            n,seed,alg,shots=jobs[j];i,ref=cases[n,seed]
            if shots:
                _,states=E.assignment_basis(i);p=np.zeros(1<<n);p[states]=1/len(states)
                # Same 30 outcome streams each round isolates host timing variation.
                samples=[];elapsed=[]
                for draw_rep in range(30):
                    stream=np.random.default_rng(2026101600+n*10000+seed*100+1000000+draw_rep)
                    start=time.perf_counter();draw=stream.choice(len(p),shots,p=p);sol=sparse_select(i,draw);t=time.perf_counter()-start
                    samples.append(candidate_record(i,ref,sol,t,50*t,alg,shots));elapsed.append(t)
                row={k:np.mean([x[k] for x in samples]) for k in ['quality','timely','useful','compute_j','network_j','transition_j','total_j']}
                row.update(size=n,seed=seed,algorithm=alg,shots=shots,runtime_s=np.mean(elapsed))
            else:
                r=E.classical(i,alg);row=candidate_record(i,ref,r['solution'],r['runtime_s'],50*r['runtime_s'],alg)
                row['runtime_s']=r['runtime_s']
            row['round']=rep;rows.append(row)
        pd.DataFrame(rows).to_csv(out/'timing.csv',index=False);print('timing round',rep,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=RESULTS)
    ap.add_argument('--stages',nargs='+',choices=['training','demand','larger','scaling','timing'],default=['training','demand','larger','scaling','timing'])
    ap.add_argument('--training-objective',choices=['useful_mass','expectation'],default='useful_mass')
    ap.add_argument('--demand-range',nargs=2,type=float,default=[.2,.5]);ap.add_argument('--servers',nargs='+',type=int,default=[20,50,100])
    ap.add_argument('--functions',nargs='+',type=int,default=[4,8]);ap.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    ap.add_argument('--time-limit',type=float,default=30.);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    assert 0<=a.demand_range[0]<=a.demand_range[1] and a.time_limit>0
    for stage in a.stages:
        start=time.perf_counter()
        if stage=='training':training(a.output,a.training_objective)
        elif stage=='demand':demand(a.output,*a.demand_range)
        elif stage=='larger':larger(a.output)
        elif stage=='scaling':scaling(a.output,a.servers,a.functions,a.seeds,a.time_limit)
        else:timing(a.output)
        manifest=dict(stage=stage,elapsed_s=time.perf_counter()-start,python=platform.python_version(),platform=platform.platform(),
            parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},training_seeds=list(range(15)),heldout_seeds=list(range(15,30)),
            evidence='local wall times measured; quantum quality simulated; all energy modeled',
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'revision_v121.py',ROOT/'energy_v12.py']})
        (a.output/(stage+'_manifest.json')).write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
