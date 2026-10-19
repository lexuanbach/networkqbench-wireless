#!/usr/bin/env python3
"""Measured round-four additions; historical inputs are read-only."""
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from round4 import *
from run_v12 import sparse_select


def transfer(a):
    started=datetime.now(timezone.utc).isoformat(); inputs=[]
    aligned=json.loads((a.previous/'aligned_transfer.json').read_text())
    aligned_fits=json.loads((a.previous/'aligned_fits.json').read_text())
    inputs += [a.previous/'aligned_transfer.json', a.previous/'aligned_fits.json']
    rng=np.random.default_rng(a.seed)
    # Warm up native numerical and compiler code outside measurements.
    warm=E.workload(6,0);E.onehot_probs(warm,[.3,.2]);E.compile_circuit(E.circuit(warm,[.3,.2]))
    for family in FAMILIES:
        sleep=None if family=='all_on' else float(family.split('_')[1]); folder=a.input/family
        refs=pd.read_csv(folder/'instances.csv').set_index(['size','seed'])
        raw=pd.read_csv(folder/'outcomes.csv'); raw=raw[(raw.algorithm=='onehot_structured')&(raw.seed>=15)]
        profiles=raw.drop_duplicates(['size','seed','shots']).set_index(['size','seed','shots'])
        oldfits=json.loads((folder/'fits.json').read_text()); oldtransfer=json.loads((folder/'transfer.json').read_text())
        inputs += [folder/n for n in ['instances.csv','outcomes.csv','fits.json','transfer.json']]
        for n in SIZES:
            dest=a.output/'transfer'/f'{family}_{n}';dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():continue
            models={}; fit_records=[]
            for objective in ['expectation','useful_mass']:
                fs=[r for r in (oldfits if objective=='expectation' else aligned_fits)
                    if r['size']==n and r.get('depth',1)==1 and r['seed']<15
                    and r.get('workload',family)==family]
                assert len(fs)==15
                direct=fit_family(n,sleep,objective,sum(r['evals'] for r in fs))
                direct.update(workload=family,size=n,method='direct'); fit_records.append(direct)
                models[f'direct_{objective}']=(direct['theta'],direct['offline_s'],'Round4Transfer')
                tr=next(r for r in (oldtransfer if objective=='expectation' else aligned)
                        if r['size']==n and r.get('depth',1)==1 and r.get('workload',family)==family)
                models[f'circular_{objective}']=(tr['theta'],sum(r['simulation_s'] for r in fs),
                                                'Primary' if objective=='expectation' else 'Follow-up')
            (dest/'fits.json').write_text(json.dumps(fit_records,indent=2)+'\n')
            rows=[]; resources=[]; probabilities=[]
            jobs=[(seed,method) for seed in range(15,30) for method in models];rng.shuffle(jobs)
            for seed,method in jobs:
                theta,offline,fit_campaign=models[method];ref=refs.loc[(n,seed)]
                tick=time.perf_counter();inst=E.workload(n,seed,sleep,incumbent=int(ref.incumbent));prep=time.perf_counter()-tick
                p=E.onehot_probs(inst,theta);opt=ref.optimum_w
                mask=inst.feasible & ((inst.metadata['power_w']-opt)/(opt+1)<=.05)
                exact=inst.feasible & np.isclose(inst.metadata['power_w'],opt)
                _,states=E.assignment_basis(inst)
                probabilities.append(dict(workload=family,size=n,seed=seed,method=method,p_quality=float(p[mask].sum()),
                                          p_optimal=float(p[exact].sum()),uniform=float(mask[states].mean())))
                tick=time.perf_counter();q=E.circuit(inst,theta);build=time.perf_counter()-tick
                _,res=E.compile_circuit(q)
                resources.append(dict(workload=family,size=n,seed=seed,method=method,preparation_s=prep,
                                      circuit_build_s=build,offline_s=offline,fitting_campaign=fit_campaign,**res))
                for shots in SHOTS:
                    post=None
                    for rep in range(30):
                        stream=np.random.default_rng(2026101600+n*10000+seed*100+2000000+rep)
                        draw=stream.choice(len(p),shots,p=p)
                        if rep==0:
                            times=[]
                            for _ in range(3):
                                tick=time.perf_counter();sparse_select(inst,draw);times.append(time.perf_counter()-tick)
                            post=float(np.median(times))
                        valid=draw[inst.feasible[draw]]
                        sol=int(valid[np.argmin(inst.metadata['power_w'][valid])]) if len(valid) else int(draw[0])
                        base=profiles.loc[(n,seed,shots)].to_dict()
                        ok,power=E.score(inst,sol);old=E.decode(inst,int(ref.incumbent));new=E.decode(inst,sol)
                        # Retain a bounded invalid-state power for the vectorized rejection path.
                        power=float(inst.metadata['power_w'][sol])
                        base.update(workload=family,size=n,seed=seed,method=method,algorithm=method,shots=shots,
                                    repeat=rep,solution=sol,feasible=ok,power_w=power,
                                    quality=bool(mask[sol]),optimal=bool(exact[sol]),p_quality=float(p[mask].sum()),
                                    p_optimal=float(p[exact].sum()),weight=1/30,
                                    moves=0 if new is None else int(np.count_nonzero(old!=new)),
                                    wakes=0 if new is None or sleep is None else len(set(new)-set(old)),
                                    dense_preparation_s=prep,circuit_build_s=build,compile_s=res['compile_s'],
                                    shot_s=res['shot_s'],postprocessing_s=post,offline_s=offline,
                                    fitting_campaign=fit_campaign,cx=res['cx'],depth=res['depth'])
                        rows.append(base)
            pd.DataFrame(rows).to_csv(dest/'outcomes.csv',index=False)
            pd.DataFrame(resources).to_csv(dest/'resources.csv',index=False)
            pd.DataFrame(probabilities).to_csv(dest/'probabilities.csv',index=False)
            (dest/'complete.json').write_text(json.dumps({'models':4,'heldout':15,'rows':len(rows),'finished_utc':datetime.now(timezone.utc).isoformat()}))
            print('transfer',family,n,'complete',flush=True)
    manifest(a.output/'transfer_manifest.json','Round4Transfer',started,inputs,dict(seed=a.seed,training_seeds=list(range(15)),test_seeds=list(range(15,30)),shots=list(SHOTS),repetitions=30,objectives=['expectation','useful_mass'],budget='historical objective-specific individual probability evaluations; 15 per family call',maxiter=40))


def scaling(a):
    started=datetime.now(timezone.utc).isoformat();rng=np.random.default_rng(a.seed+1)
    configs=sorted({(n,f) for n in [20,50,100] for f in [4,8]} |
                   {(n,int(n*r)) for n in [20,50,100] for r in [.2,.4]})
    jobs=[(n,f,s,d,alg) for n,f in configs for s in range(10) for d in a.deadlines for alg in ['milp','heuristic']]
    rng.shuffle(jobs);path=a.output/'deadline_runs.csv'
    records=pd.read_csv(path,dtype={'solution':str,'applied_action':str}).to_dict('records') if path.exists() else []
    done={(int(r['nodes']),int(r['functions']),int(r['seed']),float(r['deadline_s']),r['algorithm']) for r in records}
    warm,_=sparse_workload(20,4,999);deadline_milp(warm,deadline=.8);E.classical(warm,'heuristic')
    for n,f,seed,deadline,alg in jobs:
        if (n,f,seed,deadline,alg) in done:continue
        inst,gen=sparse_workload(n,f,seed);inst.deadline_s=deadline
        if alg=='milp':r=deadline_milp(inst,deadline=deadline)
        else:
            tick=time.perf_counter();g=E.classical(inst,'heuristic');ok,p=E.score(inst,g['solution'])
            r=dict(solution=g['solution'],power_w=p,feasible=ok,status=0,exact=False,message='one coordinate pass',
                   construction_s=0.,solve_s=g['runtime_s'],runtime_s=time.perf_counter()-tick,bound_w=None,mip_gap=None,solver_called=True,solver_budget_s=None)
        ledger=deploy_sparse(inst,r['solution'],r['runtime_s'],50*r['runtime_s'])
        row=dict(nodes=n,functions=f,seed=seed,deadline_s=deadline,algorithm=alg,generation_s=gen,
                 proportional= f in [int(.2*n),int(.4*n)],original=f in [4,8],
                 utilization=float(inst.metadata['demands'].sum()/inst.metadata['capacities'].sum()),
                 function_ratio=f/n,**(r | ledger))
        row['solution']=None if r['solution'] is None else str(r['solution'])
        row['overrun_s']=max(0,r['runtime_s']-deadline)
        records.append(row);pd.DataFrame(records).to_csv(path,index=False)
        if len(records)%10==0:print('deadline',len(records),'/',len(jobs),flush=True)
    # Thirty-second quality references for new configurations, outside online ledgers.
    path=a.output/'new_references.csv';refs=pd.read_csv(path,dtype={'solution':str}).to_dict('records') if path.exists() else []
    done={(int(r['nodes']),int(r['functions']),int(r['seed'])) for r in refs}
    old=pd.read_csv(a.previous/'scaling.csv');oldkeys=set(zip(old.nodes,old.functions,old.seed))
    for n,f in configs:
        for seed in range(10):
            if (n,f,seed) in oldkeys or (n,f,seed) in done:continue
            inst,gen=sparse_workload(n,f,seed);r=deadline_milp(inst,time_limit=a.reference_limit)
            r['solution']=None if r['solution'] is None else str(r['solution'])
            refs.append(dict(nodes=n,functions=f,seed=seed,generation_s=gen,**r))
            pd.DataFrame(refs).to_csv(path,index=False)
            print('reference',n,f,seed,'status',r['status'],'time',round(r['runtime_s'],2),flush=True)
    manifest(a.output/'scaling_manifest.json','Round4Deadline',started,[a.previous/'scaling.csv'],dict(seed=a.seed+1,seeds=list(range(10)),configurations=configs,deadlines=a.deadlines,reference_limit=a.reference_limit,deadline_boundary='model construction + solve + validation; physical inputs available; late computation charged',incumbent='concentrated feasible server 0'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'results/v1_2');ap.add_argument('--previous',type=Path,default=ROOT/'results/v1_2_1')
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT);ap.add_argument('--stages',nargs='+',choices=['transfer','scaling'],default=['transfer','scaling'])
    ap.add_argument('--seed',type=int,default=20261019);ap.add_argument('--deadlines',type=float,nargs='+',default=[.1,.4,.8]);ap.add_argument('--reference-limit',type=float,default=30.)
    a=ap.parse_args();assert all(d>0 for d in a.deadlines) and a.reference_limit>0
    assert a.output.resolve() not in [a.input.resolve(),a.previous.resolve(),(ROOT/'results/v1_2_2').resolve()]
    a.output.mkdir(parents=True,exist_ok=True)
    for stage in a.stages:globals()[stage](a)
if __name__=='__main__':main()
