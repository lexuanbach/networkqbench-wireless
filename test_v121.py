#!/usr/bin/env python3
"""Independent small-instance and accounting regression checks for revision 1.2.1."""
import json
from itertools import product
from dataclasses import replace
import numpy as np,pandas as pd
from revision_v121 import *


def main():
    count=0
    for sleep in [None,0.,10.,50.]:
        for n in [6,8,9,12]:
            for seed in [0,15,29]:
                inst=E.workload(n,seed,sleep);r=sparse_milp(inst)
                assert r['exact'] and r['feasible'];assert abs(r['power_w']-inst.metadata['power_w'][inst.feasible].min())<1e-7
                assert r['bound_w']<=r['power_w']+1e-7
                count+=1
    for n,f in [(3,2),(4,3)]:
        for seed in [0,2,7]:
            i,_=sparse_workload(n,f,seed);r=sparse_milp(i)
            scores=[E.score(i,assignment_state(a,n)) for a in product(range(n),repeat=f)]
            best=min(p for ok,p in scores if ok);assert r['exact'] and abs(best-r['power_w'])<1e-7;count+=1
    # Force a genuine solver time limit; missing solution must remain missing.
    i,_=sparse_workload(20,8,123);r=sparse_milp(i,1e-6)
    assert not r['exact'] and r['status']==1
    if not r['feasible']:assert r['solution'] is None and r['power_w'] is None
    frozen=pd.read_csv(FROZEN/'all_on/outcomes.csv');frozen=frozen[frozen.seed>=15]
    checks=0
    for scenario in [DEFAULT,replace(DEFAULT,boundary='marginal',marginal_fraction=.001),replace(DEFAULT,profile='nominal'),replace(DEFAULT,migration_s=.1,repetition_s=.0002,setup_s=.1)]:
        selected=frozen[(frozen.seed.isin([15,22,29]))&(frozen.shots.isin([0,64]))]
        vals,ready,_=raw_evaluate(selected,scenario)
        for j,r in enumerate(selected.itertuples()):
            inst=E.workload(r.size,r.seed,incumbent=int(r.incumbent));ev=E.event_energy(inst,[(ready[j],int(r.solution),r.algorithm)],scenario)
            for c in ['network_j','transition_j']:assert abs(ev[c]-vals[c][j])<1e-6
            assert abs(vals['total_j'][j]-vals['compute_j'][j]-ev['network_j']-ev['transition_j'])<1e-6
            checks+=1
    pi=pd.read_csv(RESULTS/'primary_instances.csv');stay=pi[pi.algorithm=='keep_incumbent']
    assert len(stay)==60 and (stay.compute_j==0).all() and (stay.transition_j==0).all()
    assert stay[['quality','timely','useful']].isna().all().all()
    summary=pd.read_csv(RESULTS/'primary_summary.csv');b=summary[summary.algorithm=='heuristic'].iloc[0]
    assert np.allclose(summary.greedy_excess_j,summary.excess_j-b.excess_j,atol=1e-8)
    assert (pi[pi.algorithm!='keep_incumbent'].useful<=pi[pi.algorithm!='keep_incumbent'].quality+1e-12).all()
    for r in json.loads((RESULTS/'aligned_fits.json').read_text()):assert r['seed']<15
    for r in json.loads((RESULTS/'aligned_transfer.json').read_text()):assert r['training_seeds']==list(range(15))
    assert pd.read_csv(RESULTS/'aligned_samples.csv').seed.min()==15
    for n in [6,8,9,12]:
        i=E.workload(n,0);f=fit_useful(i);p=E.onehot_probs(i,f['theta']);opt=i.metadata['power_w'][i.feasible].min();mask=i.feasible&((i.metadata['power_w']-opt)/(opt+1)<=.05)
        assert abs(p.sum()-1)<1e-12 and abs(f['objective_value']+p[mask].sum())<1e-10
    ledger=pd.read_csv(RESULTS/'decomposition.csv')
    for boundary in ['allocated','marginal']:
        sums=ledger[ledger.boundary==boundary].groupby('algorithm').difference_j.sum();s=pd.read_csv(RESULTS/('primary_summary.csv' if boundary=='allocated' else 'marginal_summary.csv'))
        for alg,value in sums.items():assert abs(value-s[(s.algorithm==alg)&(s.shots==64)].excess_j.iloc[0])<1e-7
    # Check exact interpolation against direct event evaluation at selected grid corners.
    grid=pd.read_csv(RESULTS/'component_grid.csv')
    for idx in [0,len(grid)//2,len(grid)-1]:
        r=grid.iloc[idx];sc=replace(DEFAULT,profile=r.profile,local_w=r.local_w,qpu_w=r.qpu_w,reuse=int(r.reuse),horizon=r.horizon,migration_j=r.migration_j,boundary=r.boundary,marginal_fraction=r.marginal_fraction)
        p=per_instance(frozen[frozen.algorithm.isin(['milp',r.algorithm])],sc);g=p[(p.algorithm==r.algorithm)&(p.shots==r.shots)]
        assert abs(g.excess_j.mean()-r.excess_j)<1e-6
    report=dict(sparse_exact_checks=count,event_replays=checks,timeout_check=True,zero_compute_baseline=True,training_split=True,paired_comparators=True,ledger_reconciliation=True,component_interpolation=True)
    (RESULTS/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
