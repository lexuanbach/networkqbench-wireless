"""Round-two additions. Frozen schema-1.2 inputs are read only.

Sparse placement uses the same physical constraints without a dense state oracle.
All watts and joules remain models; perf_counter durations are local measurements.
"""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json, time, hashlib, platform
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint, minimize
from scipy.sparse import coo_matrix
import energy_v12 as E
from energy_model import WATTS, assignment_state, server_power, SHOTS
from networkqbench import _normalized_cost
from analyze_v12 import DEFAULT, per_instance, summary, raw_evaluate, KEY, METRICS

ROOT=Path(__file__).resolve().parent
FROZEN=ROOT/'results/v1_2'
RESULTS=ROOT/'results/v1_2_1'

def paired_interval(frame, column):
    rng=np.random.default_rng(20261017)
    boot=[]
    for _,g in frame.groupby('size',sort=True):
        x=g[column].to_numpy(float)
        boot.append(x[rng.integers(len(x),size=(20000,len(x)))].mean(axis=1))
    return np.quantile(np.mean(boot,axis=0),[.025,.975])

def sparse_workload(nodes, functions, seed):
    start=time.perf_counter();rng=np.random.default_rng(2026101700+seed+1000*nodes+functions)
    demands=rng.integers(1,4,size=functions).astype(float)
    cb=max(int(np.ceil(demands.sum()/nodes)),int(demands.max()))
    capacities=rng.integers(cb,cb+3,size=nodes).astype(float)
    capacities[0]=max(capacities[0],demands.sum())
    latency=rng.uniform(.5,3.,size=(nodes,nodes));np.fill_diagonal(latency,0.)
    m=dict(functions=functions,nodes=nodes,demands=demands,capacities=capacities,
           latency=latency,sleep_w=None,incumbent=assignment_state(np.zeros(functions,int),nodes))
    inst=SimpleNamespace(metadata=m,n_vars=nodes*functions,seed=seed,deadline_s=.8)
    return inst,time.perf_counter()-start


def sparse_milp(inst,time_limit=30.):
    start=time.perf_counter();m=inst.metadata;f,n=m['functions'],m['nodes']
    nx=f*n;nw=(f-1)*n*n;z0=nx+nw;t0=z0+10*n;count=t0+10*n
    c=np.zeros(count);integ=np.zeros(count);integ[:nx]=1;integ[z0:t0]=1
    row_ix=[];col_ix=[];values=[];lower=[];upper=[]
    def add(terms,lo,hi):
        k=len(lower)
        for j,v in terms:
            if v:row_ix.append(k);col_ix.append(j);values.append(v)
        lower.append(lo);upper.append(hi)
    for k in range(f):add([(k*n+j,1) for j in range(n)],1,1)
    for j in range(n):
        zs=[(z0+j*10+k,1) for k in range(10)]
        if m['sleep_w'] is None:add(zs,1,1)
        else:
            add(zs,0,1)
            for k in range(f):add([(k*n+j,1)]+[(z,-v) for z,v in zs],-np.inf,0)
            add(zs+[(k*n+j,-1) for k in range(f)],-np.inf,0)
        terms=[(k*n+j,float(m['demands'][k])/m['capacities'][j]) for k in range(f)]
        terms += [(z0+j*10+k,-k/10) for k in range(10)] + [(t0+j*10+k,-.1) for k in range(10)]
        add(terms,0,0)
        for k in range(10):
            z=z0+j*10+k;t=t0+j*10+k;add([(t,1),(z,-1)],-np.inf,0)
            c[z]=WATTS[k]-(m['sleep_w'] or 0);c[t]=WATTS[k+1]-WATTS[k]
    lat=[]
    for k in range(f-1):
        for i in range(n):
            for j in range(n):
                w=nx+k*n*n+i*n+j;a=k*n+i;b=(k+1)*n+j
                add([(w,1),(a,-1)],-np.inf,0);add([(w,1),(b,-1)],-np.inf,0)
                add([(w,1),(a,-1),(b,-1)],-1,np.inf);lat.append((w,float(m['latency'][i,j])))
    add(lat,-np.inf,4)
    matrix=coo_matrix((values,(row_ix,col_ix)),shape=(len(lower),count)).tocsc()
    build=time.perf_counter()-start;tick=time.perf_counter()
    result=milp(c,integrality=integ,bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix,lower,upper),
                options={'mip_rel_gap':0.,'time_limit':time_limit})
    solve=time.perf_counter()-tick
    state=None;valid=False;power=None
    if result.x is not None:
        state=assignment_state(np.rint(result.x[:nx]).reshape(f,n).argmax(axis=1),n)
        valid,power=E.score(inst,state)
        if not valid:state=None;power=None
    offset=n*(m['sleep_w'] or 0.)
    bound=getattr(result,'mip_dual_bound',None)
    return dict(solution=state,power_w=power,feasible=bool(valid),status=int(result.status),
                exact=bool(result.success and valid),message=result.message,
                mip_gap=getattr(result,'mip_gap',None),bound_w=None if bound is None else float(bound)+offset,
                construction_s=build,solve_s=solve,runtime_s=time.perf_counter()-start,
                variables=count,constraints=len(lower),nonzeros=matrix.nnz)


def fit_useful(inst,objective='useful_mass'):
    start=time.perf_counter();calls=0;cost=_normalized_cost(inst.costs)
    opt=float(inst.metadata['power_w'][inst.feasible].min())
    useful=inst.feasible & ((inst.metadata['power_w']-opt)/(opt+1)<=.05)
    def obj(theta):
        nonlocal calls
        calls+=1;p=E.onehot_probs(inst,theta)
        return -float(p[useful].sum()) if objective=='useful_mass' else float(p@cost)
    starts=[np.array([g,b]) for g in np.linspace(0,2*np.pi,7,endpoint=False) for b in np.linspace(0,np.pi,7,endpoint=False)]
    best=min(starts,key=obj);value=obj(best)
    r=minimize(obj,best,method='Nelder-Mead',options={'maxiter':40,'xatol':1e-3,'fatol':1e-5})
    if r.fun<value:best,value=r.x,r.fun
    return dict(theta=np.asarray(best).tolist(),evals=calls,simulation_s=time.perf_counter()-start,objective=objective,objective_value=value)


def candidate_record(inst,ref,solution,ready,compute,algorithm,shots=0,scenario=DEFAULT):
    ok,p=E.score(inst,int(solution));opt=ref['optimum_w'];quality=ok and (p-opt)/(opt+1)<=.05
    event=E.event_energy(inst,[(ready,int(solution),algorithm)],scenario)
    timely=event['events'][0]['arrival_s']<=inst.deadline_s and event['events'][0]['arrival_s']<scenario.horizon
    return dict(size=inst.n_vars,seed=inst.seed,algorithm=algorithm,shots=shots,solution=int(solution),
                quality=float(quality),timely=float(timely),useful=float(quality and timely),
                compute_j=compute,network_j=event['network_j'],transition_j=event['transition_j'],
                total_j=compute+event['network_j']+event['transition_j'])
