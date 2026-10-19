"""NetworkQBench-Wireless 1.2: physical workloads, encodings and action events.

The schema is additive. All electrical quantities are modeled; measured host
wall times and simulation times are recorded separately by the experiment runner.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from itertools import product
import math
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint, minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, StatePreparation
from qiskit.transpiler import CouplingMap
from networkqbench import Instance, bit_table, _normalized_cost
from energy_model import energy_placement, server_power, WATTS, SIZES, SHOTS, assignment_state
from run_energy_experiments import critical_path

SCHEMA_VERSION='1.2'

@dataclass(frozen=True)
class Scenario:
    profile: str='optimistic'
    local_w: float=50.
    qpu_w: float=20000.
    reuse: int=100
    horizon: float=60.
    migration_j: float=10.
    boundary: str='allocated'
    marginal_fraction: float=.01
    utilization: float=1.
    repetition_s: float=0.
    setup_s: float=0.
    migration_s: float=0.
    wake_j: float=0.
    wake_s: float=0.
    benefit_gate: bool=True


def workload(size, seed, sleep_w=None, demands=None, incumbent=None):
    b=energy_placement(size,seed);m=dict(b.metadata)
    if demands is not None:m['demands']=np.asarray(demands,float)
    m['sleep_w']=sleep_w
    bits=bit_table(size).reshape(-1,m['functions'],m['nodes'])
    loads=(bits*m['demands'][None,:,None]).sum(axis=1)
    exact=(bits.sum(axis=2)==1).all(axis=1)
    feasible=exact & (loads<=m['capacities']+1e-10).all(axis=1) & (m['latency_ms']<=4+1e-10)
    pw=server_power(np.clip(loads/m['capacities'],0,1))
    if sleep_w is not None:pw=np.where(loads>1e-12,pw,sleep_w)
    powers=pw.sum(axis=1)
    low=m['nodes']*(WATTS[0] if sleep_w is None else sleep_w)
    high=m['nodes']*WATTS[-1]
    costs=(powers-low)/(high-low)+2*(~feasible)
    m.update(power_w=powers,loads=loads,utilizations=loads/m['capacities'],sleep_w=sleep_w)
    if incumbent is not None:m['incumbent']=int(incumbent)
    return replace(b,costs=costs,feasible=feasible,metadata=m)


def decode(inst,state):
    m=inst.metadata;bits=[(int(state)>>(k*m['nodes']))&((1<<m['nodes'])-1) for k in range(m['functions'])]
    if any(x==0 or x&(x-1) for x in bits):return None
    return np.array([x.bit_length()-1 for x in bits],int)


def score(inst,state):
    """Sparse physical evaluator used identically by all sample selectors."""
    a=decode(inst,state);m=inst.metadata
    if a is None:return False,math.inf
    load=np.bincount(a,weights=m['demands'],minlength=m['nodes'])
    feasible=bool(np.all(load<=m['capacities']+1e-10) and sum(m['latency'][i,j] for i,j in zip(a[:-1],a[1:]))<=4+1e-10)
    p=server_power(np.clip(load/m['capacities'],0,1))
    if m['sleep_w'] is not None:p=np.where(load>1e-12,p,m['sleep_w'])
    return feasible,float(p.sum())


def solve_milp(inst):
    start=time.perf_counter();m=inst.metadata;f,n=m['functions'],m['nodes']
    nx=f*n;nw=(f-1)*n*n;z0=nx+nw;t0=z0+10*n;count=t0+10*n
    c=np.zeros(count);integ=np.zeros(count);integ[:nx]=1;integ[z0:t0]=1
    rows=[];lower=[];upper=[]
    def add(terms,lo,hi):
        row=np.zeros(count)
        for j,v in terms:row[j]+=v
        rows.append(row);lower.append(lo);upper.append(hi)
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
    result=milp(c,integrality=integ,bounds=Bounds(0,1),constraints=LinearConstraint(np.asarray(rows),lower,upper),options={'mip_rel_gap':0})
    elapsed=time.perf_counter()-start
    if not result.success:return None
    state=assignment_state(np.rint(result.x[:nx]).reshape(f,n).argmax(axis=1),n)
    valid,p=score(inst,state)
    assert valid
    return dict(solution=state,power_w=p,runtime_s=elapsed,mip_gap=float(result.mip_gap))


def previous_incumbent(inst):
    m=inst.metadata;rng=np.random.default_rng(202610160+inst.n_vars*100+inst.seed)
    perturb=rng.uniform(0,.2,len(m['demands']))
    elapsed=0.
    for step in range(12):
        factor=2.**(-step) if step<11 else 0.
        demand=m['demands']*(1+factor*perturb)
        prior=workload(inst.n_vars,inst.seed,m['sleep_w'],demand)
        r=solve_milp(prior)
        if r is not None:
            assert score(inst,r['solution'])[0]
            return dict(incumbent=r['solution'],prior_demands=demand.tolist(),perturbation=float((factor*perturb).max()),halvings=step,prior_s=r['runtime_s'])
    raise AssertionError('zero perturbation must be feasible')


def classical(inst,method):
    if method=='milp':return solve_milp(inst)
    start=time.perf_counter();m=inst.metadata;f,n=m['functions'],m['nodes']
    initial=decode(inst,m['incumbent']);rng=np.random.default_rng(73000+inst.seed+100*inst.n_vars)
    starts=[initial] if method=='heuristic' else [initial]+[rng.integers(n,size=f) for _ in range(7)]
    best=initial.copy();bestp=score(inst,assignment_state(best,n))[1]
    for a in starts:
        a=a.copy()
        if not score(inst,assignment_state(a,n))[0]:a=initial.copy()
        for _ in range(1 if method=='heuristic' else 10):
            changed=False
            for k in range(f):
                options=[]
                for j in range(n):
                    b=a.copy();b[k]=j;ok,p=score(inst,assignment_state(b,n))
                    if ok:options.append((p,j))
                _,j=min(options);changed |= j!=a[k];a[k]=j
            if not changed:break
        p=score(inst,assignment_state(a,n))[1]
        if p<bestp:best,bestp=a.copy(),p
    return dict(solution=assignment_state(best,n),power_w=bestp,runtime_s=time.perf_counter()-start)


def assignment_basis(inst):
    m=inst.metadata;a=np.array(list(product(range(m['nodes']),repeat=m['functions'])),int)
    states=np.array([assignment_state(x,m['nodes']) for x in a],int)
    return a,states


def onehot_probs(inst,theta):
    a,states=assignment_basis(inst);m=inst.metadata;p=len(theta)//2
    v=np.ones(len(states),complex)/np.sqrt(len(states));cost=_normalized_cost(inst.costs)[states]
    # Same ordered first-order product of ring XY rotations as the circuit.
    for layer in range(p):
        v*=np.exp(-1j*theta[layer]*cost)
        for k in range(m['functions']):
            for j in range(m['nodes']):
                other=(j+1)%m['nodes'];left=np.where(a[:,k]==j)[0]
                b=a[left].copy();b[:,k]=other
                right=np.ravel_multi_index(b.T,(m['nodes'],)*m['functions'])
                x,y=v[left].copy(),v[right].copy();c=np.cos(theta[p+layer]);s=-1j*np.sin(theta[p+layer])
                v[left]=c*x+s*y;v[right]=s*x+c*y
    out=np.zeros(1<<inst.n_vars);out[states]=abs(v)**2
    return out/out.sum()


def fit_onehot(inst,depth,initial=None):
    start=time.perf_counter();calls=0;cost=_normalized_cost(inst.costs)
    def obj(x):
        nonlocal calls
        calls+=1;return float(onehot_probs(inst,x)@cost)
    rng=np.random.default_rng(20261016+inst.n_vars*100+inst.seed+depth*10000)
    if depth==1:starts=[np.array([g,b]) for g in np.linspace(0,2*np.pi,7,endpoint=False) for b in np.linspace(0,np.pi,7,endpoint=False)];starts=[min(starts,key=obj)]
    else:
        g,b=initial if initial is not None else (1.,.5)
        starts=[np.array([g,0,b,0]),np.array([g/2,g/2,b/2,b/2])]
        starts += [np.r_[rng.uniform(0,2*np.pi,2),rng.uniform(0,np.pi,2)] for _ in range(2)]
    best=min(starts,key=obj);value=obj(best)
    for x in starts:
        r=minimize(obj,x,method='Nelder-Mead',options={'maxiter':40 if depth==1 else 60,'xatol':1e-3,'fatol':1e-5})
        if r.fun<value:best,value=r.x,r.fun
    return dict(theta=np.asarray(best).tolist(),evals=calls,simulation_s=time.perf_counter()-start,expectation=value)


def circuit(inst,theta,mixer='onehot',encoding='structured',depth=None):
    m=inst.metadata;n=inst.n_vars;q=QuantumCircuit(n);q.reset(range(n))
    if mixer=='onehot':
        v=np.zeros(1<<m['nodes'],complex)
        for k in range(m['nodes']):v[1<<k]=1/np.sqrt(m['nodes'])
        for f in range(m['functions']):q.append(StatePreparation(v),range(f*m['nodes'],(f+1)*m['nodes']))
    else:q.h(range(n))
    p=len(theta)//2 if depth is None else depth
    norm=_normalized_cost(inst.costs)
    for layer in range(p):
        if encoding=='dense':q.append(DiagonalGate(np.exp(-1j*theta[layer]*norm)),range(n))
        else:
            a,states=assignment_basis(inst)
            for assign,state in zip(a,states):
                qs=[f*m['nodes']+int(j) for f,j in enumerate(assign)]
                q.mcp(-float(theta[layer]*norm[state]),qs[:-1],qs[-1])
        if mixer=='onehot':
            for f in range(m['functions']):
                for k in range(m['nodes']):
                    i=f*m['nodes']+k;j=f*m['nodes']+(k+1)%m['nodes'];q.rxx(theta[p+layer],i,j);q.ryy(theta[p+layer],i,j)
        else:q.rx(2*theta[p+layer],range(n))
    q.measure_all();return q


def compile_circuit(q,target='grid4x4'):
    start=time.perf_counter();kwargs={}
    if target=='grid4x4':kwargs['coupling_map']=CouplingMap.from_grid(4,4)
    elif target.startswith('grid:'):
        a,b=map(int,target.split(':')[1].split('x'));kwargs['coupling_map']=CouplingMap.from_grid(a,b)
    comp=transpile(q,basis_gates=['rz','sx','x','cx'],optimization_level=2,seed_transpiler=20261015,**kwargs)
    return comp,dict(compile_s=time.perf_counter()-start,shot_s=critical_path(comp),cx=int(comp.count_ops().get('cx',0)),depth=comp.depth(),physical_width=comp.num_qubits,target=target)


def qubo(inst):
    """Exact feasible-optimum QUBO; not equivalent dynamics outside one-hot space."""
    m=inst.metadata;f,n=m['functions'],m['nodes'];a,states=assignment_basis(inst)
    coeff=_normalized_cost(inst.costs)[states];penalty=1+float(abs(coeff).sum())
    nq=inst.n_vars+(n*n if f==3 else 0);linear=np.zeros(nq);pairs={};constant=0.
    def term(i,j,c):
        if i==j:linear[i]+=c
        else:
            key=tuple(sorted((i,j)));pairs[key]=pairs.get(key,0)+c
    for k in range(f):
        constant+=penalty
        for i in range(n):
            linear[k*n+i]-=penalty
            for j in range(i+1,n):term(k*n+i,k*n+j,2*penalty)
    for row,c in zip(a,coeff):
        if f==2:term(int(row[0]),n+int(row[1]),float(c))
        else:term(inst.n_vars+int(row[0])*n+int(row[1]),2*n+int(row[2]),float(c))
    if f==3:
        for i in range(n):
            for j in range(n):
                x,y,z=i,n+j,inst.n_vars+i*n+j
                term(x,y,penalty);term(x,z,-2*penalty);term(y,z,-2*penalty);linear[z]+=3*penalty
    return linear,pairs,constant,penalty


def qubo_circuit(inst,gamma=1.):
    linear,pairs,constant,penalty=qubo(inst);q=QuantumCircuit(len(linear));q.reset(range(len(linear)));q.h(range(len(linear)))
    for i,c in enumerate(linear):
        if abs(c)>1e-12:q.p(-gamma*c,i)
    for (i,j),c in pairs.items():
        if abs(c)>1e-12:q.cp(-gamma*c,i,j)
    q.rx(.5,range(len(linear)));q.measure_all()
    return q


def audit_qubo(inst):
    linear,pairs,const,penalty=qubo(inst);n=inst.n_vars;m=inst.metadata
    x=bit_table(n);values=np.full(len(x),const)+x@linear[:n]
    for (i,j),c in pairs.items():
        if j<n:values+=c*x[:,i]*x[:,j]
    for z in range(n,len(linear)):
        cz=np.full(len(x),linear[z])
        for (i,j),c in pairs.items():
            if j==z and i<n:cz+=c*x[:,i]
            elif i==z and j<n:cz+=c*x[:,j]
        values+=np.minimum(0,cz)
    a,states=assignment_basis(inst);norm=_normalized_cost(inst.costs)
    assert np.allclose(values[states],norm[states],atol=1e-9)
    best=np.flatnonzero(np.isclose(values,values.min(),atol=1e-9))
    assert inst.feasible[best].all()
    optimum=float(inst.metadata['power_w'][inst.feasible].min())
    assert np.allclose(inst.metadata['power_w'][best],optimum)
    return dict(logical_qubits=n,auxiliary_qubits=len(linear)-n,qubo_qubits=len(linear),quadratic_terms=len(pairs),penalty=penalty,verified=True)


def event_energy(inst,candidates,scenario=Scenario(),incumbent=None):
    """Candidates: (ready time, solution, solver); charge computation externally.

    Processing order is ready-time then solver name. A migration in progress
    serializes a later action. Wake energy is incremental above incumbent draw.
    """
    m=inst.metadata;state=int(m['incumbent'] if incumbent is None else incumbent)
    H=scenario.horizon;clock=0.;network=0.;transition=0.;events=[];segments=[]
    optimum=float(m['power_w'][inst.feasible].min());start_state=state
    for ready,solution,solver in sorted(candidates,key=lambda x:(x[0],x[2])):
        solution=int(solution);old=decode(inst,state);new=decode(inst,solution)
        ok,pnew=score(inst,solution);pold=score(inst,state)[1]
        reason='accepted';moved=0;wake=0;arrival=max(float(ready),clock)
        if not ok:reason='infeasible'
        else:
            moved=int(np.count_nonzero(old!=new))
            if m['sleep_w'] is not None:wake=len(set(new)-set(old))
            arrival+=moved*scenario.migration_s+(scenario.wake_s if wake else 0.)
            cost=moved*scenario.migration_j+wake*scenario.wake_j
            if arrival>inst.deadline_s or arrival>=H:reason='late'
            elif scenario.benefit_gate and (H-arrival)*(pold-pnew)<=cost+1e-12:reason='no_benefit'
        if reason=='accepted':
            duration=arrival-clock;network+=duration*pold
            segments.append(dict(start=clock,end=arrival,state=state,power_w=pold,kind='arrival' if state==start_state else 'quality'))
            transition+=cost;clock=arrival;state=solution
        events.append(dict(solver=solver,ready_s=float(ready),arrival_s=float(arrival),computed_action=solution,applied_action=state,rejection_reason=reason,moves=moved,wakes=wake,transition_j=cost if reason=='accepted' else 0.))
    p=score(inst,state)[1];network+=(H-clock)*p
    rejected=not any(e['rejection_reason']=='accepted' for e in events)
    segments.append(dict(start=clock,end=H,state=state,power_w=p,kind='fallback' if rejected else 'quality'))
    loss={'arrival':0.,'quality':0.,'fallback':0.}
    for s in segments:loss[s['kind']]+=(s['end']-s['start'])*(s['power_w']-optimum)
    assert np.isclose(sum(loss.values()),network-H*optimum,atol=1e-7)
    return dict(network_j=network,transition_j=transition,applied_action=state,accepted=not rejected,events=events,segments=segments,
                delay_loss_j=loss['arrival'],quality_loss_j=loss['quality'],fallback_loss_j=loss['fallback'])
