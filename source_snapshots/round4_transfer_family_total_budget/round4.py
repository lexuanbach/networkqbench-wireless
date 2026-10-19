"""Additive round-four helpers: training-only family fitting and sparse deadlines."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, platform, time
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint, minimize
from scipy.sparse import coo_matrix
import energy_v12 as E
from energy_model import WATTS, assignment_state, SHOTS
from networkqbench import _normalized_cost
from revision_v121 import sparse_workload
ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'results/round4_20261019'
FAMILIES = ['all_on', 'sleep_0', 'sleep_10', 'sleep_50']
SIZES = [6, 8, 9, 12]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(path, campaign, started, inputs, parameters):
    import scipy, qiskit
    sources = [ROOT / n for n in ['round4.py', 'run_round4.py', 'energy_v12.py', 'revision_v121.py', 'analyze_v12.py']]
    record = dict(campaign=campaign, started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__, qiskit=qiskit.__version__),
                  parameters=parameters, source_sha256={p.name:sha(p) for p in sources},
                  input_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in inputs},
                  evidence='Local wall times measured; quantum quality simulated; all joules modeled',
                  assumptions=dict(host_w=50, qpu_w=20000, reuse=100, horizon_s=60, migration_j=10,
                                   queue_s=.01, submit_s=.0005, roundtrip_s=.002, target='grid4x4',
                                   durations='unchanged declared scenario values', timing='sequential campaigns on one host'))
    Path(path).write_text(json.dumps(record, indent=2)+'\n')


def fit_family(size, sleep, objective, instance_budget):
    """Count every family objective as 15 individual probability evaluations."""
    start = time.perf_counter()
    problems = [E.workload(size, seed, sleep) for seed in range(15)]
    costs = [_normalized_cost(i.costs) for i in problems]
    masks = [i.feasible & ((i.metadata['power_w'] - i.metadata['power_w'][i.feasible].min()) /
                           (i.metadata['power_w'][i.feasible].min()+1) <= .05) for i in problems]
    calls = 0
    limit = int(instance_budget)//15
    assert limit >= 49
    def evaluate(theta):
        nonlocal calls
        assert calls < limit, 'training evaluation cap'
        calls += 1
        values = []
        for i,c,m in zip(problems,costs,masks):
            p = E.onehot_probs(i,theta)
            values.append(float(p@c) if objective=='expectation' else -float(p[m].sum()))
        return float(np.mean(values))
    grid = [np.array([g,b]) for g in np.linspace(0,2*np.pi,7,endpoint=False)
            for b in np.linspace(0,np.pi,7,endpoint=False)]
    values = [evaluate(t) for t in grid]
    best = int(np.argmin(values)); theta = grid[best]; value=values[best]
    reason='grid_only_budget'; iterations=0
    if calls < limit:
        fit = minimize(evaluate, theta, method='Nelder-Mead',
                       options={'maxiter':40, 'maxfev':limit-calls, 'xatol':1e-3, 'fatol':1e-5})
        if fit.fun < value: theta,value=fit.x,float(fit.fun)
        reason=str(fit.message); iterations=int(fit.nit)
    return dict(theta=np.asarray(theta).tolist(), objective=objective, objective_value=value,
                training_seeds=list(range(15)), family_evaluations=calls, instance_evaluations=15*calls,
                historical_instance_budget=int(instance_budget), max_family_evaluations=limit,
                iterations=iterations, stopping_reason=reason, offline_s=time.perf_counter()-start,
                optimum_informed=objective=='useful_mass')


def deploy_sparse(inst, solution, ready, compute):
    """One-action event integral with no enumeration or test-optimum access."""
    incumbent=int(inst.metadata['incumbent']); p0=E.score(inst,incumbent)[1]
    ok,p = E.score(inst,int(solution)) if solution is not None else (False,None)
    timely=bool(ready <= inst.deadline_s and ready < 60)
    moves=int(np.count_nonzero(E.decode(inst,int(solution)) != E.decode(inst,incumbent))) if ok else 0
    cost=10.*moves
    accepted=bool(ok and timely and (60-ready)*(p0-p)>cost+1e-12)
    transition=cost if accepted else 0.
    network=p0*ready+p*(60-ready) if accepted else p0*60
    return dict(incumbent_w=p0, feasible=bool(ok), timely=timely, accepted=accepted,
                applied_action=str(int(solution)) if accepted else str(incumbent),
                moves=moves, compute_j=compute, transition_j=transition, network_j=network,
                total_j=compute+transition+network,
                rejection_reason='accepted' if accepted else 'no_candidate' if not ok else 'late' if not timely else 'no_benefit')


def deadline_milp(inst,time_limit=30.,deadline=None):
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
    build=time.perf_counter()-start
    remaining=time_limit if deadline is None else min(time_limit,deadline-build)
    if remaining<=0:
        return dict(solution=None,power_w=None,feasible=False,status=1,exact=False,
                    message='construction exhausted deadline; solver not called',mip_gap=None,bound_w=None,
                    construction_s=build,solve_s=0.,runtime_s=time.perf_counter()-start,
                    variables=count,constraints=len(lower),nonzeros=matrix.nnz,
                    solver_called=False,solver_budget_s=0.)
    tick=time.perf_counter()
    result=milp(c,integrality=integ,bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix,lower,upper),
                options={'mip_rel_gap':0.,'time_limit':remaining})
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
                variables=count,constraints=len(lower),nonzeros=matrix.nnz,solver_called=True,solver_budget_s=remaining)

