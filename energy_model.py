"""Physical-unit scenario model; no values in this module are QPU measurements.

SPEC powers are transcribed from a published server benchmark, then transferred
to synthetic service loads. The legacy placement weights are never joules.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time
import itertools
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from networkqbench import Instance, bit_table, make_service_placement
from extension_api import EvidenceKind

SPEC_URL = "https://www.spec.org/power_ssj2008/results/res2025q1/power_ssj2008-20250207-01482.html"
LOAD = np.arange(11, dtype=float) / 10
WATTS = np.array([220, 258, 303, 347, 394, 444, 498, 574, 718, 904, 924.])
SIZES = (6, 8, 9, 12)
SHOTS = (16, 64, 256, 1024)
PROFILES = {"optimistic": (0.01, 0.0005, 0.002),
            "nominal": (0.5, 0.005, 0.02), "stressed": (5., 0.02, 0.1)}


@dataclass(frozen=True)
class EnergyComponent:
    component: str
    joules: float | None
    boundary: str
    evidence: str
    source: str
    scenario: str
    amortization_count: int = 1

    def validate(self):
        EvidenceKind(self.evidence)
        if self.joules is not None and (not math.isfinite(self.joules) or self.joules < 0):
            raise ValueError("energy must be nonnegative and finite, or unknown")
        if self.amortization_count < 1 or not all((self.component, self.boundary,
                                                   self.evidence, self.source, self.scenario)):
            raise ValueError("explicit boundary, evidence, source and scenario required")


@dataclass(frozen=True)
class EnergyLedger:
    components: tuple[EnergyComponent, ...]

    def total(self):
        keys = [(c.component, c.boundary) for c in self.components]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate component would double count energy")
        for c in self.components:
            c.validate()
        if any(c.joules is None for c in self.components):
            return None
        return sum(c.joules / c.amortization_count for c in self.components)


@dataclass(frozen=True)
class AdmissionDecision:
    policy: str
    selected_solver: str
    rejection_reason: str
    predicted_quality_probability: float
    predicted_delay_s: float
    predicted_total_energy_j: float
    computed_action: int
    applied_action: int


def server_power(utilization):
    u = np.asarray(utilization, dtype=float)
    if np.any(~np.isfinite(u)) or np.any(u < -1e-10) or np.any(u > 1 + 1e-10):
        raise ValueError("utilization outside [0,1]")
    return np.interp(np.clip(u, 0, 1), LOAD, WATTS)


def energy_placement(size: int, seed: int) -> Instance:
    base = make_service_placement(size, seed)
    m = dict(base.metadata)
    f, n = m["functions"], m["nodes"]
    b = bit_table(size).reshape(-1, f, n)
    loads = (b * m["demands"][None, :, None]).sum(axis=1)
    latency = np.zeros(len(b))
    for k in range(f - 1):
        latency += np.einsum("bi,ij,bj->b", b[:, k], m["latency"], b[:, k+1])
    feasible = base.feasible & (latency <= 4 + 1e-10)
    powers = server_power(np.clip(loads / m["capacities"], 0, 1)).sum(axis=1)
    # A bounded normalized objective and binary penalty: feasible costs <=1,
    # infeasible costs >=2. Dense synthesis is explicitly an exponential audit.
    costs = (powers - n * WATTS[0]) / (n * (WATTS[-1] - WATTS[0])) + 2*(~feasible)
    m.update(power_w=powers, latency_ms=latency, latency_limit_ms=4.,
             power_source=SPEC_URL, power_evidence="modeled_network", costs_scope="complete")
    # All functions on node 0 is guaranteed feasible by the original generator.
    incumbent = sum(1 << (k*n) for k in range(f))
    assert feasible[incumbent]
    m["incumbent"] = incumbent
    return Instance("placement", size, size, seed, costs, feasible,
                    base.volatility, base.deadline_s, m)


def assignment_state(assignment, nodes):
    return sum(1 << (f*nodes + int(node)) for f, node in enumerate(assignment))


def physical_score(inst, assignment):
    """Sparse evaluator for deployable classical controls, without dense oracle."""
    m = inst.metadata
    loads = np.bincount(assignment, weights=m["demands"], minlength=m["nodes"])
    latency = sum(m["latency"][a,b] for a,b in zip(assignment[:-1], assignment[1:]))
    feasible = np.all(loads <= m["capacities"] + 1e-10) and latency <= 4 + 1e-10
    power = float(server_power(np.minimum(loads/m["capacities"], 1)).sum())
    return bool(feasible), power


def placement_milp(inst):
    """Exact nonconvex piecewise-linear MILP using segment selection, not a hull.

    Variables: assignment x, adjacent-function products w, selected segments z,
    and fractional progress t on each selected segment. No dense costs used.
    """
    start = time.perf_counter()
    m = inst.metadata; f, n = m["functions"], m["nodes"]
    nx = f*n; nw=(f-1)*n*n; z0=nx+nw; t0=z0+10*n; count=t0+10*n
    c=np.zeros(count); integ=np.zeros(count); integ[:nx]=1; integ[z0:t0]=1
    rows=[]; lower=[]; upper=[]
    def constraint(items, lo, hi):
        row=np.zeros(count)
        for j, v in items: row[j] += v
        rows.append(row); lower.append(lo); upper.append(hi)
    for k in range(f): constraint([(k*n+j,1) for j in range(n)],1,1)
    for j in range(n):
        constraint([(z0+j*10+k,1) for k in range(10)],1,1)
        terms=[(k*n+j,float(m["demands"][k])/m["capacities"][j]) for k in range(f)]
        terms += [(z0+j*10+k,-k/10) for k in range(10)]
        terms += [(t0+j*10+k,-0.1) for k in range(10)]
        constraint(terms,0,0)
        for k in range(10):
            z=z0+j*10+k; t=t0+j*10+k
            constraint([(t,1),(z,-1)],-np.inf,0)
            c[z]=WATTS[k]; c[t]=WATTS[k+1]-WATTS[k]
    lat=[]
    for k in range(f-1):
        for i in range(n):
            for j in range(n):
                w=nx+k*n*n+i*n+j; a=k*n+i; b=(k+1)*n+j
                constraint([(w,1),(a,-1)],-np.inf,0)
                constraint([(w,1),(b,-1)],-np.inf,0)
                constraint([(w,1),(a,-1),(b,-1)],-1,np.inf)
                lat.append((w,float(m["latency"][i,j])))
    constraint(lat,-np.inf,4)
    build_s=time.perf_counter()-start
    result=milp(c, integrality=integ, bounds=Bounds(0,1),
                constraints=LinearConstraint(np.array(rows),lower,upper),
                options={"mip_rel_gap":0.0})
    if not result.success: raise RuntimeError(result.message)
    a=np.rint(result.x[:nx]).reshape(f,n).argmax(axis=1)
    state=assignment_state(a,n); valid, power=physical_score(inst,a)
    if not valid: raise AssertionError("MILP infeasible output")
    return {"solution":state,"power_w":power,"runtime_s":time.perf_counter()-start,
            "construction_s":build_s,"mip_gap":float(result.mip_gap)}


def classical_control(inst, method):
    if method == "milp": return placement_milp(inst)
    start=time.perf_counter(); m=inst.metadata; f,n=m["functions"],m["nodes"]
    incumbent=np.zeros(f,dtype=int)
    if method == "heuristic":
        # One deterministic coordinate pass, with complete feasibility checks.
        starts=[incumbent]
    elif method == "local_search":
        rng=np.random.default_rng(73000+inst.seed+100*inst.n_vars)
        starts=[incumbent]+[rng.integers(n,size=f) for _ in range(7)]
    else: raise ValueError(method)
    best=incumbent.copy(); best_p=physical_score(inst,best)[1]
    for initial in starts:
        a=initial.copy()
        if not physical_score(inst,a)[0]: a=incumbent.copy()
        passes=1 if method=="heuristic" else 10
        for _ in range(passes):
            changed=False
            for k in range(f):
                options=[]
                for node in range(n):
                    b=a.copy(); b[k]=node; ok,p=physical_score(inst,b)
                    if ok: options.append((p,node))
                _,node=min(options)
                if node != a[k]: changed=True
                a[k]=node
            if not changed: break
        p=physical_score(inst,a)[1]
        if p < best_p: best,best_p=a.copy(),p
    return {"solution":assignment_state(best,n),"power_w":best_p,
            "runtime_s":time.perf_counter()-start,"construction_s":0.}


def minimum_shots(p, alpha=.9):
    if not 0 <= p <= 1 or not 0 < alpha < 1: raise ValueError("invalid probability")
    if p==0: return math.inf
    if p==1: return 1
    return math.ceil(math.log1p(-alpha)/math.log1p(-p))


def shot_frontier(p, deadline, time_fixed, time_shot, budget, energy_fixed, energy_shot, alpha=.9):
    if min(time_shot,energy_shot)<=0: raise ValueError("positive shot cost required")
    nt=max(0, math.floor((deadline-time_fixed)/time_shot))
    ne=max(0, math.floor((budget-energy_fixed)/energy_shot))
    need=minimum_shots(p,alpha)
    return need,nt,ne, need <= min(nt,ne)


def applied_energy(inst, solution, delay, horizon, migration_j):
    if min(delay,horizon,migration_j)<0: raise ValueError("negative time or energy")
    m=inst.metadata; prev=m["incumbent"]; valid=bool(inst.feasible[solution])
    accepted=valid and delay<=inst.deadline_s and delay<horizon
    applied=solution if accepted else prev
    arrival=min(delay,horizon) if accepted else horizon
    p0=float(m["power_w"][prev]); p1=float(m["power_w"][applied])
    old=[(prev >> (k*m["nodes"])) & ((1<<m["nodes"])-1) for k in range(m["functions"])]
    new=[(applied >> (k*m["nodes"])) & ((1<<m["nodes"])-1) for k in range(m["functions"])]
    moves=sum(a!=b for a,b in zip(old,new))
    return {"applied_solution":applied,"accepted":accepted,"network_j":p0*arrival+p1*(horizon-arrival),
            "migration_j":moves*migration_j,"moves":moves,"applied_power_w":p1}


def break_even(delta_compute_j, delta_transition_j, power_c, power_q):
    dp=power_c-power_q
    if dp<=0: return math.inf
    return max(0.,(delta_compute_j+delta_transition_j)/dp)


def aggregate_energy_per_useful(energy, useful):
    count=np.asarray(useful,dtype=bool).sum()
    return float(np.sum(energy)/count) if count else math.inf
