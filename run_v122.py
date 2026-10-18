#!/usr/bin/env python3
"""Bounded template experiment; only the explicit output directory is written."""
import argparse,hashlib,json,platform,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector,Parameter
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
import energy_v12 as E
from networkqbench import _normalized_cost
from run_energy_experiments import critical_path
ROOT=Path(__file__).resolve().parent

def template(inst):
    m=inst.metadata;n=inst.n_vars;q=QuantumCircuit(n);q.reset(range(n))
    v=np.zeros(1<<m['nodes'],complex)
    for k in range(m['nodes']):v[1<<k]=1/np.sqrt(m['nodes'])
    for f in range(m['functions']):q.append(StatePreparation(v),range(f*m['nodes'],(f+1)*m['nodes']))
    assignments,states=E.assignment_basis(inst);phases=ParameterVector('phase',len(states));beta=Parameter('beta')
    for a,phase in zip(assignments,phases):
        qs=[f*m['nodes']+int(j) for f,j in enumerate(a)];q.mcp(phase,qs[:-1],qs[-1])
    for f in range(m['functions']):
        for k in range(m['nodes']):
            i=f*m['nodes']+k;j=f*m['nodes']+(k+1)%m['nodes'];q.rxx(beta,i,j);q.ryy(beta,i,j)
    q.measure_all();return q,phases,beta

def bind(comp,phases,beta,inst,theta):
    _,states=E.assignment_basis(inst);norm=_normalized_cost(inst.costs)
    values={p:-float(theta[0]*norm[s]) for p,s in zip(phases,states)};values[beta]=float(theta[1])
    return comp.assign_parameters(values,inplace=False)

def logical_probabilities(comp,n):
    q=comp.remove_final_measurements(inplace=False)
    q.data=[instruction for instruction in q.data if instruction.operation.name!='reset']
    probabilities=Statevector.from_instruction(q).probabilities()
    layout=comp.layout.final_index_layout(filter_ancillas=True)
    physical=np.arange(len(probabilities),dtype=np.uint64);logical=np.zeros(len(probabilities),dtype=np.uint64)
    for j in range(n):logical|=((physical>>layout[j])&1)<<j
    return np.bincount(logical.astype(int),weights=probabilities,minlength=1<<n)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=ROOT/'results/v1_2');ap.add_argument('--output',type=Path,default=ROOT/'results/v1_2_2');ap.add_argument('--rounds',type=int,default=10);ap.add_argument('--seed',type=int,default=20261018);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat();clock=time.perf_counter();angles=json.loads((a.input/'all_on/transfer.json').read_text());refs=pd.read_csv(a.input/'all_on/instances.csv').set_index(['size','seed']);theta={n:next(r['theta'] for r in angles if r['size']==n and r['depth']==1) for n in [6,8,9,12]}
    i=E.workload(6,15);q,ph,b=template(i);c,_=E.compile_circuit(q);bind(c,ph,b,i,theta[6]);E.compile_circuit(E.circuit(i,theta[6]))
    rng=np.random.default_rng(a.seed);rows=[];offline=[];canonical={}
    for rep in range(a.rounds):
        for n in rng.permutation([6,8,9,12]):
            i=E.workload(int(n),15);t=time.perf_counter();q,ph,b=template(i);build=time.perf_counter()-t;c,res=E.compile_circuit(q)
            offline.append(dict(size=n,round=rep,build_s=build,compile_s=res['compile_s'],total_s=build+res['compile_s']))
            if n not in canonical:canonical[n]=(c,ph,b)
        cases=[(n,seed) for n in [6,8,9,12] for seed in range(15,30)];rng.shuffle(cases)
        for n,seed in cases:
            t=time.perf_counter();i=E.workload(n,seed,incumbent=int(refs.loc[(n,seed),'incumbent']));prep=time.perf_counter()-t
            c,ph,b=canonical[n];record=dict(size=n,seed=seed,round=rep,preparation_s=prep)
            for mode in rng.permutation(['template','numeric']):
                t=time.perf_counter()
                if mode=='template':bound=bind(c,ph,b,i,theta[n]);record['binding_s']=time.perf_counter()-t;record.update(template_shot_s=critical_path(bound),template_cx=bound.count_ops().get('cx',0),template_depth=bound.depth())
                else:q=E.circuit(i,theta[n]);record['numeric_build_s']=time.perf_counter()-t;_,res=E.compile_circuit(q);record.update(numeric_compile_s=res['compile_s'],numeric_shot_s=res['shot_s'],numeric_cx=res['cx'],numeric_depth=res['depth'])
            rows.append(record)
        pd.DataFrame(rows).to_csv(a.output/'template_timing.csv',index=False);pd.DataFrame(offline).to_csv(a.output/'template_offline.csv',index=False);print('template timing round',rep+1,flush=True)
    audits=[]
    for n,seed in [(n,seed) for n in [6,8,9,12] for seed in range(15,30)]:
        i=E.workload(n,seed);c,ph,b=canonical[n];bound=bind(c,ph,b,i,theta[n]);p=logical_probabilities(bound,n);ref=E.onehot_probs(i,theta[n]);error=float(abs(p-ref).max());assert error<1e-9,(n,seed,error)
        audits.append(dict(size=n,seed=seed,max_probability_error=error,probability_sum=float(p.sum()),parameters_remaining=len(bound.parameters)))
    pd.DataFrame(audits).to_csv(a.output/'template_equivalence.csv',index=False)
    manifest=dict(campaign='Template',started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.perf_counter()-clock,python=platform.python_version(),platform=platform.platform(),rounds=a.rounds,seed=a.seed,heldout_seeds=list(range(15,30)),sizes=[6,8,9,12],target='grid4x4',reuse=100,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'energy_v12.py']},evidence='Host durations measured; distributions simulated; energy modeled')
    (a.output/'template_manifest.json').write_text(json.dumps(manifest,indent=2));print('template equivalence passed',len(audits),flush=True)
if __name__=='__main__':main()
