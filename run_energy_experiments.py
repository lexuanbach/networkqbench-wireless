#!/usr/bin/env python3
"""Run the sustainability extension without changing historical results.

Only classical wall times are measured. Quantum outcomes are ideal simulation;
all electricity and remote timing estimates are explicitly scenario based.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile, __version__ as qiskit_version
from qiskit.circuit.library import DiagonalGate, StatePreparation
from qiskit.transpiler import CouplingMap
from networkqbench import qaoa_solver, qaoa_probabilities, _normalized_cost
from run_review_experiments import onehot_qaoa, onehot_probabilities
from energy_model import SIZES, SHOTS, LOAD, WATTS, SPEC_URL, energy_placement, classical_control

ROOT=Path(__file__).resolve().parent
# Explicit operation-duration scenarios, not FakeGuadalupe measurements.
DURATIONS={"rz":0.,"sx":35e-9,"x":70e-9,"cx":300e-9,
           "reset":5e-6,"measure":1.6e-6,"barrier":0.}


def circuit(inst,gamma,beta,onehot=False):
    n=inst.n_vars; q=QuantumCircuit(n); q.reset(range(n))
    if onehot:
        width=inst.metadata["nodes"]
        v=np.zeros(1<<width,dtype=complex)
        for k in range(width): v[1<<k]=1/np.sqrt(width)
        for f in range(inst.metadata["functions"]):
            q.append(StatePreparation(v),range(f*width,(f+1)*width))
    else: q.h(range(n))
    q.append(DiagonalGate(np.exp(-1j*gamma*_normalized_cost(inst.costs))),range(n))
    if onehot:
        width=inst.metadata["nodes"]
        for f in range(inst.metadata["functions"]):
            for k in range(width):
                a=f*width+k; b=f*width+(k+1)%width
                q.rxx(beta,a,b); q.ryy(beta,a,b)
    else: q.rx(2*beta,range(n))
    q.measure_all()
    return q


def critical_path(q):
    free=np.zeros(q.num_qubits)
    for inst in q.data:
        name=inst.operation.name
        if name not in DURATIONS: raise ValueError(f"unaccounted gate {name}")
        ids=[q.find_bit(b).index for b in inst.qubits]
        if ids:
            finish=max(free[ids])+DURATIONS[name]
            free[ids]=finish
    return float(max(free))


def compile_record(inst,gamma,beta,onehot):
    start=time.perf_counter(); q=circuit(inst,gamma,beta,onehot)
    build_s=time.perf_counter()-start
    start=time.perf_counter()
    compiled=transpile(q,basis_gates=["rz","sx","x","cx"],
                       coupling_map=CouplingMap.from_grid(4,4),
                       optimization_level=2,seed_transpiler=20261015)
    compile_s=time.perf_counter()-start
    return {"circuit_build_s":build_s,"compile_s":compile_s,
            "shot_s":critical_path(compiled),"cx":int(compiled.count_ops().get("cx",0)),
            "depth":compiled.depth(),"physical_width":compiled.num_qubits,
            "gamma":gamma,"beta":beta,"mixer":"onehot" if onehot else "x",
            "target":"grid4x4","resource_evidence":"compiled_estimate",
            "duration_evidence":"modeled_operation_durations",
            "objective_sha256":hashlib.sha256(inst.costs.tobytes()).hexdigest()}


def angle_mean(xs,period):
    return float(np.angle(np.exp(2j*np.pi*np.asarray(xs)/period).mean())%(2*np.pi)*period/(2*np.pi))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,default=ROOT/"results"/"sustainability")
    ap.add_argument("--sizes",type=int,nargs="+",default=list(SIZES));ap.add_argument("--seeds",type=int,default=30)
    ap.add_argument("--repeats",type=int,default=30);args=ap.parse_args()
    if args.seeds<16: raise ValueError("need fixed train seeds 0--14 and at least one held-out seed")
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    instances={};fits=[];classical=[];reference=[];start_all=time.perf_counter()
    for size in args.sizes:
        for seed in range(args.seeds):
            start=time.perf_counter();inst=energy_placement(size,seed);prep=time.perf_counter()-start
            instances[size,seed]=inst
            opt=float(inst.metadata["power_w"][inst.feasible].min())
            ref={"size":size,"seed":seed,"split":"train" if seed<15 else "test",
                 "optimum_w":opt,"incumbent":inst.metadata["incumbent"],
                 "incumbent_w":float(inst.metadata["power_w"][inst.metadata["incumbent"]]),
                 "deadline_s":inst.deadline_s,"dense_preparation_s":prep,
                 "demands":json.dumps(inst.metadata["demands"].tolist()),
                 "capacities":json.dumps(inst.metadata["capacities"].tolist()),
                 "latency_matrix_ms":json.dumps(inst.metadata["latency"].tolist())}
            reference.append(ref)
            for method in ["milp","local_search","heuristic"]:
                r=classical_control(inst,method)
                if method=="milp" and abs(r["power_w"]-opt)>1e-6:
                    raise AssertionError((size,seed,r,opt))
                classical.append({**ref,"algorithm":method,**r})
            for mixer,solver in [("x",qaoa_solver),("onehot",onehot_qaoa)]:
                r=solver(inst,np.random.default_rng(900000+size*100+seed))
                fits.append({"size":size,"seed":seed,"mixer":mixer,"gamma":r["gamma"],
                             "beta":r["beta"],"evals":r["evals"],"simulation_s":r["runtime_s"]})
        print(f"fit size={size}: {args.seeds} MILP oracle matches, 2 mixers",flush=True)
    fit=pd.DataFrame(fits);fit.to_csv(out/"fits.csv",index=False)
    pd.DataFrame(reference).to_csv(out/"instances.csv",index=False)
    pd.DataFrame(classical).to_csv(out/"classical.csv",index=False)
    transfer={};training=[]
    for (size,mixer),g in fit[fit.seed<15].groupby(["size","mixer"]):
        ga=angle_mean(g.gamma,2*np.pi);be=angle_mean(g.beta,np.pi)
        transfer[size,mixer]=(ga,be,float(g.simulation_s.sum()))
        training.append({"size":int(size),"mixer":mixer,"gamma":ga,"beta":be,
                         "training_seeds":list(range(15)),"offline_simulation_s":float(g.simulation_s.sum())})
    (out/"transfer.json").write_text(json.dumps(training,indent=2)+"\n")
    resources=[];records=[];arrays={}
    refs={(r["size"],r["seed"]):r for r in reference}
    fitidx={(r["size"],r["seed"],r["mixer"]):r for r in fits}
    for size in args.sizes:
        for seed in range(args.seeds):
            inst=instances[size,seed];ref=refs[size,seed];power=inst.metadata["power_w"]
            opt=ref["optimum_w"]; quality=inst.feasible & ((power-opt)/(abs(opt)+1)<=.05)
            optimal=inst.feasible & (power<=opt+1e-8)
            for aidx,alg in enumerate(["qaoa_online","qaoa_transfer","onehot_transfer"]):
                mixer="onehot" if alg=="onehot_transfer" else "x"
                own=fitidx[size,seed,mixer]
                if alg=="qaoa_online":
                    ga,be=own["gamma"],own["beta"];evals=int(own["evals"])+1;offline=0.
                else: ga,be,offline=transfer[size,mixer];evals=1
                p=onehot_probabilities(inst,ga,be) if mixer=="onehot" else qaoa_probabilities(inst.costs,size,ga,be)
                arrays[f"n{size}_s{seed}_{alg}"]=p
                res=compile_record(inst,ga,be,mixer=="onehot")
                resources.append({"size":size,"seed":seed,"algorithm":alg,**res})
                for shots in SHOTS:
                    for repeat in range(args.repeats):
                        rng=np.random.default_rng(10_000_000+size*100000+seed*1000+aidx*100+repeat)
                        samples=rng.choice(len(p),size=shots,p=p)
                        start=time.perf_counter()
                        valid=samples[inst.feasible[samples]]
                        chosen=int(valid[np.argmin(power[valid])]) if len(valid) else int(samples[0])
                        post=time.perf_counter()-start
                        records.append({**ref,"algorithm":alg,"shots":shots,"repeat":repeat,
                                        "solution":chosen,"power_w":float(power[chosen]),
                                        "feasible":bool(inst.feasible[chosen]),"quality":bool(quality[chosen]),
                                        "optimal":bool(optimal[chosen]),"p_quality":float(p[quality].sum()),
                                        "p_optimal":float(p[optimal].sum()),"p_feasible":float(p[inst.feasible].sum()),
                                        "evals":evals,"postprocessing_s":post,"offline_s":offline,
                                        "simulation_evidence":"exact_simulated"})
            if (seed+1)%5==0:
                print(f"compile/sample size={size} seed={seed} rows={len(records)}",flush=True)
                pd.DataFrame(resources).to_csv(out/"circuits.csv",index=False)
        pd.DataFrame(records).to_csv(out/"samples.csv",index=False)
        np.savez_compressed(out/"distributions.npz",**arrays)
    manifest={"schema_version":"1.1","python":platform.python_version(),"platform":platform.platform(),
              "numpy":np.__version__,"qiskit":qiskit_version,"sizes":args.sizes,"seeds":args.seeds,
              "instances":len(reference),"test_instances":sum(r["split"]=="test" for r in reference),
              "repeats":args.repeats,"shots":list(SHOTS),"milp_oracle_matches":len(reference),
              "fit_rows":len(fits),"matched_circuits":len(resources),"sampling_rows":len(records),
              "operation_durations_s":DURATIONS,"power_curve":{"url":SPEC_URL,"load":LOAD.tolist(),"watts":WATTS.tolist()},
              "local_power_w":[10,50,100],"qpu_allocated_power_w":[10000,20000,40000],
              "qpu_marginal_fraction":[.001,.01,.1],"default_marginal_fraction":.01,
              "marginal_fraction_evidence":"illustrative sensitivity; no marginal-power measurement",
              "reuse_counts":[1,10,100,1000],"horizons_s":[1,10,60,300,3600],"migration_j_per_function":[0,1,10,100],
              "host_optimizer_s_per_evaluation":.0001,"host_optimizer_evidence":"illustrative lower-cost scenario",
              "telemetry":"powermetrics requires superuser; no energy measurement performed",
              "allocation":"full system power times active service; no external queue charged; utilization=1",
              "communication_boundary":"local client activity during submit/roundtrip; external WAN excluded",
              "compilation":"per-instance online cost; never amortized across different objectives",
              "offline":"actual local ideal-simulator training cost shared by family; charged / M",
              "quality":"feasible and (power-optimum)/(abs(optimum)+1)<=0.05",
              "quantum_evidence":"ideal statevector, complete compiled circuits, modeled timing and power; no live QPU",
              "elapsed_s":time.perf_counter()-start_all,
              "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__),ROOT/"energy_model.py",ROOT/"networkqbench.py",ROOT/"run_review_experiments.py"]}}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ["instances","matched_circuits","sampling_rows","elapsed_s"]}),flush=True)

if __name__=="__main__": main()
