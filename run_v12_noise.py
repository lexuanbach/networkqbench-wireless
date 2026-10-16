#!/usr/bin/env python3
"""Exact noisy-distribution study on an explicitly separate all-to-all target."""
from pathlib import Path
import json,time,hashlib
from concurrent.futures import ProcessPoolExecutor
from exact_density import probabilities as exact_probabilities
import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel,depolarizing_error
from energy_v12 import *
from networkqbench import qaoa_probabilities
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'/'v1_2'/'noise'
SEEDS=[15,18,21,24,27];RATES=[0.,.0001,.001,.01]


def precompute_job(task):
 size,seed,alg,mixer,encoding,rate,theta=task
 key=f'n{size}_s{seed}_{alg}_r{rate:g}';cache=OUT/(key+'.npz')
 if cache.exists():return key+' cached'
 inst=workload(size,seed);q=circuit(inst,theta,mixer,encoding);compiled,_=compile_circuit(q,'all_to_all');compiled=compiled.remove_final_measurements(inplace=False)
 start=time.perf_counter()
 if rate==0:p=Statevector.from_instruction(compiled).probabilities()
 else:p=exact_probabilities(compiled,rate,workers=1)
 seconds=time.perf_counter()-start
 np.savez_compressed(cache,probability=p,simulation_s=seconds,engine='statevector_zero_noise' if rate==0 else 'exact_density_cpp')
 return key+' exact seconds '+str(round(seconds,3))

def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[];samples=[]
 transfer=json.loads((ROOT/'results/sustainability/transfer.json').read_text())
 tasks=[]
 for seed in SEEDS:
  for alg,mixer,enc in [('qaoa_transfer','x','dense'),('onehot_transfer','onehot','dense'),('onehot_structured','onehot','structured')]:
   t=next(r for r in transfer if r['size']==12 and r['mixer']==mixer)
   for rate in RATES:tasks.append((12,seed,alg,mixer,enc,rate,[t['gamma'],t['beta']]))
 with ProcessPoolExecutor(max_workers=6) as pool:
  for status in pool.map(precompute_job,tasks):print(status,flush=True)
 for size in SIZES:
  for seed in SEEDS:
   inst=workload(size,seed);opt=inst.metadata['power_w'][inst.feasible].min()
   good=inst.feasible & ((inst.metadata['power_w']-opt)/(opt+1)<=.05)
   for alg,mixer,encoding in [('qaoa_transfer','x','dense'),('onehot_transfer','onehot','dense'),('onehot_structured','onehot','structured')]:
    t=next(r for r in transfer if r['size']==size and r['mixer']==mixer);theta=[t['gamma'],t['beta']]
    expected=onehot_probs(inst,theta) if mixer=='onehot' else qaoa_probabilities(inst.costs,size,*theta)
    q=circuit(inst,theta,mixer,encoding);compiled,res=compile_circuit(q,'all_to_all')
    compiled=compiled.remove_final_measurements(inplace=False)
    compiled_ideal=Statevector.from_instruction(compiled).probabilities()
    compilation_error=float(np.max(abs(compiled_ideal-expected)))
    assert compilation_error<1e-6,(size,seed,alg,compilation_error)
    compiled.save_probabilities()
    for rate in RATES:
     key=f'n{size}_s{seed}_{alg}_r{rate:g}';cache=OUT/(key+'.npz')
     if cache.exists():
      saved=np.load(cache);p=saved['probability'];seconds=float(saved['simulation_s'])
     else:
      saved={}
      noise=NoiseModel()
      if rate:noise.add_all_qubit_quantum_error(depolarizing_error(rate,2),['cx'])
      sim=AerSimulator(method='density_matrix',noise_model=noise,max_parallel_threads=4,max_memory_mb=4096,fusion_enable=True)
      start=time.perf_counter();result=sim.run(compiled,shots=1,seed_simulator=20261016).result();seconds=time.perf_counter()-start
      if not result.success:raise RuntimeError(str(result))
      p=np.asarray(result.data(0)['probabilities'],float);p=np.maximum(p,0);p/=p.sum()
      np.savez_compressed(cache,probability=p,simulation_s=seconds)
     if rate==0:assert np.allclose(p,compiled_ideal,atol=2e-10),(size,seed,alg,np.max(abs(p-compiled_ideal)))
     rows.append(dict(size=size,seed=seed,algorithm=alg,cx_depolarizing_parameter=rate,p_quality=float(p[good].sum()),p_feasible=float(p[inst.feasible].sum()),p_optimal=float(p[inst.feasible & np.isclose(inst.metadata['power_w'],opt,atol=1e-8)].sum()),simulation_s=seconds,simulation_engine=str(saved['engine']) if cache.exists() and 'engine' in saved else 'qiskit_aer_density_matrix',compilation_probability_error=compilation_error,**res))
     for shots in SHOTS:
      for rep in range(30):
       rng=np.random.default_rng(20261016+size*10000+seed*100+rep);draw=rng.choice(len(p),shots,p=p);valid=draw[inst.feasible[draw]]
       state=int(valid[np.argmin(inst.metadata['power_w'][valid])]) if len(valid) else int(draw[0])
       samples.append(dict(size=size,seed=seed,algorithm=alg,noise=rate,shots=shots,repeat=rep,solution=state,feasible=bool(inst.feasible[state]),quality=bool(good[state]),power_w=float(inst.metadata['power_w'][state])))
     print(key,'seconds',round(seconds,3),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'probabilities.csv',index=False)
  pd.DataFrame(samples).to_csv(OUT/'samples.csv',index=False)
 (OUT/'manifest.json').write_text(json.dumps(dict(schema_version='1.2',qiskit_aer='0.17.2',method='exact density_matrix; Aer for n<=9, independent C++ kernel for n=12; zero-noise statevector reference',target='all_to_all',max_qubits=12,seeds=SEEDS,rates=RATES,gate_noise='two-qubit depolarizing channel after each compiled CX; no other noise',repeats=30,shots=list(SHOTS),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),conditions=len(rows),sampling_batches=len(samples)),indent=2))
 print('noise complete',len(rows),flush=True)
if __name__=='__main__':main()
