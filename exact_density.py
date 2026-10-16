"""Portable exact density simulator, independently checked against Qiskit Aer.

Compile the included C++17 kernel on first use. Only the compiled native gate
basis is accepted. Initial resets are allowed; intermediate resets are not.
"""
from pathlib import Path
import ctypes,hashlib,subprocess,sys,tempfile
import numpy as np
SOURCE=Path(__file__).with_name('density_kernel.cpp')

def probabilities(circuit,rate,workers=4):
 tag=hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
 lib=Path(tempfile.gettempdir())/('networkqbench-density-'+tag+('.dylib' if sys.platform=='darwin' else '.so'))
 if not lib.exists():subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(SOURCE),'-o',str(lib)],check=True)
 func=ctypes.CDLL(str(lib)).density_probabilities
 ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
 func.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,ip,dp,ctypes.c_double,dp,ctypes.c_int];func.restype=ctypes.c_int
 ops=[];a=[];b=[];theta=[];touched=set()
 for inst in circuit.data:
  name=inst.operation.name;qs=[circuit.find_bit(q).index for q in inst.qubits]
  if name=='reset':
   if any(q in touched for q in qs):raise ValueError('intermediate reset is unsupported')
   continue
  if name in ['barrier','measure','save_probabilities']:continue
  if name not in ['rz','sx','x','cx']:raise ValueError('unsupported gate '+name)
  touched.update(qs);ops.append({'rz':1,'sx':2,'x':3,'cx':4}[name]);a.append(qs[0]);b.append(qs[-1]);theta.append(float(inst.operation.params[0]) if name=='rz' else 0.)
 p=np.empty(1<<circuit.num_qubits);code=func(circuit.num_qubits,len(ops),np.array(ops,np.int32),np.array(a,np.int32),np.array(b,np.int32),np.array(theta,float),float(rate),p,workers)
 if code:raise RuntimeError('density kernel error '+str(code))
 if p.min()<-1e-9:raise AssertionError('negative probability')
 return np.maximum(p,0)/p.sum()
