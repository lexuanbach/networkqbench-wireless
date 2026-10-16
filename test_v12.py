"""Scientific invariants for the unified NetworkQBench-Wireless experiment."""
import unittest
from dataclasses import replace
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel,depolarizing_error
from energy_v12 import *
from exact_density import probabilities
from energy_model import minimum_shots,aggregate_energy_per_useful

class Benchmark12Tests(unittest.TestCase):
 def test_all_physical_optima_and_previous_incumbents(self):
  for sleep in [None,0.,10.,50.]:
   for n in SIZES:
    for seed in range(30):
     inst=workload(n,seed,sleep);r=solve_milp(inst)
     self.assertAlmostEqual(r['power_w'],inst.metadata['power_w'][inst.feasible].min(),places=6)
     prev=previous_incumbent(inst);self.assertTrue(score(inst,prev['incumbent'])[0])
 def test_qubo_and_structured_phase(self):
  for n in SIZES:
   for seed in range(30):
    for sleep in [None,0.,10.,50.]:audit_qubo(workload(n,seed,sleep))
   inst=workload(n,15)
   for theta in [[.73,.24],[.73,.42,.24,.51]]:
    expected=onehot_probs(inst,theta)
    for enc in ['dense','structured']:
     q=circuit(inst,theta,encoding=enc).remove_final_measurements(inplace=False)
     np.testing.assert_allclose(Statevector.from_instruction(q).probabilities(),expected,atol=1e-11)
 def test_noise_kernel(self):
  for n in [2,3,6,8]:
   rng=np.random.default_rng(n);q=QuantumCircuit(n);q.reset(range(n))
   for k in range(12):
    a,b=rng.choice(n,2,replace=False);q.sx(int(a));q.rz(float(rng.normal()),int(a));q.cx(int(a),int(b));q.x(int(b))
   for rate in [0.,.001,.01]:
    model=NoiseModel();model.add_all_qubit_quantum_error(depolarizing_error(rate,2),['cx']);c=q.copy();c.save_probabilities()
    expected=np.asarray(AerSimulator(method='density_matrix',noise_model=model).run(c,shots=1).result().data(0)['probabilities'])
    np.testing.assert_allclose(probabilities(q,rate),expected,atol=1e-11)
  q=QuantumCircuit(2);q.cx(0,1)
  np.testing.assert_allclose(probabilities(q,.2),[.85,.05,.05,.05],atol=1e-12)
 def test_event_conservation_benefit_and_backup(self):
  inst=workload(8,20);best=solve_milp(inst)['solution'];p0=score(inst,inst.metadata['incumbent'])[1];p1=score(inst,best)[1]
  r=event_energy(inst,[(.02,best,'q')],Scenario(migration_j=0))
  self.assertAlmostEqual(r['network_j'],.02*p0+(60-.02)*p1)
  for s in [Scenario(migration_j=1e9),Scenario(migration_s=100),Scenario()]:
   out=event_energy(inst,[(inst.deadline_s+1,best,'late')],s)
   self.assertFalse(out['accepted']);self.assertEqual(out['transition_j'],0);self.assertAlmostEqual(out['network_j'],60*p0)
  out=event_energy(inst,[(.01,best,'milp'),(inst.deadline_s+1,0,'q')])
  self.assertEqual(out['applied_action'],best)
  self.assertAlmostEqual(out['network_j']-60*p1,out['delay_loss_j']+out['quality_loss_j']+out['fallback_loss_j'])
 def test_sleep_transition_boundary(self):
  inst=workload(8,20,10);a,states=assignment_basis(inst);inc=inst.metadata['incumbent']
  chosen=next(int(x) for x in states if inst.feasible[x] and len(set(decode(inst,x))-set(decode(inst,inc)))>0)
  s=Scenario(benefit_gate=False,migration_j=1,wake_j=10,wake_s=.01)
  r=event_energy(inst,[(.02,chosen,'q')],s);e=r['events'][0]
  self.assertAlmostEqual(e['arrival_s'],.03);self.assertEqual(e['transition_j'],e['moves']+10*e['wakes'])
 def test_probability_and_units(self):
  self.assertEqual(minimum_shots(1),1);self.assertTrue(math.isinf(minimum_shots(0)))
  self.assertTrue(math.isinf(aggregate_energy_per_useful([1,2],[False,False])))
  self.assertEqual(20_000*1e-6,.02)
  self.assertEqual(3*float(server_power(0)),660)

if __name__=='__main__':unittest.main()
