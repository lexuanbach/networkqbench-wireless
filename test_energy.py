"""Meaningful invariants for physical accounting, MILP, and circuit semantics."""
import math
import unittest
import numpy as np
from energy_model import *


class EnergyTests(unittest.TestCase):
    def test_power_units_and_curve(self):
        np.testing.assert_allclose(server_power(LOAD),WATTS)
        self.assertEqual(float(server_power(0))*3,660.)
        with self.assertRaises(ValueError): server_power(1.01)

    def test_unknown_and_duplicate_ledger(self):
        def c(v,m=1): return EnergyComponent("training",v,"allocated","modeled_energy","test","s",m)
        self.assertIsNone(EnergyLedger((c(None),)).total())
        self.assertEqual(EnergyLedger((c(20,10),)).total(),2.)
        with self.assertRaises(ValueError): EnergyLedger((c(1),c(1))).total()

    def test_shot_endpoints(self):
        self.assertTrue(math.isinf(minimum_shots(0)))
        self.assertEqual(minimum_shots(1),1)
        self.assertEqual(minimum_shots(.5,.95),5)
        self.assertFalse(shot_frontier(.5,0,1,.1,1,2,.1)[-1])
        self.assertTrue(math.isinf(aggregate_energy_per_useful([1,2],[False,False])))
        self.assertEqual(aggregate_energy_per_useful([1,2],[True,False]),3.)

    def test_all_milp_optima(self):
        for n in SIZES:
            for seed in range(30):
                inst=energy_placement(n,seed);r=placement_milp(inst)
                self.assertAlmostEqual(r["power_w"],inst.metadata["power_w"][inst.feasible].min(),places=6)

    def test_late_and_invalid_keep_incumbent(self):
        inst=energy_placement(6,2);best=int(np.where(inst.feasible)[0][np.argmin(inst.metadata["power_w"][inst.feasible])])
        p0=inst.metadata["power_w"][inst.metadata["incumbent"]]
        for state,delay in [(0,0),(best,inst.deadline_s+1)]:
            r=applied_energy(inst,state,delay,60,100)
            self.assertFalse(r["accepted"]);self.assertEqual(r["migration_j"],0)
            self.assertEqual(r["network_j"],60*p0)
        r=applied_energy(inst,best,0,60,0)
        self.assertAlmostEqual(r["network_j"],60*inst.metadata["power_w"][best])

    def test_break_even(self):
        self.assertEqual(break_even(200,100,110,100),30)
        self.assertTrue(math.isinf(break_even(200,100,100,110)))

    def test_complete_circuits_match_simulation(self):
        from qiskit.quantum_info import Statevector
        from run_energy_experiments import circuit, critical_path
        from run_review_experiments import onehot_probabilities
        from networkqbench import qaoa_probabilities
        for size in [6,8]:
            inst=energy_placement(size,15)
            for onehot in [False,True]:
                q=circuit(inst,.73,.24,onehot).remove_final_measurements(inplace=False)
                actual=Statevector.from_instruction(q).probabilities()
                expected=onehot_probabilities(inst,.73,.24) if onehot else qaoa_probabilities(inst.costs,size,.73,.24)
                np.testing.assert_allclose(actual,expected,atol=1e-12)
        from qiskit import QuantumCircuit
        q=QuantumCircuit(2);q.sx(0);q.sx(1);q.cx(0,1);q.measure_all()
        self.assertAlmostEqual(critical_path(q),35e-9+300e-9+1.6e-6,places=12)

if __name__=="__main__":unittest.main()
