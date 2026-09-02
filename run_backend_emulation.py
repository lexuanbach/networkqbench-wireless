#!/usr/bin/env python3
"""Calibration-snapshot noisy emulation of representative full QAOA circuits.

The backend is Qiskit's archived ``FakeGuadalupeV2`` calibration snapshot.
Aer executes its native-gate noise model locally.  These rows validate circuit
construction, mapping, noisy sampling, and phase timing, but they are not live
QPU jobs and contain no provider queue measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, __version__ as qiskit_version, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator, __version__ as aer_version
from qiskit_ibm_runtime import __version__ as runtime_version
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

from networkqbench import GENERATORS, _normalized_cost, qaoa_solver


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def circuit_for(inst, gamma: float, beta: float) -> QuantumCircuit:
    circuit = QuantumCircuit(inst.n_vars)
    circuit.h(range(inst.n_vars))
    circuit.append(DiagonalGate(np.exp(-1j * gamma * _normalized_cost(inst.costs))),
                   range(inst.n_vars))
    circuit.rx(2.0 * beta, range(inst.n_vars))
    circuit.measure_all()
    return circuit


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    backend = FakeGuadalupeV2()
    simulator = AerSimulator.from_backend(backend)
    rows = []
    for task, generator in GENERATORS.items():
        for seed in range(5):
            t0 = time.perf_counter(); inst = generator(6, 40_000 + seed)
            build_instance_s = time.perf_counter() - t0
            fitted = qaoa_solver(inst, np.random.default_rng(65_000_000 + seed))
            t0 = time.perf_counter(); circuit = circuit_for(inst, fitted["gamma"], fitted["beta"])
            build_circuit_s = time.perf_counter() - t0
            t0 = time.perf_counter(); compiled = transpile(
                circuit, backend=backend, optimization_level=3, seed_transpiler=20260902)
            transpile_s = time.perf_counter() - t0
            t0 = time.perf_counter(); result = simulator.run(
                compiled, shots=1024, seed_simulator=66_000_000 + seed).result()
            sample_s = time.perf_counter() - t0
            t0 = time.perf_counter(); counts = result.get_counts()
            states = np.array([int(key.replace(" ", ""), 2) for key in counts])
            count_values = np.array(list(counts.values()), dtype=int)
            feasible = states[inst.feasible[states]]
            selected = int(feasible[np.argmin(inst.costs[feasible])]) if len(feasible) \
                else int(states[np.argmin(inst.costs[states])])
            optimum = float(inst.costs[inst.feasible].min())
            optimum_states = set(np.flatnonzero(inst.feasible & (inst.costs <= optimum + 1e-9)))
            optimum_frequency = sum(count for state, count in zip(states, count_values)
                                    if int(state) in optimum_states) / 1024.0
            postprocess_s = time.perf_counter() - t0
            ops = compiled.count_ops()
            rows.append({
                "task": task, "n_vars": 6, "seed": seed,
                "backend_snapshot": backend.name, "shots": 1024,
                "selected_cost": float(inst.costs[selected]), "optimum": optimum,
                "optimal_return": abs(inst.costs[selected] - optimum) <= 1e-9,
                "selected_feasible": bool(inst.feasible[selected]),
                "observed_optimum_frequency": optimum_frequency,
                "ideal_optimum_probability": fitted["success_prob"],
                "compiled_cx": int(ops.get("cx", 0)),
                "compiled_depth": int(compiled.depth()),
                "build_instance_s": build_instance_s,
                "fit_statevector_s": fitted["runtime_s"],
                "build_circuit_s": build_circuit_s,
                "transpile_s": transpile_s, "noisy_sample_s": sample_s,
                "postprocess_s": postprocess_s,
                "local_pipeline_s": (build_instance_s + fitted["runtime_s"]
                                     + build_circuit_s + transpile_s + sample_s
                                     + postprocess_s),
                "evidence": "archived-calibration noisy emulation",
            })
            print(task, seed, optimum_frequency, ops.get("cx", 0), flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "review_backend_emulation.csv", index=False)
    (RESULTS / "review_backend_emulation_manifest.json").write_text(json.dumps({
        "backend": backend.name, "backend_qubits": backend.num_qubits,
        "qiskit": qiskit_version, "qiskit_aer": aer_version,
        "qiskit_ibm_runtime": runtime_version, "python": platform.python_version(),
        "shots": 1024, "optimization_level": 3, "seed_transpiler": 20260902,
        "evidence": "local Aer execution using an archived IBM calibration snapshot",
        "not_measured": ["live QPU output", "provider queue", "provider network latency"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
