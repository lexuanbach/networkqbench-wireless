#!/usr/bin/env python3
"""Compile complete small-instance cost oracles and one-hot controls.

The existing resource table compiles only pairwise interactions.  This script
adds exact diagonal synthesis of the *entire enumerated objective*, including
the nonlinear positive-part overflow penalties.  Exact diagonal synthesis is
exponential and is intentionally reported as a complete small-instance upper
path rather than a scalable encoding.  The gap to the pairwise circuit makes
the previous lower-bound status explicit.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, __version__ as qiskit_version
from qiskit.circuit.library import DiagonalGate, StatePreparation
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager

from compile_resources import build_circuit, interactions, topology_edges
from networkqbench import GENERATORS, _normalized_cost, bit_table


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def exact_cost_circuit(inst, onehot: bool) -> QuantumCircuit:
    n = inst.n_vars
    circuit = QuantumCircuit(n)
    if onehot:
        bits = bit_table(n)
        if inst.task == "placement":
            groups, width = int(inst.metadata["functions"]), int(inst.metadata["nodes"])
        elif inst.task == "routing":
            groups, width = int(inst.metadata["commodities"]), int(inst.metadata["paths"])
        else:
            raise ValueError(inst.task)
        support = np.ones(1 << n, dtype=bool)
        for group in range(groups):
            support &= bits[:, group * width:(group + 1) * width].sum(axis=1) == 1
        initial = np.zeros(1 << n, dtype=np.complex128)
        initial[support] = 1.0 / np.sqrt(support.sum())
        circuit.append(StatePreparation(initial), range(n))
    else:
        circuit.h(range(n))
    phases = np.exp(-1j * 0.2 * _normalized_cost(inst.costs))
    circuit.append(DiagonalGate(phases), range(n))
    if onehot:
        for group in range(groups):
            qubits = list(range(group * width, (group + 1) * width))
            pairs = [(qubits[0], qubits[1])] if width == 2 else [
                (qubits[k], qubits[(k + 1) % width]) for k in range(width)]
            for i, j in pairs:
                circuit.rxx(0.4, i, j)
                circuit.ryy(0.4, i, j)
    else:
        circuit.rx(0.4, range(n))
    circuit.measure_all()
    return circuit


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, generator in GENERATORS.items():
        for size in (6, 8, 12):
            inst = generator(size, 30_000)
            variants = {
                "quadratic_lower_bound": build_circuit(inst.n_vars, interactions(inst)),
                "exact_full_x": exact_cost_circuit(inst, onehot=False),
            }
            if task in ("placement", "routing"):
                variants["exact_full_onehot"] = exact_cost_circuit(inst, onehot=True)
            for variant, circuit in variants.items():
                for topology in ("line16", "grid4x4"):
                    backend = GenericBackendV2(
                        num_qubits=16,
                        coupling_map=CouplingMap(topology_edges(topology)),
                        basis_gates=["rz", "sx", "x", "cx"], seed=20260902)
                    pm = generate_preset_pass_manager(
                        optimization_level=2, backend=backend, seed_transpiler=20260902)
                    start = time.perf_counter(); compiled = pm.run(circuit)
                    elapsed = time.perf_counter() - start
                    ops = compiled.count_ops()
                    rows.append({
                        "task": task, "n_vars": size, "variant": variant,
                        "topology": topology, "cx": int(ops.get("cx", 0)),
                        "single_qubit": int(sum(ops.get(g, 0) for g in ("rz", "sx", "x"))),
                        "depth": int(compiled.depth()), "size": int(compiled.size()),
                        "width": int(compiled.num_qubits), "transpile_s": elapsed,
                        "objective_scope": ("pairwise interactions only" if variant == "quadratic_lower_bound"
                                            else "complete enumerated objective including overflow"),
                        "initialization": ("uniform one-hot state"
                                           if variant.endswith("onehot")
                                           else "uniform computational state"),
                    })
                    print(task, size, variant, topology, ops.get("cx", 0),
                          compiled.depth(), flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "review_full_circuit_resources.csv", index=False)
    (RESULTS / "review_full_circuit_manifest.json").write_text(json.dumps({
        "qiskit": qiskit_version, "python": platform.python_version(),
        "optimization_level": 2, "seed_transpiler": 20260902,
        "basis_gates": ["rz", "sx", "x", "cx"],
        "targets": ["16-qubit bidirectional line", "4x4 bidirectional grid"],
        "complete_oracle": "Qiskit DiagonalGate over the full enumerated cost vector",
        "interpretation": "exact small-instance synthesis; exponential upper path, not scalable",
        "evidence": "deterministic GenericBackendV2 compilation; no hardware execution",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
