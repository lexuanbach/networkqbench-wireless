#!/usr/bin/env python3
"""Deterministically transpile representative p=1 interaction circuits."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, __version__ as qiskit_version
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager

from networkqbench import GENERATORS


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def interactions(inst) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    if inst.task == "channel":
        # Imbalance makes the logical QUBO complete even when interference is sparse.
        pairs.update((i, j) for i in range(inst.n_vars)
                     for j in range(i + 1, inst.n_vars))
    elif inst.task == "placement":
        functions = int(inst.metadata["functions"]); nodes = int(inst.metadata["nodes"])
        for f in range(functions):
            pairs.update((f * nodes + i, f * nodes + j)
                         for i in range(nodes) for j in range(i + 1, nodes))
        for f in range(functions - 1):
            pairs.update((f * nodes + i, (f + 1) * nodes + j)
                         for i in range(nodes) for j in range(nodes))
        for node in range(nodes):
            pairs.update((f * nodes + node, g * nodes + node)
                         for f in range(functions) for g in range(f + 1, functions))
    else:
        commodities = int(inst.metadata["commodities"]); paths = int(inst.metadata["paths"])
        incidence = np.asarray(inst.metadata["incidence"], dtype=bool)
        for c in range(commodities):
            pairs.update((c * paths + i, c * paths + j)
                         for i in range(paths) for j in range(i + 1, paths))
        for c in range(commodities):
            for p in range(paths):
                for d in range(c, commodities):
                    start = p + 1 if d == c else 0
                    for q in range(start, paths):
                        if np.any(incidence[c, p] & incidence[d, q]):
                            pairs.add((c * paths + p, d * paths + q))
    return pairs


def topology_edges(kind: str, n: int = 16) -> list[tuple[int, int]]:
    if kind == "line16":
        undirected = [(i, i + 1) for i in range(n - 1)]
    elif kind == "grid4x4":
        undirected = []
        for row in range(4):
            for col in range(4):
                q = 4 * row + col
                if col < 3: undirected.append((q, q + 1))
                if row < 3: undirected.append((q, q + 4))
    else:
        raise ValueError(kind)
    return undirected + [(b, a) for a, b in undirected]


def build_circuit(n: int, pairs: set[tuple[int, int]]) -> QuantumCircuit:
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    for i, j in sorted(pairs):
        circuit.rzz(0.2, i, j)
    circuit.rx(0.4, range(n))
    circuit.measure_all()
    return circuit


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, generator in GENERATORS.items():
        for size in (6, 12, 16):
            inst = generator(size, 30_000)
            pairs = interactions(inst)
            circuit = build_circuit(inst.n_vars, pairs)
            for topology in ("line16", "grid4x4"):
                coupling = CouplingMap(topology_edges(topology))
                backend = GenericBackendV2(
                    num_qubits=16, coupling_map=coupling,
                    basis_gates=["rz", "sx", "x", "cx"], seed=20260901)
                started = time.perf_counter()
                pm = generate_preset_pass_manager(
                    optimization_level=3, backend=backend, seed_transpiler=20260901)
                compiled = pm.run(circuit)
                elapsed = time.perf_counter() - started
                counts = compiled.count_ops()
                logical_cx_floor = 2 * len(pairs)
                rows.append({
                    "task": task, "n_vars": inst.n_vars, "topology": topology,
                    "logical_interactions": len(pairs),
                    "logical_cx_floor": logical_cx_floor,
                    "compiled_cx": int(counts.get("cx", 0)),
                    "compiled_depth": int(compiled.depth()),
                    "compiled_size": int(compiled.size()),
                    "physical_width": int(compiled.num_qubits),
                    "routing_cx_overhead": int(counts.get("cx", 0)) - logical_cx_floor,
                    "transpile_s": elapsed,
                    "optimization_level": 3,
                })
                print(task, size, topology, counts.get("cx", 0), compiled.depth(), flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "clearaccept_compiled_resources.csv", index=False)
    (RESULTS / "clearaccept_compile_manifest.json").write_text(json.dumps({
        "qiskit": qiskit_version,
        "python": platform.python_version(),
        "seed_transpiler": 20260901,
        "optimization_level": 3,
        "targets": ["16-qubit bidirectional line", "4x4 bidirectional grid"],
        "basis_gates": ["rz", "sx", "x", "cx"],
        "evidence": "compiled_estimate",
        "evidence_description": "deterministic compilation on GenericBackendV2; not hardware execution",
        "encoding_scope": "quadratic interaction circuit; nonlinear overflow/slack logic omitted",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
