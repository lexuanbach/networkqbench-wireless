#!/usr/bin/env python3
"""Revision experiments: hardness extension and Abilene trace replay.

The Abilene data are public 5-minute Internet2 backbone traffic matrices.
The replay constructs four-commodity, three-path decisions from measured OD
volumes and evaluates the selected routing decision on the next trace window.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from networkqbench import (
    GENERATORS,
    Instance,
    annealing_solver,
    bit_table,
    exact_solver,
    local_search_solver,
    mean_field_solver,
    milp_solver,
    operational_metrics,
    qaoa_solver,
    task_specific_solver,
    warm_start_qaoa_solver,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "abilene"


def solve_instance(inst: Instance, tag: str) -> list[dict]:
    base = 90_000_000 + 10_000 * inst.size_label + inst.seed
    exact = exact_solver(inst)
    optimum = exact["cost"]
    solved = {
        "exact": exact,
        "local_search": local_search_solver(inst, np.random.default_rng(base + 1)),
        "annealing": annealing_solver(inst, np.random.default_rng(base + 2)),
        "mean_field": mean_field_solver(inst, np.random.default_rng(base + 3)),
        "qaoa": qaoa_solver(inst, np.random.default_rng(base + 4)),
        "task_specific": task_specific_solver(inst),
        "milp": milp_solver(inst),
        "qaoa_warm": warm_start_qaoa_solver(inst, np.random.default_rng(base + 5)),
    }
    rows = []
    for algorithm, result in solved.items():
        profiles = ["optimistic", "nominal", "stressed"] \
            if algorithm.startswith("qaoa") else ["local"]
        for profile in profiles:
            op = operational_metrics(inst, result, optimum, algorithm, profile)
            rows.append({
                "experiment": tag,
                "task": inst.task,
                "size_label": inst.size_label,
                "n_vars": inst.n_vars,
                "seed": inst.seed,
                "algorithm": algorithm,
                "profile": profile,
                "solution": result["solution"],
                "cost": result["cost"],
                "optimum": optimum,
                "feasible": result["feasible"],
                "runtime_s": result["runtime_s"],
                "evals": result["evals"],
                "deadline_s": inst.deadline_s,
                "volatility": inst.volatility,
                "success_prob": result.get("success_prob", np.nan),
                **op,
            })
    return rows


def hardness_extension(seeds: int) -> pd.DataFrame:
    rows = []
    for task in GENERATORS:
        for size in (14, 16):
            for seed in range(seeds):
                inst = GENERATORS[task](size, 30_000 + seed)
                rows.extend(solve_instance(inst, "hardness"))
                print(f"hardness {task} n={size} seed={seed}", flush=True)
    return pd.DataFrame(rows)


def _load_abilene() -> tuple[nx.DiGraph, list[tuple[str, str]], np.ndarray]:
    graph = nx.DiGraph()
    for line in (DATA / "links").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        link, _, kind = line.split()
        if kind == "0":
            u, v = link.split(",")
            graph.add_edge(u, v)
    demands = []
    for line in (DATA / "demands").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        pair, _ = line.split()
        demands.append(tuple(pair.split(",")))
    with gzip.open(DATA / "X01.gz", "rt") as handle:
        raw = np.loadtxt(handle)
    return graph, demands, raw[:, ::5]


def _trace_instance(graph: nx.DiGraph, pairs: list[tuple[str, str]],
                    current: np.ndarray, nxt: np.ndarray, window: int):
    ranked = np.argsort(-current)
    chosen = []
    path_sets = []
    for idx in ranked:
        source, target = pairs[int(idx)]
        if source == target or current[idx] <= 0:
            continue
        try:
            paths = []
            for path in nx.shortest_simple_paths(graph, source, target):
                paths.append(path)
                if len(paths) == 3:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(paths) == 3:
            chosen.append(int(idx))
            path_sets.append(paths)
        if len(chosen) == 4:
            break
    if len(chosen) < 4:
        raise RuntimeError("Abilene topology did not yield four 3-path commodities")

    edges = list(graph.edges())
    edge_id = {e: i for i, e in enumerate(edges)}
    incidence = np.zeros((4, 3, len(edges)), dtype=bool)
    hops = np.zeros((4, 3), dtype=float)
    for c, paths in enumerate(path_sets):
        for p, path in enumerate(paths):
            used = list(zip(path[:-1], path[1:]))
            hops[c, p] = len(used)
            for edge in used:
                incidence[c, p, edge_id[edge]] = True

    scale = np.median(current[chosen]) + 1e-12
    demand = np.clip(current[chosen] / scale, 0.35, 3.0)
    next_demand = np.clip(nxt[chosen] / scale, 0.35, 3.0)
    reference = np.einsum("c,ce->e", demand, incidence[:, 0, :])
    capacities = np.maximum(1.10 * reference, 1.15 * np.max(demand))

    bits = bit_table(12).reshape(-1, 4, 3)
    exact_one = (bits.sum(axis=2) == 1).all(axis=1)
    load = np.einsum("bcp,cpe,c->be", bits, incidence.astype(float), demand)
    feasible = exact_one & (load <= capacities[None, :] + 1e-12).all(axis=1)
    weighted_hops = (bits * (hops * demand[:, None])[None, :, :]).sum(axis=(1, 2))
    overflow = np.maximum(load - capacities[None, :], 0.0)
    exact_pen = 25.0 * ((bits.sum(axis=2) - 1) ** 2).sum(axis=1)
    costs = weighted_hops + exact_pen + 50.0 * (overflow**2).sum(axis=1)
    # Convert the fractional change over one five-minute trace interval to a
    # per-second first-order staleness coefficient used by the benchmark.
    volatility = float(np.mean(np.abs(nxt[chosen] - current[chosen]))
                       / (np.mean(current[chosen]) + 1e-12) / 300.0)
    inst = Instance(
        task="abilene_routing",
        size_label=12,
        n_vars=12,
        seed=window,
        costs=costs,
        feasible=feasible,
        volatility=volatility,
        deadline_s=300.0,
        metadata={
            "edges": len(edges),
            "trace_window": window,
            "commodities": 4,
            "paths": 3,
            "incidence": incidence,
            "demand": demand,
            "capacities": capacities,
            "path_latency": hops * demand[:, None],
        },
    )
    return inst, incidence, capacities, next_demand


def trace_replay(windows: int) -> pd.DataFrame:
    graph, pairs, matrices = _load_abilene()
    indices = np.linspace(0, len(matrices) - 2, windows, dtype=int)
    rows = []
    for window in indices:
        inst, incidence, capacities, next_demand = _trace_instance(
            graph, pairs, matrices[window], matrices[window + 1], int(window))
        solved = solve_instance(inst, "abilene_trace")
        for row in solved:
            state = int(row["solution"])
            choice = bit_table(12)[state].reshape(4, 3)
            load = np.einsum("cp,cpe,c->e", choice, incidence.astype(float),
                             next_demand)
            utilization = load / capacities
            row["next_max_utilization"] = float(utilization.max())
            row["next_overflow"] = float(np.maximum(load - capacities, 0.0).sum())
            row["trace_interval_s"] = 300.0
            rows.append(row)
        print(f"Abilene trace window={window}", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-seeds", type=int, default=10)
    parser.add_argument("--trace-windows", type=int, default=30)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    hard = pd.DataFrame()
    trace = pd.DataFrame()
    if args.hard_seeds > 0:
        hard = hardness_extension(args.hard_seeds)
        hard.to_csv(RESULTS / "revision_hardness.csv", index=False)
    if args.trace_windows > 0:
        trace = trace_replay(args.trace_windows)
        trace.to_csv(RESULTS / "revision_abilene_trace.csv", index=False)
    print(f"wrote {len(hard)} hardness rows and {len(trace)} trace rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
