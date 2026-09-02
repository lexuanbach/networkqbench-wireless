#!/usr/bin/env python3
"""Review-driven constraint-aware and closed-loop experiments.

This script adds two evidence layers requested by the simulated reviews:

1. a one-hot-preserving alternating-operator QAOA control for placement and
   routing, evaluated against the standard X-mixer on identical instances;
2. a closed-loop 5G channel-allocation replay across multiple base stations,
   including a deployable rolling incumbent and an explicit retain-previous
   fallback after a missed decision deadline.

All quantum probabilities are exact statevector simulations.  QPU service
delay in the closed loop is replayed from public queue records and is not a
hardware measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from networkqbench import (
    GENERATORS,
    Instance,
    _normalized_cost,
    bit_table,
    exact_solver,
    milp_solver,
    qaoa_probabilities,
    qaoa_solver,
    task_specific_solver,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
NETDATA = ROOT / "data" / "netdata" / "Performance_5G_Weekday.csv"
QUEUE_REPLAY = ROOT / "results" / "fable_quantumqueue_replay.csv"


def assignment_groups(inst: Instance) -> list[list[int]]:
    if inst.task == "placement":
        groups, width = int(inst.metadata["functions"]), int(inst.metadata["nodes"])
    elif inst.task == "routing":
        groups, width = int(inst.metadata["commodities"]), int(inst.metadata["paths"])
    else:
        raise ValueError("one-hot mixer applies to placement and routing")
    return [list(range(group * width, (group + 1) * width)) for group in range(groups)]


def _apply_xy_pair(state: np.ndarray, beta: float, i: int, j: int) -> None:
    """Apply an excitation-preserving pair rotation in place."""
    indexes = np.arange(len(state), dtype=np.uint32)
    select = (((indexes >> i) & 1) == 0) & (((indexes >> j) & 1) == 1)
    left = indexes[select].astype(int)
    right = (indexes[select] ^ (1 << i) ^ (1 << j)).astype(int)
    a, b = state[left].copy(), state[right].copy()
    c, s = math.cos(beta), -1j * math.sin(beta)
    state[left] = c * a + s * b
    state[right] = s * a + c * b


def onehot_probabilities(inst: Instance, gamma: float, beta: float) -> np.ndarray:
    groups = assignment_groups(inst)
    bits = bit_table(inst.n_vars)
    exact_one = np.ones(len(bits), dtype=bool)
    for group in groups:
        exact_one &= bits[:, group].sum(axis=1) == 1
    state = np.zeros(1 << inst.n_vars, dtype=np.complex128)
    state[exact_one] = 1.0 / math.sqrt(int(exact_one.sum()))
    state *= np.exp(-1j * gamma * _normalized_cost(inst.costs))
    for group in groups:
        pairs = [(group[0], group[1])] if len(group) == 2 else [
            (group[k], group[(k + 1) % len(group)]) for k in range(len(group))]
        for i, j in pairs:
            _apply_xy_pair(state, beta, i, j)
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def onehot_qaoa(inst: Instance, rng: np.random.Generator, shots: int = 1024) -> dict:
    start = time.perf_counter()
    norm = _normalized_cost(inst.costs)
    calls = 0

    def objective(theta):
        nonlocal calls
        calls += 1
        return float(np.dot(onehot_probabilities(inst, theta[0], theta[1]), norm))

    grid = [np.array([g, b]) for g in np.linspace(0, 2 * np.pi, 7, endpoint=False)
            for b in np.linspace(0, np.pi, 7, endpoint=False)]
    start_theta = min(grid, key=objective)
    base = objective(start_theta)
    opt = minimize(objective, start_theta, method="Nelder-Mead",
                   options={"maxiter": 40, "xatol": 1e-3, "fatol": 1e-5})
    theta = opt.x if opt.fun < base else start_theta
    probs = onehot_probabilities(inst, float(theta[0]), float(theta[1]))
    samples = rng.choice(len(probs), size=shots, p=probs)
    valid = samples[inst.feasible[samples]]
    chosen = int(valid[np.argmin(inst.costs[valid])]) if len(valid) \
        else int(samples[np.argmin(inst.costs[samples])])
    optimum = float(inst.costs[inst.feasible].min())
    optimal = inst.feasible & (inst.costs <= optimum + 1e-9)
    groups = assignment_groups(inst)
    mixer_pairs = sum(1 if len(group) == 2 else len(group) for group in groups)
    return {
        "solution": chosen, "cost": float(inst.costs[chosen]),
        "feasible": bool(inst.feasible[chosen]),
        "runtime_s": time.perf_counter() - start, "evals": calls,
        "success_prob": float(probs[optimal].sum()),
        "feasible_mass": float(probs[inst.feasible].sum()),
        "expected_norm_cost": float(np.dot(probs, norm)),
        "gamma": float(theta[0]), "beta": float(theta[1]),
        "logical_xy_pairs": mixer_pairs,
    }


def constraint_aware_experiment(seeds: int) -> pd.DataFrame:
    rows = []
    for task in ("placement", "routing"):
        for size in (6, 8, 9, 12):
            for seed in range(seeds):
                inst = GENERATORS[task](size, seed)
                optimum = exact_solver(inst)["cost"]
                standard = qaoa_solver(inst, np.random.default_rng(
                    61_000_000 + size * 1000 + seed))
                standard_probs = qaoa_probabilities(
                    inst.costs, inst.n_vars, standard["gamma"], standard["beta"])
                controlled = onehot_qaoa(inst, np.random.default_rng(
                    62_000_000 + size * 1000 + seed))
                for algorithm, result, probs in (
                    ("standard_x", standard, standard_probs),
                    ("onehot_xy", controlled, None),
                ):
                    feasible_mass = result.get("feasible_mass")
                    if feasible_mass is None:
                        feasible_mass = float(probs[inst.feasible].sum())
                    rows.append({
                        "task": task, "size_label": size, "n_vars": inst.n_vars,
                        "seed": seed, "algorithm": algorithm,
                        "optimum": optimum, "cost": result["cost"],
                        "optimal": abs(result["cost"] - optimum) <= 1e-9,
                        "sample_feasible": result["feasible"],
                        "feasible_mass": feasible_mass,
                        "optimum_mass": result["success_prob"],
                        "hit_probability_1024": 1.0 - (1.0 - result["success_prob"]) ** 1024,
                        "expected_norm_cost": result.get("expected_norm_cost", np.nan),
                        "optimizer_s": result["runtime_s"], "evals": result["evals"],
                        "logical_xy_pairs": result.get("logical_xy_pairs", 0),
                    })
                print(f"onehot {task} n={size} seed={seed}", flush=True)
    return pd.DataFrame(rows)


def dynamic_channel_instance(load: np.ndarray, station: int, window: int,
                             deadline_s: float = 0.30) -> Instance:
    n = len(load)
    bits = bit_table(n)
    weights = interference_weights(load, station)
    same = bits[:, :, None] == bits[:, None, :]
    interference = (same * weights[None, :, :]).sum(axis=(1, 2))
    imbalance = (bits.sum(axis=1) - n / 2.0) ** 2
    return Instance(
        task="channel", size_label=n, n_vars=n, seed=station * 100 + window,
        costs=(interference + 0.25 * imbalance).astype(float),
        feasible=np.ones(1 << n, dtype=bool), volatility=0.05,
        deadline_s=deadline_s, metadata={"edges": int(np.count_nonzero(weights)),
                                         "weights": weights})


def interference_weights(load: np.ndarray, station: int) -> np.ndarray:
    """Station-specific sparse interference modulated by the current load."""
    n = len(load)
    rng = np.random.default_rng(67_000_000 + station)
    affinity = np.triu(rng.uniform(0.2, 1.0, size=(n, n)), 1)
    mask = np.triu(rng.random((n, n)) < 0.55, 1)
    # Keep a ring so every cell participates in the interference graph.
    for i in range(n):
        a, b = sorted((i, (i + 1) % n))
        mask[a, b] = True
    weights = affinity * mask * (0.20 + load[:, None] + load[None, :])
    return np.triu(weights, 1)


def rolling_incumbent(inst: Instance, previous: int) -> dict:
    start = time.perf_counter(); state = int(previous); evals = 1
    for _ in range(2):
        improved = False
        for q in range(inst.n_vars):
            candidate = state ^ (1 << q); evals += 1
            if inst.costs[candidate] + 1e-12 < inst.costs[state]:
                state = candidate; improved = True
        if not improved:
            break
    return {"solution": state, "cost": float(inst.costs[state]),
            "feasible": True, "runtime_s": time.perf_counter() - start,
            "evals": evals}


def fixed_qaoa(inst: Instance, gamma: float, beta: float,
               rng: np.random.Generator, shots: int = 512) -> dict:
    start = time.perf_counter()
    probs = qaoa_probabilities(inst.costs, inst.n_vars, gamma, beta)
    samples = rng.choice(len(probs), size=shots, p=probs)
    state = int(samples[np.argmin(inst.costs[samples])])
    return {"solution": state, "cost": float(inst.costs[state]),
            "feasible": True, "runtime_s": time.perf_counter() - start,
            "evals": 1}


def congestion(load: np.ndarray, assignment: np.ndarray, station: int) -> np.ndarray:
    n = len(load); out = np.zeros(n)
    weights = interference_weights(load, station)
    weights = weights + weights.T
    for i in range(n):
        same = np.array([j != i and assignment[j] == assignment[i]
                         for j in range(n)])
        out[i] = float(weights[i, same].sum() / max(1, same.sum()))
    return out


def closed_loop_5g(stations: int) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(NETDATA, usecols=["Base Station ID", "Cell ID", "Timestamp",
                                        "PRB Usage Ratio (%)"])
    counts = raw.groupby("Base Station ID")["Cell ID"].nunique()
    eligible = sorted(map(int, counts[counts >= 8].index))[:stations]
    queue = pd.read_csv(QUEUE_REPLAY)
    queue_values = np.sort(queue[queue.algorithm == "qaoa"].queue_s.dropna().unique())
    rows = []
    methods = ("milp", "task_specific", "rolling", "qaoa_transfer")
    for station in eligible:
        block = raw[raw["Base Station ID"] == station].copy()
        cells = sorted(block["Cell ID"].unique())[:8]
        matrix = block[block["Cell ID"].isin(cells)].pivot(
            index="Timestamp", columns="Cell ID", values="PRB Usage Ratio (%)")
        matrix = matrix.reindex(sorted(matrix.index, key=lambda x: int(x[:2]) * 60 + int(x[3:])))
        trace = matrix.to_numpy(dtype=float) / 100.0
        previous = {m: int(sum((i % 2) << i for i in range(8))) for m in methods}
        feedback = {m: np.zeros(8) for m in methods}
        first_inst = dynamic_channel_instance(trace[0], station, 0)
        fitted = qaoa_solver(first_inst, np.random.default_rng(63_000_000 + station))
        for window in range(len(trace) - 1):
            for method in methods:
                current = np.clip(trace[window] + 0.30 * feedback[method], 0.0, 1.5)
                inst = dynamic_channel_instance(current, station, window)
                if method == "milp":
                    result = milp_solver(inst); delay = result["runtime_s"]
                    evidence = "measured_local"
                elif method == "task_specific":
                    result = task_specific_solver(inst); delay = result["runtime_s"]
                    evidence = "measured_local"
                elif method == "rolling":
                    result = rolling_incumbent(inst, previous[method]); delay = result["runtime_s"]
                    evidence = "measured_local"
                else:
                    result = fixed_qaoa(inst, fitted["gamma"], fitted["beta"],
                                        np.random.default_rng(64_000_000 + station * 100 + window))
                    replay_index = (station * 47 + window) % len(queue_values)
                    delay = float(queue_values[replay_index] + 0.20)
                    evidence = "trace_replayed_qpu"
                missed = delay > inst.deadline_s
                applied = previous[method] if missed else int(result["solution"])
                assignment = bit_table(8)[applied]
                next_exogenous = trace[window + 1]
                feedback[method] = congestion(current, assignment, station)
                next_effective = np.clip(next_exogenous + 0.30 * feedback[method], 0.0, 1.5)
                next_inst = dynamic_channel_instance(next_effective, station, window + 1)
                switches = int(np.count_nonzero(assignment != bit_table(8)[previous[method]]))
                rows.append({
                    "station": station, "window": window, "method": method,
                    "computed_solution": int(result["solution"]),
                    "applied_solution": applied, "fallback": "retain_previous" if missed else "none",
                    "delay_evidence": evidence, "delay_s": delay,
                    "deadline_s": inst.deadline_s, "deadline_miss": missed,
                    "next_cost": float(next_inst.costs[applied]),
                    "next_max_effective_load": float(next_effective.max()),
                    "next_mean_effective_load": float(next_effective.mean()),
                    "switches": switches,
                })
                previous[method] = applied
        print(f"closed-loop station={station}", flush=True)
    provenance = {
        "stations": eligible, "station_count": len(eligible), "cells_per_station": 8,
        "windows_per_station": 48, "feedback_coefficient": 0.30,
        "deadline_s": 0.30, "missed_deadline_fallback": "retain previous action",
        "netdata_source": "Performance_5G_Weekday.csv",
        "queue_source": "QuantumQueue replay quantiles",
        "qpu_evidence": "exact statevector plus trace-replayed queue; no QPU execution",
    }
    return pd.DataFrame(rows), provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--stations", type=int, default=24)
    parser.add_argument("--skip-onehot", action="store_true")
    parser.add_argument("--skip-closed-loop", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not args.skip_onehot:
        constraint_aware_experiment(args.seeds).to_csv(
            RESULTS / "review_constraint_aware.csv", index=False)
    provenance = None
    if not args.skip_closed_loop:
        closed, provenance = closed_loop_5g(args.stations)
        closed.to_csv(RESULTS / "review_closed_loop_5g.csv", index=False)
        (RESULTS / "review_closed_loop_provenance.json").write_text(
            json.dumps(provenance, indent=2))
    (RESULTS / "review_experiment_manifest.json").write_text(json.dumps({
        "created_unix": time.time(), "elapsed_s": time.time() - started,
        "python": platform.python_version(), "platform": platform.platform(),
        "constraint_seeds": args.seeds, "closed_loop_stations": args.stations,
        "evidence": "measured local CPU, exact statevector, and trace-replayed service; no QPU measurement",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
