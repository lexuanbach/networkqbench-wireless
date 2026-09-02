#!/usr/bin/env python3
"""Targeted experiments requested by the Fable review.

The script is deterministic and separates exact statevector quantities from
sample-derived deployment quantities.  It adds no hardware claims: the public
QuantumQueue replay is an empirical queue envelope around simulated QAOA.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from networkqbench import (
    GENERATORS,
    _normalized_cost,
    bit_table,
    qaoa_probabilities_depth,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "submission" / "03_experiments" / "results"
QUEUE_REPO = ROOT.parent / "02-QPU-Aware-CoOptimization" / "benchmark" / "data" / "QuantumQueue"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def optimize_depth(inst, depth: int, previous: np.ndarray | None = None):
    """Multistart exact-expectation fit with an explicit layerwise start."""
    norm = _normalized_cost(inst.costs)
    calls = 0

    def objective(theta):
        nonlocal calls
        calls += 1
        probabilities = qaoa_probabilities_depth(inst.costs, inst.n_vars, theta)
        return float(np.dot(probabilities, norm))

    starts = []
    if previous is not None:
        old_depth = len(previous) // 2
        starts.append(np.concatenate([
            previous[:old_depth], [previous[old_depth - 1]],
            previous[old_depth:], [0.5 * previous[-1]],
        ]))
    rng = np.random.default_rng(81_001 + 31 * inst.seed + depth)
    for _ in range(3 if depth > 1 else 2):
        starts.append(np.concatenate([
            rng.uniform(0.0, 2.0 * np.pi, depth),
            rng.uniform(0.0, np.pi, depth),
        ]))
    if depth == 1:
        for gamma in np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False):
            for beta in np.linspace(0.0, np.pi, 7, endpoint=False):
                starts.append(np.array([gamma, beta]))

    start_time = time.perf_counter()
    best_theta = min(starts, key=objective)
    best_value = objective(best_theta)
    # Refine the best deterministic/grid start and every non-grid multistart.
    refine = [best_theta] + starts[:(4 if depth > 1 else 2)]
    for x0 in refine:
        result = minimize(objective, x0, method="Nelder-Mead", options={
            "maxfev": 180 * depth, "xatol": 1e-4, "fatol": 1e-7,
        })
        if result.fun < best_value:
            best_value, best_theta = float(result.fun), result.x
    elapsed = time.perf_counter() - start_time
    probabilities = qaoa_probabilities_depth(inst.costs, inst.n_vars, best_theta)
    feasible_optimum = float(inst.costs[inst.feasible].min())
    optimum_mask = inst.feasible & (inst.costs <= feasible_optimum + 1e-9)
    optimum_probability = float(probabilities[optimum_mask].sum())
    return best_theta, {
        "expected_normalized_cost": best_value,
        "optimum_probability": optimum_probability,
        "hit_probability_1024": float(-math.expm1(1024 * math.log1p(-min(optimum_probability, 1 - 1e-16))))
        if optimum_probability > 0 else 0.0,
        "optimizer_calls": calls,
        "optimizer_s": elapsed,
    }


def depth_sensitivity():
    rows = []
    for task in ("channel", "placement", "routing"):
        for size in (6, 9, 12):
            for seed in range(6):
                inst = GENERATORS[task](size, seed)
                previous = None
                for depth in (1, 2, 3):
                    theta, metrics = optimize_depth(inst, depth, previous)
                    previous = theta
                    rows.append({
                        "task": task, "size_label": size, "n_vars": inst.n_vars,
                        "seed": seed, "mixer": "standard_X", "depth_p": depth,
                        **metrics,
                    })
                print(f"depth {task} n={size} seed={seed}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "fable_depth_sensitivity.csv", index=False)
    return result


def circular_distance(a, b, period):
    return np.abs((a - b + period / 2.0) % period - period / 2.0)


def circular_mean(values, period):
    phase = np.asarray(values) * (2.0 * np.pi / period)
    return float(np.angle(np.mean(np.exp(1j * phase))) % (2.0 * np.pi) * period / (2.0 * np.pi))


def circular_median(values, period):
    values = np.asarray(values)
    return float(values[np.argmin([circular_distance(values, x, period).sum() for x in values])])


def transfer_aggregation():
    raw = pd.read_csv(RESULTS / "raw_results.csv")
    qaoa = raw[(raw.algorithm == "qaoa") & (raw.profile == "nominal")]
    rows = []
    for (task, size), group in qaoa.groupby(["task", "size_label"]):
        train = group[group.seed < 15]
        if train.empty:
            continue
        angles = train[["gamma", "beta"]].dropna().to_numpy()
        mean = np.array([circular_mean(angles[:, 0], 2 * np.pi),
                         circular_mean(angles[:, 1], np.pi)])
        median = np.array([circular_median(angles[:, 0], 2 * np.pi),
                           circular_median(angles[:, 1], np.pi)])
        distances = np.array([
            circular_distance(angles[:, 0], a[0], 2 * np.pi).sum()
            + circular_distance(angles[:, 1], a[1], np.pi).sum()
            for a in angles
        ])
        medoid = angles[int(np.argmin(distances))]
        for seed in range(15, 30):
            inst = GENERATORS[task](int(size), seed)
            optimum = float(inst.costs[inst.feasible].min())
            mask = inst.feasible & (inst.costs <= optimum + 1e-9)
            norm = _normalized_cost(inst.costs)
            for method, theta in (("circular_mean", mean),
                                  ("component_circular_median", median),
                                  ("observed_circular_medoid", medoid)):
                probability = qaoa_probabilities_depth(inst.costs, inst.n_vars, theta)
                p_opt = float(probability[mask].sum())
                rows.append({
                    "task": task, "size_label": int(size), "seed": seed,
                    "aggregation": method, "optimum_probability": p_opt,
                    "hit_probability_1024": float(1 - (1 - p_opt) ** 1024),
                    "expected_normalized_cost": float(np.dot(probability, norm)),
                })
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "fable_transfer_aggregation.csv", index=False)
    return result


def penalty_components(inst):
    bits = bit_table(inst.n_vars)
    if inst.task == "placement":
        functions, nodes = inst.metadata["functions"], inst.metadata["nodes"]
        x = bits.reshape(-1, functions, nodes)
        demand = inst.metadata["demands"]
        base = (x * inst.metadata["energy"][None, :, :]).sum(axis=(1, 2))
        latency = inst.metadata["latency"]
        for function in range(functions - 1):
            base += np.einsum("bi,ij,bj->b", x[:, function, :], latency,
                              x[:, function + 1, :])
        load = (x * demand[None, :, None]).sum(axis=1)
        violation = ((x.sum(axis=2) - 1) ** 2).sum(axis=1).astype(float)
        violation += (np.maximum(load - inst.metadata["capacities"][None, :], 0) ** 2).sum(axis=1)
        return base, violation, 15.0
    commodities, paths = inst.metadata["commodities"], inst.metadata["paths"]
    x = bits.reshape(-1, commodities, paths)
    base = (x * inst.metadata["path_latency"][None, :, :]).sum(axis=(1, 2))
    load = np.einsum("bcp,cpe,c->be", x, inst.metadata["incidence"].astype(float),
                     inst.metadata["demand"])
    violation = ((x.sum(axis=2) - 1) ** 2).sum(axis=1).astype(float)
    violation += (np.maximum(load - inst.metadata["capacities"][None, :], 0) ** 2).sum(axis=1)
    return base, violation, 20.0


def penalty_sensitivity():
    rows = []
    for task in ("placement", "routing"):
        for size in (6, 8, 9, 12, 14, 16):
            seeds = range(10) if size <= 12 else range(30_000, 30_005)
            for seed in seeds:
                inst = GENERATORS[task](size, seed)
                base, violation, nominal = penalty_components(inst)
                feasible_best = float(base[inst.feasible].min())
                for scale in (0.10, 0.25, 0.50, 1.0, 2.0, 4.0):
                    penalized = base + scale * nominal * violation
                    ground = float(penalized.min())
                    ground_mask = penalized <= ground + 1e-9
                    rows.append({
                        "task": task, "size_label": size, "seed": seed,
                        "penalty_scale": scale,
                        "any_feasible_ground_state": bool(np.any(ground_mask & inst.feasible)),
                        "all_ground_states_feasible": bool(np.all(inst.feasible[ground_mask])),
                        "feasible_objective": feasible_best,
                        "penalized_ground_energy": ground,
                    })
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "fable_penalty_sensitivity.csv", index=False)
    return result


def staleness_sensitivity():
    raw = pd.read_csv(RESULTS / "raw_results.csv")
    use = raw[((raw.algorithm == "qaoa") & (raw.profile == "nominal"))
              | (raw.algorithm == "local_search")].copy()
    rows = []
    for multiplier in (0.0, 0.5, 1.0, 2.0, 5.0):
        use["rescored_utility"] = -(use.gap + 5.0 * (~use.feasible).astype(float)
                                     + multiplier * use.volatility * use.delay_s
                                     + 2.0 * use.deadline_miss.astype(float))
        pivot = use.pivot_table(index=["task", "size_label", "seed"],
                                columns="algorithm", values="rescored_utility")
        for task, group in pivot.groupby(level=0):
            difference = group.qaoa - group.local_search
            rows.append({"task": task, "staleness_multiplier": multiplier,
                         "paired_instances": len(difference),
                         "mean_qaoa_minus_local_utility": float(difference.mean()),
                         "qaoa_win_rate": float((difference > 0).mean())})
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "fable_staleness_sensitivity.csv", index=False)
    return result


def load_quantumqueue():
    frames, files = [], []
    if not QUEUE_REPO.exists():
        raise FileNotFoundError(f"clone https://github.com/rgokulsm/QuantumQueue at {QUEUE_REPO}")
    for path in sorted(QUEUE_REPO.glob("*.csv")):
        frame = pd.read_csv(path)
        if {"queue_time", "run_time", "status", "machine"}.issubset(frame.columns):
            frames.append(frame)
            files.append({"path": path.name, "sha256": sha256(path), "rows": len(frame)})
    data = pd.concat(frames, ignore_index=True)
    status = data.status.astype(str).str.upper()
    machine = data.machine.astype(str).str.lower()
    data = data[status.str.contains("DONE", na=False)
                & ~machine.str.contains("simulator", na=False)].copy()
    data["queue_minutes"] = pd.to_numeric(data.queue_time, errors="coerce")
    data["run_minutes"] = pd.to_numeric(data.run_time, errors="coerce")
    data = data[(data.queue_minutes >= 0) & (data.run_minutes > 0)]
    try:
        commit = subprocess.check_output(["git", "-C", str(QUEUE_REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    provenance = {
        "source": "https://github.com/rgokulsm/QuantumQueue",
        "companion": "https://arxiv.org/abs/2203.13121",
        "commit": commit, "files": files, "filter": "DONE, non-simulator, nonnegative queue, positive runtime",
        "unit_check": "repository capture notebook divides timedelta.total_seconds() by 60",
        "retained_rows": len(data),
    }
    return data, provenance


def queue_replay():
    data, provenance = load_quantumqueue()
    queue_seconds = data.queue_minutes.to_numpy(dtype=float) * 60.0
    quantiles = np.quantile(queue_seconds, np.linspace(0.0, 1.0, 101))
    raw = pd.read_csv(RESULTS / "raw_results.csv")
    qaoa = raw[(raw.algorithm == "qaoa") & (raw.profile == "nominal")].copy()
    local = raw[raw.algorithm == "local_search"].copy()
    rows = []
    for record in qaoa.itertuples():
        for queue_s in quantiles:
            delay = max(0.0, record.delay_s - 0.5) + queue_s
            missed = delay > record.deadline_s
            utility = -(record.gap + 5.0 * (not record.feasible)
                        + record.volatility * delay + 2.0 * missed)
            rows.append({"task": record.task, "size_label": record.size_label,
                         "seed": record.seed, "algorithm": "qaoa",
                         "queue_s": queue_s, "delay_s": delay,
                         "deadline_miss": missed, "utility": utility})
    for record in local.itertuples():
        rows.append({"task": record.task, "size_label": record.size_label,
                     "seed": record.seed, "algorithm": "local_search",
                     "queue_s": 0.0, "delay_s": record.delay_s,
                     "deadline_miss": record.deadline_miss, "utility": record.utility})
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "fable_quantumqueue_replay.csv", index=False)
    stats = {f"queue_minutes_q{q}": float(np.quantile(data.queue_minutes, q / 100))
             for q in (10, 25, 50, 75, 90, 95, 99)}
    stats.update({"fraction_over_120_minutes": float((data.queue_minutes > 120).mean()),
                  "fraction_over_1440_minutes": float((data.queue_minutes > 1440).mean())})
    provenance["statistics"] = stats
    (RESULTS / "fable_quantumqueue_provenance.json").write_text(json.dumps(provenance, indent=2))
    return result, provenance


def timing_frontier_and_mip():
    deadlines = []
    for task in GENERATORS:
        for size in (6, 8, 9, 12):
            deadlines.extend(GENERATORS[task](size, seed).deadline_s for seed in range(30))
    d10 = float(np.quantile(deadlines, 0.10))
    profiles = {
        "optimistic": (0.0005, 0.00001, 0.002),
        "nominal": (0.005, 0.00005, 0.020),
        "stressed": (0.020, 0.00020, 0.100),
    }
    frontier = []
    for profile, (per_eval, per_shot, rtt) in profiles.items():
        for shots in (64, 256, 1024):
            per_round = per_eval + rtt + shots * per_shot
            for queue_s in (0.0, 0.01, 0.05, 0.10, 0.50):
                rounds = max(0, math.floor((d10 - queue_s) / per_round))
                frontier.append({"profile": profile, "shots": shots, "queue_s": queue_s,
                                 "deadline_q10_s": d10, "max_rounds_for_90pct_ontime": rounds})
    frontier = pd.DataFrame(frontier)
    frontier.to_csv(RESULTS / "fable_timing_frontier.csv", index=False)

    frames = [pd.read_csv(RESULTS / "clearaccept_baselines.csv"),
              pd.read_csv(RESULTS / "clearaccept_hardness.csv")]
    data = pd.concat(frames, ignore_index=True)
    data = data[(data.algorithm == "milp") & data.task.isin(["placement", "routing"])
                & data.n_vars.isin([12, 14, 16])].copy()
    data["deadline_s"] = [GENERATORS[row.task](int(row.size_label), int(row.seed)).deadline_s
                          for row in data.itertuples()]
    data["ontime"] = data.complete_local_s <= data.deadline_s
    for limit in (0.05, 0.10, 0.50):
        data[f"within_{int(limit * 1000)}ms"] = data.complete_local_s <= limit
    summary = data.groupby(["task", "n_vars"], as_index=False).agg(
        instances=("seed", "size"), optimal_rate=("gap", lambda x: float((x <= 1e-12).mean())),
        median_complete_s=("complete_local_s", "median"), p95_complete_s=("complete_local_s", lambda x: float(np.quantile(x, .95))),
        deadline_coverage=("ontime", "mean"), within_50ms=("within_50ms", "mean"),
        within_100ms=("within_100ms", "mean"), within_500ms=("within_500ms", "mean"))
    summary.to_csv(RESULTS / "fable_mip_deadline_coverage.csv", index=False)
    return frontier, summary


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    cached = RESULTS / "fable_depth_sensitivity.csv"
    depth = pd.read_csv(cached) if cached.exists() else depth_sensitivity()
    cached = RESULTS / "fable_transfer_aggregation.csv"
    transfer = pd.read_csv(cached) if cached.exists() else transfer_aggregation()
    cached = RESULTS / "fable_penalty_sensitivity.csv"
    penalty = pd.read_csv(cached) if cached.exists() else penalty_sensitivity()
    cached = RESULTS / "fable_staleness_sensitivity.csv"
    stale = pd.read_csv(cached) if cached.exists() else staleness_sensitivity()
    replay, provenance = queue_replay()
    frontier, mip = timing_frontier_and_mip()
    manifest = {
        "script_sha256": sha256(Path(__file__)),
        "depth_rows": len(depth), "transfer_rows": len(transfer),
        "penalty_rows": len(penalty), "staleness_rows": len(stale),
        "queue_replay_rows": len(replay), "queue_source_commit": provenance["commit"],
        "frontier_rows": len(frontier), "mip_groups": len(mip),
        "claims": {"qaoa": "exact statevector simulation", "queue": "public trace replay",
                   "mip_time": "measured local wall clock; not production controller latency"},
    }
    (RESULTS / "fable_experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
