#!/usr/bin/env python3
"""Clear-accept experiments: practical baselines, repeated shots, and 5G replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from networkqbench import (
    GENERATORS,
    Instance,
    bit_table,
    exact_solver,
    milp_solver,
    operational_metrics,
    qaoa_probabilities,
    qaoa_solver,
    task_specific_solver,
    warm_start_qaoa_solver,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
NETDATA = ROOT / "data" / "netdata" / "Performance_5G_Weekday.csv"
NETDATA_URL = ("https://media.githubusercontent.com/media/tsinghua-fib-lab/"
               "NetData/093e13ebcbee4f18a3fc4d6a6f3aeab9ac1283a3/"
               "Performance_5G_Weekday.csv")
NETDATA_COMMIT = "093e13ebcbee4f18a3fc4d6a6f3aeab9ac1283a3"


def _result_row(inst: Instance, algorithm: str, result: dict, construction_s: float,
                optimum: float, experiment: str) -> dict:
    profile = "nominal" if algorithm.startswith("qaoa") else "local"
    op = operational_metrics(inst, result, optimum, algorithm, profile)
    return {
        "experiment": experiment,
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
        "construction_s": construction_s,
        "solver_s": result["runtime_s"],
        "complete_local_s": construction_s + result["runtime_s"],
        "cost_vector_bytes": int(inst.costs.nbytes + inst.feasible.nbytes),
        "evals": result["evals"],
        "success_prob": result.get("success_prob", np.nan),
        "classical_seed_s": result.get("classical_seed_runtime_s", np.nan),
        **op,
    }


def practical_baselines(seeds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_rows, hard_rows = [], []
    for task, generator in GENERATORS.items():
        for size in (6, 8, 9, 12):
            for seed in range(seeds):
                t0 = time.perf_counter(); inst = generator(size, seed)
                construction_s = time.perf_counter() - t0
                optimum = exact_solver(inst)["cost"]
                methods = {
                    "task_specific": task_specific_solver(inst),
                    "milp": milp_solver(inst),
                    "qaoa_warm": warm_start_qaoa_solver(
                        inst, np.random.default_rng(81_000_000 + size * 1000 + seed)),
                }
                for name, result in methods.items():
                    main_rows.append(_result_row(inst, name, result, construction_s,
                                                 optimum, "clearaccept_main"))
                print(f"clearaccept main {task} n={size} seed={seed}", flush=True)
        for size in (14, 16):
            for seed in range(min(seeds, 10)):
                actual_seed = 30_000 + seed
                t0 = time.perf_counter(); inst = generator(size, actual_seed)
                construction_s = time.perf_counter() - t0
                optimum = exact_solver(inst)["cost"]
                methods = {
                    "task_specific": task_specific_solver(inst),
                    "milp": milp_solver(inst),
                    "qaoa_warm": warm_start_qaoa_solver(
                        inst, np.random.default_rng(82_000_000 + size * 1000 + seed)),
                }
                for name, result in methods.items():
                    hard_rows.append(_result_row(inst, name, result, construction_s,
                                                 optimum, "clearaccept_hardness"))
                print(f"clearaccept hard {task} n={size} seed={seed}", flush=True)
    return pd.DataFrame(main_rows), pd.DataFrame(hard_rows)


def repeated_shots(replicates: int) -> pd.DataFrame:
    raw = pd.read_csv(RESULTS / "raw_results.csv")
    q = raw[(raw.algorithm == "qaoa") & (raw.profile == "nominal")]
    rows = []
    for rec in q.itertuples():
        inst = GENERATORS[rec.task](int(rec.size_label), int(rec.seed))
        probs = qaoa_probabilities(inst.costs, inst.n_vars, float(rec.gamma), float(rec.beta))
        for shots in (64, 256, 1024):
            for replicate in range(replicates):
                rng = np.random.default_rng(83_000_000 + int(rec.size_label) * 100_000
                                            + int(rec.seed) * 1000 + shots + replicate)
                sampled = rng.choice(len(probs), size=shots, p=probs)
                valid = sampled[inst.feasible[sampled]]
                pick = int(valid[np.argmin(inst.costs[valid])]) if len(valid) \
                    else int(sampled[np.argmin(inst.costs[sampled])])
                rows.append({
                    "task": rec.task, "size_label": int(rec.size_label),
                    "seed": int(rec.seed), "shots": shots, "replicate": replicate,
                    "feasible": bool(inst.feasible[pick]),
                    "optimal": bool(inst.feasible[pick]
                                    and inst.costs[pick] <= float(rec.optimum) + 1e-9),
                    "gap": max(0.0, (float(inst.costs[pick]) - float(rec.optimum))
                               / (abs(float(rec.optimum)) + 1.0)),
                })
    return pd.DataFrame(rows)


def _load_netdata() -> tuple[pd.DataFrame, dict]:
    if not NETDATA.exists():
        raise FileNotFoundError(
            f"Download {NETDATA_URL} to {NETDATA}; raw data are not redistributed")
    digest = hashlib.sha256(NETDATA.read_bytes()).hexdigest()
    data = pd.read_csv(NETDATA, usecols=["Base Station ID", "Cell ID", "Timestamp",
                                        "PRB Usage Ratio (%)", "Traffic Volume (KByte)",
                                        "Number of Users"])
    for column in ("PRB Usage Ratio (%)", "Traffic Volume (KByte)", "Number of Users"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    counts = data.groupby("Base Station ID")["Cell ID"].nunique()
    candidates = counts[counts >= 3].index
    traffic = data[data["Base Station ID"].isin(candidates)].groupby(
        "Base Station ID")["Traffic Volume (KByte)"].mean().sort_values(ascending=False)
    stations = list(traffic.head(4).index)
    cells = []
    for station in stations:
        ranked = data[data["Base Station ID"] == station].groupby("Cell ID")[
            "Traffic Volume (KByte)"].mean().sort_values(ascending=False)
        cells.extend(ranked.head(3).index.tolist())
    subset = data[data["Cell ID"].isin(cells)].copy()
    info = {"sha256": digest, "source_url": NETDATA_URL, "source_commit": NETDATA_COMMIT,
            "selected_stations": [str(s) for s in stations],
            "selected_cells": [str(c) for c in cells]}
    return subset, info


def netdata_replay() -> tuple[pd.DataFrame, dict]:
    data, provenance = _load_netdata()
    prb = data.pivot_table(index="Timestamp", columns="Cell ID",
                           values="PRB Usage Ratio (%)", aggfunc="mean")
    traffic = data.pivot_table(index="Timestamp", columns="Cell ID",
                               values="Traffic Volume (KByte)", aggfunc="mean")
    common = sorted(set(prb.columns) & set(traffic.columns))
    prb = prb[common].dropna(); traffic = traffic[common].reindex(prb.index).dropna()
    common_times = sorted(set(prb.index) & set(traffic.index),
                          key=lambda value: tuple(map(int, value.split(":"))))
    prb = prb.loc[common_times]; traffic = traffic.loc[common_times]
    if len(common) != 12 or len(common_times) < 20:
        raise RuntimeError(f"Expected 12 complete cells and >=20 windows; got {len(common)}, {len(common_times)}")
    station_of = data.drop_duplicates("Cell ID").set_index("Cell ID")["Base Station ID"].to_dict()
    base = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            if station_of[common[i]] == station_of[common[j]]:
                base[i, j] = 1.0
    # Add a fixed sparse coordination graph from the first eight windows only.
    corr = prb.iloc[:8].corr().fillna(0.0).to_numpy()
    cross = [(corr[i, j], i, j) for i in range(12) for j in range(i + 1, 12)
             if base[i, j] == 0]
    for _, i, j in sorted(cross, reverse=True)[:12]:
        base[i, j] = 0.35

    bits = bit_table(12)
    rows = []
    for window in range(8, len(common_times) - 1):
        current = prb.iloc[window].to_numpy() / 100.0
        nxt = prb.iloc[window + 1].to_numpy() / 100.0
        weights = base * (0.25 + current[:, None] + current[None, :])
        same = bits[:, :, None] == bits[:, None, :]
        costs = (same * weights[None, :, :]).sum(axis=(1, 2)) \
            + 0.25 * (bits.sum(axis=1) - 6.0) ** 2
        inst = Instance(
            task="netdata_channel", size_label=12, n_vars=12, seed=window,
            costs=costs, feasible=np.ones(len(costs), dtype=bool),
            volatility=float(np.mean(np.abs(nxt - current)) / 1800.0),
            deadline_s=1800.0,
            metadata={"edges": int((base > 0).sum()), "weights": weights},
        )
        optimum = exact_solver(inst)["cost"]
        methods = {
            "task_specific": task_specific_solver(inst),
            "milp": milp_solver(inst),
            "qaoa": qaoa_solver(inst, np.random.default_rng(84_000_000 + window)),
            "qaoa_warm": warm_start_qaoa_solver(inst, np.random.default_rng(85_000_000 + window)),
        }
        next_weights = base * (0.25 + nxt[:, None] + nxt[None, :])
        for name, result in methods.items():
            assignment = bits[int(result["solution"])]
            cochannel = float(sum(next_weights[i, j] for i in range(12)
                                  for j in range(i + 1, 12)
                                  if assignment[i] == assignment[j]))
            row = _result_row(inst, name, result, 0.0, optimum, "netdata_5g")
            row.update({"timestamp": common_times[window],
                        "next_timestamp": common_times[window + 1],
                        "next_cochannel_load": cochannel,
                        "next_mean_prb": float(nxt.mean()),
                        "next_total_traffic_kbyte": float(traffic.iloc[window + 1].sum())})
            rows.append(row)
        print(f"NetData 5G window {window}/{len(common_times)-2}", flush=True)
    provenance["windows"] = common_times
    provenance["license_note"] = "No license was declared in the source repository; raw CSV not redistributed."
    return pd.DataFrame(rows), provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--shot-replicates", type=int, default=30)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-shots", action="store_true")
    parser.add_argument("--skip-netdata", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not args.skip_baselines:
        main_df, hard_df = practical_baselines(args.seeds)
        main_df.to_csv(RESULTS / "clearaccept_baselines.csv", index=False)
        hard_df.to_csv(RESULTS / "clearaccept_hardness.csv", index=False)
    if not args.skip_shots:
        repeated_shots(args.shot_replicates).to_csv(
            RESULTS / "clearaccept_shot_replicates.csv", index=False)
    if not args.skip_netdata:
        trace, provenance = netdata_replay()
        trace.to_csv(RESULTS / "clearaccept_netdata_5g.csv", index=False)
        (RESULTS / "clearaccept_netdata_provenance.json").write_text(
            json.dumps(provenance, indent=2))
    manifest = {
        "created_unix": time.time(), "elapsed_s": time.time() - started,
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "seeds": args.seeds,
        "shot_replicates": args.shot_replicates,
        "evidence": ["measured_local", "exact_simulated", "trace_derived"],
        "not_measured": ["live QPU output", "provider queue"],
    }
    (RESULTS / "clearaccept_manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
