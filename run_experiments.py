#!/usr/bin/env python3
"""Run the complete simulator-first NetworkQBench-Wireless experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from networkqbench import (
    GENERATORS,
    annealing_solver,
    exact_solver,
    local_search_solver,
    mean_field_solver,
    operational_metrics,
    qaoa_solver,
    random_solver,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "03_experiments" / "results"


def run(seed_count: int, sizes: list[int], tasks: list[str], shots: int) -> pd.DataFrame:
    rows = []
    total = len(tasks) * len(sizes) * seed_count
    done = 0
    wall_start = time.perf_counter()
    for task in tasks:
        for size in sizes:
            for seed in range(seed_count):
                inst = GENERATORS[task](size, seed)
                optimum_result = exact_solver(inst)
                optimum = optimum_result["cost"]
                rng_base = 100_000 * (1 + list(GENERATORS).index(task)) + 1_000 * size + seed
                results = {
                    "exact": optimum_result,
                    "local_search": local_search_solver(inst, np.random.default_rng(rng_base + 1)),
                    "annealing": annealing_solver(inst, np.random.default_rng(rng_base + 2)),
                    "random": random_solver(inst, np.random.default_rng(rng_base + 3)),
                    "qaoa": qaoa_solver(inst, np.random.default_rng(rng_base + 4), shots=shots),
                    "mean_field": mean_field_solver(inst, np.random.default_rng(rng_base + 5)),
                }
                for algorithm, result in results.items():
                    profiles = ["local"] if algorithm != "qaoa" else ["optimistic", "nominal", "stressed"]
                    for profile in profiles:
                        op = operational_metrics(inst, result, optimum, algorithm, profile)
                        row = {
                            "task": task,
                            "size_label": size,
                            "n_vars": inst.n_vars,
                            "seed": seed,
                            "algorithm": algorithm,
                            "profile": profile,
                            "cost": result["cost"],
                            "optimum": optimum,
                            "feasible": result["feasible"],
                            "runtime_s": result["runtime_s"],
                            "evals": result["evals"],
                            "deadline_s": inst.deadline_s,
                            "volatility": inst.volatility,
                            **op,
                        }
                        for key in ("shots", "success_prob", "gamma", "beta", "expected_norm_cost"):
                            row[key] = result.get(key, np.nan)
                        rows.append(row)
                done += 1
                if done % 10 == 0 or done == total:
                    elapsed = time.perf_counter() - wall_start
                    print(f"[{done:>4}/{total}] {task} n={size} seed={seed}; elapsed={elapsed:.1f}s", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--sizes", default="6,8,9,12")
    parser.add_argument("--tasks", default="channel,placement,routing")
    parser.add_argument("--shots", type=int, default=1024)
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    tasks = args.tasks.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    df = run(args.seeds, sizes, tasks, args.shots)
    df.to_csv(OUT / "raw_results.csv", index=False)
    manifest = {
        "created_unix": time.time(),
        "elapsed_s": time.time() - start,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seeds": args.seeds,
        "sizes": sizes,
        "tasks": tasks,
        "shots": args.shots,
        "qaoa_execution": "local exact statevector simulation, p=1",
        "qpu_profiles": "modeled, not hardware measurements",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(df)} rows to {OUT / 'raw_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
