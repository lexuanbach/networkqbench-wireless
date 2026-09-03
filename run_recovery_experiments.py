#!/usr/bin/env python3
"""Truth-preserving recovery experiment: transferred QAOA parameters.

Parameters are learned only from seeds 0--14 and evaluated on seeds 15--29.
The experiment tests whether removing the closed-loop tuning cost can recover
deadline feasibility, and quantifies the corresponding solution-quality loss.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from networkqbench import GENERATORS, exact_solver, operational_metrics, qaoa_probabilities


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def circular_mean(values: np.ndarray, period: float) -> float:
    z = np.exp(2j * np.pi * values / period).mean()
    return float((np.angle(z) % (2 * np.pi)) * period / (2 * np.pi))


def main() -> int:
    raw = pd.read_csv(RESULTS / "raw_results.csv")
    tuned = raw[(raw.algorithm == "qaoa") & (raw.profile == "nominal")].copy()
    train = tuned[tuned.seed < 15]
    test_seeds = range(15, 30)
    rows = []
    shot_budgets = [16, 64, 256, 1024]
    for (task, size), group in train.groupby(["task", "size_label"]):
        gamma = circular_mean(group.gamma.to_numpy(), 2 * np.pi)
        beta = circular_mean(group.beta.to_numpy(), np.pi)
        for seed in test_seeds:
            inst = GENERATORS[task](int(size), int(seed))
            exact = exact_solver(inst)
            start = time.perf_counter()
            probs = qaoa_probabilities(inst.costs, inst.n_vars, gamma, beta)
            statevector_runtime = time.perf_counter() - start
            optimum = exact["cost"]
            opt_mask = inst.feasible & (inst.costs <= optimum + 1e-9)
            success_prob = float(probs[opt_mask].sum())
            for shots in shot_budgets:
                rng = np.random.default_rng(500_000 + int(size) * 1000 + int(seed) * 10 + shots)
                sampled = rng.choice(len(probs), size=shots, p=probs)
                valid = sampled[inst.feasible[sampled]]
                idx = int(valid[np.argmin(inst.costs[valid])]) if len(valid) else int(sampled[np.argmin(inst.costs[sampled])])
                result = {
                    "solution": idx,
                    "cost": float(inst.costs[idx]),
                    "feasible": bool(inst.feasible[idx]),
                    "runtime_s": statevector_runtime,
                    "evals": 1,
                    "shots": shots,
                    "success_prob": success_prob,
                    "gamma": gamma,
                    "beta": beta,
                }
                for profile in ("optimistic", "nominal", "stressed"):
                    op = operational_metrics(inst, result, optimum, "qaoa_transfer", profile)
                    rows.append({
                        "task": task,
                        "size_label": int(size),
                        "n_vars": inst.n_vars,
                        "seed": int(seed),
                        "algorithm": "qaoa_transfer",
                        "profile": profile,
                        "shots": shots,
                        "cost": result["cost"],
                        "optimum": optimum,
                        "feasible": result["feasible"],
                        "runtime_s": statevector_runtime,
                        "evals": 1,
                        "deadline_s": inst.deadline_s,
                        "volatility": inst.volatility,
                        "success_prob": success_prob,
                        "gamma": gamma,
                        "beta": beta,
                        **op,
                    })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "recovery_parameter_transfer.csv", index=False)
    print(out.groupby(["shots", "profile"]).agg(
        feasible_rate=("feasible", "mean"),
        optimal_rate=("gap", lambda x: float((x <= 1e-12).mean())),
        deadline_miss_rate=("deadline_miss", "mean"),
        mean_utility=("utility", "mean"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
