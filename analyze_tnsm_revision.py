#!/usr/bin/env python3
"""TNSM-revision emitters: SLA weight-grid ordering stability, seed-cluster
standard error for the headline optimum rate, and Abilene block-length
sensitivity. Reads stored CSVs only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
# macros stored as .txt: the release carries no manuscript files
TABLES, MACRO_NAME = ROOT / "generated", "tnsm_numbers.txt"

MACROS = []


def macro(name, value):
    MACROS.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def block_bootstrap_difference(a, b, block=5, seed=20260903):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    difference = a - b; n = len(difference)
    starts = np.arange(max(1, n - block + 1))
    rng = np.random.default_rng(seed)
    samples = np.empty(20_000)
    for i in range(len(samples)):
        pieces = []
        while sum(map(len, pieces)) < n:
            start = int(rng.choice(starts)); pieces.append(a[start:start + block] - b[start:start + block])
        samples[i] = np.concatenate(pieces)[:n].mean()
    return float(difference.mean()), *map(float, np.percentile(samples, [2.5, 97.5]))


def weight_grid():
    """Re-scalarize stored per-instance components over a deadline x
    infeasibility weight grid and test ordering stability of the
    per-(task, profile) algorithm ranking by mean utility. Classical
    solvers contribute their measured-local rows to every profile cell,
    matching the paper's operational comparison."""
    raw = pd.read_csv(RESULTS / "raw_results.csv")
    infeasible = 1.0 - raw.feasible.astype(float)
    raw = raw.assign(infeasible=infeasible)
    classical = raw[raw.algorithm.isin(
        ["local_search", "annealing", "mean_field"])
        & (raw.profile == "local")]
    base_rank = {}
    changed = []
    grid = [(2.0, 5.0)] + [(wd, wf) for wd in (1.0, 2.0, 5.0)
            for wf in (2.0, 5.0, 10.0) if (wd, wf) != (2.0, 5.0)]
    for wd, wf in grid:
        for profile in ("optimistic", "nominal", "stressed"):
            q = raw[(raw.algorithm == "qaoa") & (raw.profile == profile)]
            cell_rows = pd.concat([q, classical], ignore_index=True)
            u = -(cell_rows.gap + wf * cell_rows.infeasible
                  + cell_rows.staleness_loss + wd * cell_rows.deadline_miss)
            means = cell_rows.assign(u=u).groupby(
                ["task", "algorithm"]).u.mean()
            for task, sub in means.groupby(level=0):
                order = tuple(sub.sort_values(ascending=False)
                              .index.get_level_values(1))
                key = (task, profile)
                if (wd, wf) == (2.0, 5.0):
                    base_rank[key] = order
                elif order != base_rank[key]:
                    changed.append((wd, wf, task, profile))
    cells = len(base_rank) * (len(grid) - 1)
    macro("WeightGridCells", cells)
    macro("WeightGridChanged", len(changed))
    if changed:
        pd.DataFrame(changed, columns=["w_deadline", "w_infeasible",
                                       "task", "profile"]).to_csv(
            RESULTS / "tnsm_weight_grid_changes.csv", index=False)


def seed_cluster_se():
    """Seed-cluster standard error of the depth-one QAOA optimum-return
    rate on the 360 primary instances (local rows)."""
    raw = pd.read_csv(RESULTS / "raw_results.csv")
    q = raw[(raw.algorithm == "qaoa") & (raw.profile == "nominal")].copy()
    q["optimal"] = (q.feasible.astype(bool)) & (q.gap.abs() <= 1e-12)
    per_seed = q.groupby("seed").optimal.mean()
    rate = float(q.optimal.mean())
    se = float(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
    macro("QaoaRatePct", f"{100*rate:.1f}\\%")
    macro("QaoaRateSeedSEPct", f"{100*se:.1f}")
    macro("QaoaRateSeeds", len(per_seed))


def abilene_blocks():
    abilene = pd.read_csv(RESULTS / "revision_abilene_trace.csv")
    abilene = abilene[(abilene.algorithm.isin(["local_search", "qaoa"]))
                      & ((abilene.algorithm != "qaoa")
                         | (abilene.profile == "nominal"))]
    apiv = abilene.pivot_table(index="seed", columns="algorithm",
                               values="next_max_utilization").sort_index()
    for block, nm in ((3, "Three"), (5, "Five"), (7, "Seven")):
        m, lo, hi = block_bootstrap_difference(apiv.qaoa, apiv.local_search,
                                               block=block)
        macro(f"AbileneBlock{nm}Lo", f"{lo:+.4f}")
        macro(f"AbileneBlock{nm}Hi", f"{hi:+.4f}")


def main() -> int:
    weight_grid()
    seed_cluster_se()
    abilene_blocks()
    (TABLES / MACRO_NAME).write_text("\n".join(MACROS) + "\n")
    print(f"emitted {len(MACROS)} macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
