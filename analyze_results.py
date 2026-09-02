#!/usr/bin/env python3
"""Analyze raw benchmark results and generate paper tables/figures."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "submission" / "03_experiments" / "results"
FIGURES = ROOT / "submission" / "04_draft" / "figures"
TABLES = ROOT / "submission" / "04_draft" / "tables"


def ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return float(values[0]), float(values[0])
    res = bootstrap((values,), np.mean, n_resamples=2000, method="percentile", random_state=123)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z / denominator * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
    return float(center - radius), float(center + radius)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "raw_results.csv")
    df["optimal"] = df.feasible & (np.abs(df.cost - df.optimum) <= 1e-9)
    df["quality_loss"] = df.gap + (~df.feasible).astype(float) * 5.0

    # Static solver quality uses a single row per algorithm/instance; nominal QAOA is representative.
    static = df[(df.algorithm != "qaoa") | (df.profile == "nominal")].copy()
    summary_rows = []
    for (task, alg), g in static.groupby(["task", "algorithm"]):
        # The reported mean includes the same infeasibility penalty used by the
        # benchmark utility, so its uncertainty interval must use that identical
        # quantity rather than the unpenalized objective gap.
        lo, hi = ci(g.quality_loss.to_numpy())
        summary_rows.append({
            "task": task,
            "algorithm": alg,
            "instances": len(g),
            "feasible_rate": g.feasible.mean(),
            "optimal_rate": g.optimal.mean(),
            "mean_gap": g.quality_loss.mean(),
            "gap_ci_low": lo,
            "gap_ci_high": hi,
            "median_runtime_s": g.runtime_s.median(),
            "deadline_miss_rate": g.deadline_miss.mean(),
            "mean_utility": g.utility.mean(),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "summary_by_task_algorithm.csv", index=False)

    profile = df.groupby(["task", "algorithm", "profile"], as_index=False).agg(
        mean_utility=("utility", "mean"),
        deadline_miss_rate=("deadline_miss", "mean"),
        mean_delay_s=("delay_s", "mean"),
        optimal_rate=("optimal", "mean"),
        feasible_rate=("feasible", "mean"),
    )
    profile.to_csv(RESULTS / "operational_profiles.csv", index=False)

    # Paired QAOA vs strongest practical baseline on static penalized loss.
    paired = static[static.algorithm.isin(["qaoa", "local_search"])].pivot_table(
        index=["task", "size_label", "seed"], columns="algorithm", values="quality_loss"
    ).dropna()
    tests = []
    for task, g in paired.groupby(level=0):
        q = g.qaoa.to_numpy()
        a = g.local_search.to_numpy()
        try:
            stat, p = wilcoxon(q, a, zero_method="zsplit", alternative="greater")
        except ValueError:
            stat, p = 0.0, 1.0
        tests.append({
            "task": task,
            "comparison": "qaoa_gap_greater_than_local_search",
            "n": len(g),
            "median_qaoa_gap": float(np.median(q)),
            "median_local_search_gap": float(np.median(a)),
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
        })
    pd.DataFrame(tests).to_csv(RESULTS / "paired_tests.csv", index=False)

    # Figure 1: optimal-solution rate by task.
    alg_order = ["exact", "local_search", "mean_field", "annealing", "random", "qaoa"]
    tasks = ["channel", "placement", "routing"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, task in zip(axes, tasks):
        g = summary[summary.task == task].set_index("algorithm").reindex(alg_order)
        rates, lower, upper = [], [], []
        for algorithm in alg_order:
            observations = static[(static.task == task) & (static.algorithm == algorithm)].optimal
            rate = float(observations.mean())
            lo, hi = wilson_interval(int(observations.sum()), len(observations))
            rates.append(rate); lower.append(rate - lo); upper.append(hi - rate)
        ax.bar(range(len(g)), rates, yerr=np.array([lower, upper]), capsize=2,
               color=["#333333", "#4C78A8", "#72B7B2", "#59A14F", "#F28E2B", "#B279A2"],
               error_kw={"elinewidth": 0.8})
        ax.set_title(task.capitalize())
        ax.set_xticks(range(len(g)), ["Exact", "Local", "Mean-field", "SA", "Random", "QAOA"], rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Optimal-solution rate")
    fig.tight_layout()
    fig.savefig(FIGURES / "optimal_rate.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "optimal_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: modeled operational utility by QPU deployment profile.
    q = profile[profile.algorithm == "qaoa"].copy()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(tasks))
    width = 0.24
    for j, prof in enumerate(["optimistic", "nominal", "stressed"]):
        vals = q[q.profile == prof].set_index("task").reindex(tasks).mean_utility
        ax.bar(x + (j - 1) * width, vals, width, label=prof.capitalize())
    # Add measured-local annealing utility as reference.
    ref = profile[(profile.algorithm == "annealing")].set_index("task").reindex(tasks).mean_utility
    ax.plot(x, ref, "ko--", label="Annealing (measured local)")
    ax.set_xticks(x, [t.capitalize() for t in tasks])
    ax.set_ylabel("Mean operational utility (higher is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "operational_utility.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "operational_utility.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: QAOA success probability scaling.
    qstatic = static[static.algorithm == "qaoa"]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for task, g in qstatic.groupby("task"):
        agg = g.groupby("n_vars").success_prob.agg(["mean", "std", "count"]).reset_index()
        err = 1.96 * agg["std"].fillna(0) / np.sqrt(agg["count"])
        ax.errorbar(agg.n_vars, agg["mean"], yerr=err, marker="o", capsize=3, label=task.capitalize())
    ax.set_xlabel("Binary decision variables")
    ax.set_ylabel("QAOA probability of an optimum")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "qaoa_success_scaling.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "qaoa_success_scaling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Compact LaTeX table generated from real outputs.
    compact = summary[summary.algorithm.isin(alg_order)].copy()
    compact["Opt. rate"] = compact.optimal_rate.map(lambda x: f"{100*x:.1f}\\%")
    compact["Feas. rate"] = compact.feasible_rate.map(lambda x: f"{100*x:.1f}\\%")
    compact["Mean gap"] = compact.mean_gap.map(lambda x: f"{x:.3f}")
    compact["Median time (s)"] = compact.median_runtime_s.map(lambda x: f"{x:.4f}")
    table = compact[["task", "algorithm", "Opt. rate", "Feas. rate", "Mean gap", "Median time (s)"]].copy()
    table["task"] = table["task"].str.capitalize()
    table["algorithm"] = table["algorithm"].replace({
        "annealing": "Annealing",
        "exact": "Exact",
        "local_search": "Local search",
        "mean_field": "Mean-field",
        "qaoa": "QAOA",
        "random": "Random",
    })
    (TABLES / "solver_summary.tex").write_text(table.to_latex(index=False, escape=False))

    key = {
        "instances": int(static[["task", "size_label", "seed"]].drop_duplicates().shape[0]),
        "raw_rows": int(len(df)),
        "qaoa_mean_success_probability": float(qstatic.success_prob.mean()),
        "qaoa_median_success_probability": float(qstatic.success_prob.median()),
        "qaoa_optimal_rate": float((qstatic.feasible & (np.abs(qstatic.cost-qstatic.optimum) <= 1e-9)).mean()),
        "annealing_optimal_rate": float(((static[static.algorithm == "annealing"].feasible) & (np.abs(static[static.algorithm == "annealing"].cost-static[static.algorithm == "annealing"].optimum) <= 1e-9)).mean()),
        "local_search_optimal_rate": float(((static[static.algorithm == "local_search"].feasible) & (np.abs(static[static.algorithm == "local_search"].cost-static[static.algorithm == "local_search"].optimum) <= 1e-9)).mean()),
    }

    recovery_path = RESULTS / "recovery_parameter_transfer.csv"
    if recovery_path.exists():
        recovery = pd.read_csv(recovery_path)
        rec_summary = recovery.groupby(["shots", "profile"], as_index=False).agg(
            feasible_rate=("feasible", "mean"),
            optimal_rate=("gap", lambda x: float((x <= 1e-12).mean())),
            deadline_miss_rate=("deadline_miss", "mean"),
            mean_utility=("utility", "mean"),
            mean_delay_s=("delay_s", "mean"),
        )
        rec_summary.to_csv(RESULTS / "recovery_summary.csv", index=False)
        nominal = rec_summary[rec_summary.profile == "nominal"]
        fig, ax1 = plt.subplots(figsize=(6.5, 3.5))
        ax1.plot(nominal.shots, nominal.optimal_rate, "o-", label="Optimal rate", color="#4C78A8")
        ax1.plot(nominal.shots, 1-nominal.deadline_miss_rate, "s-", label="On-time rate", color="#59A14F")
        ax1.set_xscale("log", base=2)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlabel("Shots with transferred parameters")
        ax1.set_ylabel("Rate")
        ax1.grid(alpha=0.25)
        ax1.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES / "transfer_tradeoff.pdf", bbox_inches="tight")
        fig.savefig(FIGURES / "transfer_tradeoff.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        best = nominal.sort_values("mean_utility", ascending=False).iloc[0]
        key.update({
            "transfer_best_shots_nominal": int(best.shots),
            "transfer_best_mean_utility_nominal": float(best.mean_utility),
            "transfer_best_optimal_rate_nominal": float(best.optimal_rate),
            "transfer_best_on_time_rate_nominal": float(1-best.deadline_miss_rate),
        })
    (RESULTS / "key_results.json").write_text(json.dumps(key, indent=2))
    print(json.dumps(key, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
