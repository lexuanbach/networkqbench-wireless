#!/usr/bin/env python3
"""Generate revision statistics, figures, and LaTeX assets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times",
                                    "Nimbus Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix"})
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"


def bootstrap_ci(values, seed=19):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(20_000, len(values)))].mean(1)
    return float(values.mean()), *map(float, np.percentile(means, [2.5, 97.5]))


def wilson(successes: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RESULTS / "raw_results.csv")
    static = raw[(raw.algorithm != "qaoa") | (raw.profile == "nominal")].copy()
    static["quality_loss"] = static.gap + (~static.feasible).astype(float) * 5.0
    static["optimal"] = static.feasible & (np.abs(static.cost - static.optimum) <= 1e-9)

    piv = static.pivot_table(index=["task", "size_label", "seed"],
                             columns="algorithm", values="quality_loss")
    diff = (piv.qaoa - piv.local_search).dropna().to_numpy()
    eq_mean, eq_lo, eq_hi = bootstrap_ci(diff)
    equivalence_margin = 0.05
    equivalent = bool(eq_lo > -equivalence_margin and eq_hi < equivalence_margin)

    rates = []
    for algorithm, group in static.groupby("algorithm"):
        successes = int(group.optimal.sum())
        lo, hi = wilson(successes, len(group))
        rates.append({"algorithm": algorithm, "n": len(group),
                      "rate": successes / len(group), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(rates).to_csv(RESULTS / "revision_rate_intervals.csv", index=False)

    q = static[static.algorithm == "qaoa"].copy()
    pstar = q.success_prob.clip(lower=1e-15, upper=1 - 1e-15)
    shots95 = np.ceil(np.log(0.05) / np.log1p(-pstar)).astype(int)
    shots99 = np.ceil(np.log(0.01) / np.log1p(-pstar)).astype(int)

    hard = pd.read_csv(RESULTS / "revision_hardness.csv")
    hard = hard[(hard.algorithm != "qaoa") | (hard.profile == "nominal")].copy()
    hard["optimal"] = hard.feasible & (np.abs(hard.cost - hard.optimum) <= 1e-9)
    hard_summary = hard.groupby(["n_vars", "algorithm"], as_index=False).agg(
        instances=("seed", "size"), optimal_rate=("optimal", "mean"),
        mean_gap=("gap", "mean"), mean_utility=("utility", "mean"))
    hard_summary.to_csv(RESULTS / "revision_hardness_summary.csv", index=False)

    trace = pd.read_csv(RESULTS / "revision_abilene_trace.csv")
    trace = trace[(trace.algorithm != "qaoa") | (trace.profile == "nominal")].copy()
    trace["optimal"] = trace.feasible & (np.abs(trace.cost - trace.optimum) <= 1e-9)
    trace_summary = trace.groupby("algorithm", as_index=False).agg(
        windows=("seed", "size"), optimal_rate=("optimal", "mean"),
        on_time_rate=("deadline_miss", lambda x: 1 - x.mean()),
        mean_utility=("utility", "mean"),
        next_max_utilization=("next_max_utilization", "mean"),
        next_overflow=("next_overflow", "mean"))
    trace_summary.to_csv(RESULTS / "revision_abilene_summary.csv", index=False)

    # Deadline/queue boundary: on-time fraction over the observed deadlines.
    queues = np.geomspace(0.005, 5.0, 45)
    eval_counts = np.unique(np.round(np.geomspace(1, 128, 45)).astype(int))
    observed_deadlines = q.deadline_s.to_numpy()
    heat = np.empty((len(eval_counts), len(queues)))
    for i, evals in enumerate(eval_counts):
        for j, queue in enumerate(queues):
            delay = queue + evals * (0.005 + 0.020 + 1024 * 0.00005)
            heat[i, j] = np.mean(delay <= observed_deadlines)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(3.5, 1.8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    im = axa.pcolormesh(queues, eval_counts, heat, shading="auto", vmin=0,
                        vmax=1, cmap="viridis")
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.tick_params(labelsize=6)
    axa.set_xlabel("queue delay (s)", fontsize=7)
    axa.set_ylabel("online evaluations", fontsize=7)
    axa.set_title("(a) service boundary", fontsize=7)
    cb = fig.colorbar(im, ax=axa, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("on-time fraction", fontsize=7)
    rec = pd.read_csv(RESULTS / "recovery_summary.csv")
    nominal = rec[rec.profile == "nominal"]
    axb.plot(nominal.shots, nominal.optimal_rate, "o-", ms=2.5, lw=1,
             label="optimal", color="#4C78A8")
    axb.plot(nominal.shots, 1 - nominal.deadline_miss_rate, "s-", ms=2.5,
             lw=1, label="on-time", color="#59A14F")
    axb.set_xscale("log", base=2)
    axb.set_ylim(0, 1.05)
    axb.tick_params(labelsize=6)
    axb.set_xlabel("transferred shots", fontsize=7)
    axb.set_ylabel("rate", fontsize=7)
    axb.set_title("(b) transfer frontier", fontsize=7)
    axb.grid(alpha=0.25, lw=0.3)
    axb.legend(frameon=False, fontsize=5.5, handlelength=1.4)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIGURES / "revision_boundary_transfer.pdf", bbox_inches="tight")
    plt.close(fig)

    order = ["local_search", "mean_field", "qaoa", "annealing"]
    ts = trace_summary.set_index("algorithm").reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.25))
    labels = ["Local", "Mean-field", "QAOA", "SA"]
    axes[0].bar(labels, ts.optimal_rate, color=["#4c78a8", "#72b7b2", "#b279a2", "#59a14f"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("optimal-decision rate")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, ts.next_max_utilization, color=["#4c78a8", "#72b7b2", "#b279a2", "#59a14f"])
    axes[1].set_ylabel("next-window max utilization")
    axes[1].tick_params(axis="x", rotation=25)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "revision_abilene_replay.pdf", bbox_inches="tight")
    plt.close(fig)

    # Logical resource lower bounds for one p=1 cost layer at n=12.
    resources = [
        ("Channel assignment", 12, 0, 66, 132, "exact QUBO"),
        ("Service placement", 12, 0, 42, 84, "lower bound, overflow omitted"),
        ("Path selection", 12, 0, 12, 24, "lower bound, overflow omitted"),
    ]
    lines = [r"\begin{tabular}{@{}lrrrrl@{}}", r"\toprule",
             r"Task & Logical & Anc. & ZZ & CX & Encoding status \\", r"\midrule"]
    for name, logical, anc, zz, cx, status in resources:
        lines.append(f"{name} & {logical} & {anc} & {zz} & {cx} & {status} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "revision_resource_ledger.tex").write_text("\n".join(lines) + "\n")

    def alg_label(name):
        return {"local_search": "Local", "mean_field": "Mean-field",
                "qaoa": "QAOA", "annealing": "SA"}.get(name, name)
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Method & Opt. rate & On time & Next max util. & Overflow \\", r"\midrule"]
    for row in ts.reset_index().itertuples():
        lines.append(f"{alg_label(row.algorithm)} & {100*row.optimal_rate:.1f}\\% & "
                     f"{100*row.on_time_rate:.1f}\\% & {row.next_max_utilization:.3f} & "
                     f"{row.next_overflow:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "revision_trace_summary.tex").write_text("\n".join(lines) + "\n")

    report = {
        "equivalence_margin": equivalence_margin,
        "qaoa_minus_local_mean_quality_loss": eq_mean,
        "qaoa_minus_local_ci": [eq_lo, eq_hi],
        "equivalent_within_margin": equivalent,
        "median_shots_95pct": int(np.median(shots95)),
        "p90_shots_95pct": int(np.percentile(shots95, 90)),
        "median_shots_99pct": int(np.median(shots99)),
        "hard_n16": hard_summary[hard_summary.n_vars == 16].to_dict("records"),
        "abilene": trace_summary.to_dict("records"),
    }
    (RESULTS / "revision_key_results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
