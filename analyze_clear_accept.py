#!/usr/bin/env python3
"""Analyze clear-accept experiments and emit manuscript-ready assets."""

from __future__ import annotations

import json
from pathlib import Path

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
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
ROW_END = r"\\"


def boot_ci(values, seed=20260901, draws=20_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return float(values.mean()), *map(float, np.percentile(sampled, [2.5, 97.5]))


def stratified_bootstrap(frame, value, strata=("task", "size_label"), seed=20260902):
    groups = [g[value].to_numpy(dtype=float) for _, g in frame.groupby(list(strata))]
    rng = np.random.default_rng(seed)
    samples = np.empty(20_000)
    for i in range(len(samples)):
        samples[i] = np.mean([g[rng.integers(0, len(g), len(g))].mean() for g in groups])
    estimate = float(np.mean([g.mean() for g in groups]))
    return estimate, *map(float, np.percentile(samples, [2.5, 97.5]))


def block_bootstrap_difference(a, b, block=5, seed=20260903):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    difference = a - b; n = len(difference)
    starts = np.arange(max(1, n - block + 1))
    rng = np.random.default_rng(seed)
    samples = np.empty(20_000)
    for i in range(len(samples)):
        pieces = []
        while sum(map(len, pieces)) < n:
            start = int(rng.choice(starts)); pieces.append(difference[start:start + block])
        samples[i] = np.concatenate(pieces)[:n].mean()
    return float(difference.mean()), *map(float, np.percentile(samples, [2.5, 97.5]))


def label(name):
    return {
        "local_search": "Local", "qaoa": "QAOA", "qaoa_warm": "Warm QAOA",
        "task_specific": "Task-specific", "milp": "HiGHS MILP",
        "mean_field": "Mean-field", "annealing": "SA",
    }.get(name, name)


def write_table(path: Path, header: str, rows: list[str], spec: str):
    path.write_text("\n".join([f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule",
                                header, r"\midrule", *rows,
                                r"\bottomrule", r"\end{tabular}"]) + "\n")


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(RESULTS / "raw_results.csv")
    old = old[(old.algorithm != "qaoa") | (old.profile == "nominal")].copy()
    new = pd.read_csv(RESULTS / "clearaccept_baselines.csv")
    common = ["task", "size_label", "n_vars", "seed", "algorithm", "cost",
              "optimum", "feasible", "runtime_s", "evals", "success_prob"]
    new_for_summary = new.rename(columns={"solver_s": "runtime_s"})
    main = pd.concat([old[common], new_for_summary[common]], ignore_index=True)
    main["optimal"] = main.feasible & (np.abs(main.cost - main.optimum) <= 1e-9)
    main["gap"] = np.maximum(0.0, (main.cost - main.optimum) / (np.abs(main.optimum) + 1.0))
    algorithms = ["local_search", "milp", "task_specific", "qaoa", "qaoa_warm", "mean_field"]
    summary = main[main.algorithm.isin(algorithms)].groupby("algorithm", as_index=False).agg(
        instances=("seed", "size"), optimal_rate=("optimal", "mean"),
        feasible_rate=("feasible", "mean"), mean_gap=("gap", "mean"),
        median_solver_s=("runtime_s", "median"))
    summary.to_csv(RESULTS / "clearaccept_baseline_summary.csv", index=False)

    rows = []
    for algorithm in algorithms:
        row = summary.set_index("algorithm").loc[algorithm]
        rows.append(f"{label(algorithm)} & {100*row.optimal_rate:.1f}\\% & "
                    f"{100*row.feasible_rate:.1f}\\% & {row.mean_gap:.3f} & "
                    f"{1e3*row.median_solver_s:.2f} " + ROW_END)
    write_table(TABLES / "clearaccept_baselines.tex",
                "Method & Opt. & Feas. & Mean gap & Solver ms " + ROW_END, rows, "lrrrr")

    hard_old = pd.read_csv(RESULTS / "revision_hardness.csv")
    hard_old = hard_old[(hard_old.algorithm != "qaoa") | (hard_old.profile == "nominal")]
    hard_new = pd.read_csv(RESULTS / "clearaccept_hardness.csv").rename(
        columns={"solver_s": "runtime_s"})
    hard = pd.concat([hard_old, hard_new], ignore_index=True)
    hard["optimal"] = hard.feasible & (np.abs(hard.cost - hard.optimum) <= 1e-9)
    hard["gap2"] = np.maximum(0.0, (hard.cost - hard.optimum) / (np.abs(hard.optimum) + 1.0))
    h16 = hard[hard.n_vars == 16].groupby("algorithm", as_index=False).agg(
        instances=("seed", "size"), optimal_rate=("optimal", "mean"),
        mean_gap=("gap2", "mean"), median_solver_s=("runtime_s", "median"))
    h16.to_csv(RESULTS / "clearaccept_hardness_n16_summary.csv", index=False)
    rows = []
    hard_order = ["milp", "task_specific", "local_search", "qaoa_warm", "qaoa", "mean_field", "annealing"]
    for algorithm in hard_order:
        if algorithm not in set(h16.algorithm):
            continue
        row = h16.set_index("algorithm").loc[algorithm]
        rows.append(f"{label(algorithm)} & {100*row.optimal_rate:.1f}\\% & "
                    f"{row.mean_gap:.3f} & {1e3*row.median_solver_s:.2f} " + ROW_END)
    write_table(TABLES / "clearaccept_hardness.tex",
                "Method & Opt. at $n{=}16$ & Mean gap & Solver ms " + ROW_END, rows, "lrrr")

    # Paired, stratum-balanced equivalence and margin curve.
    static = old.copy()
    static["loss"] = np.maximum(0.0, (static.cost - static.optimum)
                                 / (np.abs(static.optimum) + 1.0)) \
        + 5.0 * (~static.feasible).astype(float)
    pivot = static.pivot_table(index=["task", "size_label", "seed"],
                               columns="algorithm", values="loss").reset_index()
    pivot["difference"] = pivot.qaoa - pivot.local_search
    eq = stratified_bootstrap(pivot, "difference")
    margin_curve = [{"margin": margin,
                     "equivalent": bool(eq[1] > -margin and eq[2] < margin)}
                    for margin in np.arange(0.01, 0.101, 0.01)]

    shots = pd.read_csv(RESULTS / "clearaccept_shot_replicates.csv")
    per_instance = shots.groupby(["task", "size_label", "seed", "shots"], as_index=False).agg(
        optimal_rate=("optimal", "mean"), feasible_rate=("feasible", "mean"),
        mean_gap=("gap", "mean"))
    shot_summary = per_instance.groupby("shots", as_index=False).agg(
        mean_optimal_rate=("optimal_rate", "mean"),
        between_instance_sd=("optimal_rate", "std"),
        mean_feasible_rate=("feasible_rate", "mean"), mean_gap=("mean_gap", "mean"))
    shot_summary.to_csv(RESULTS / "clearaccept_shot_summary.csv", index=False)
    rows = [f"{int(r.shots)} & {100*r.mean_optimal_rate:.1f}\\% & "
            f"{100*r.between_instance_sd:.1f} pp & {100*r.mean_feasible_rate:.1f}\\% & {r.mean_gap:.3f} " + ROW_END
            for r in shot_summary.itertuples()]
    write_table(TABLES / "clearaccept_shot_replicates.tex",
                "Shots & Opt. mean & Across-inst. SD & Feas. & Gap " + ROW_END, rows, "rrrrr")

    trace = pd.read_csv(RESULTS / "clearaccept_netdata_5g.csv")
    trace["optimal"] = trace.feasible & (np.abs(trace.cost - trace.optimum) <= 1e-9)
    trace_summary = trace.groupby("algorithm", as_index=False).agg(
        windows=("seed", "size"), optimal_rate=("optimal", "mean"),
        next_cochannel_load=("next_cochannel_load", "mean"),
        deadline_miss=("deadline_miss", "mean"))
    trace_summary.to_csv(RESULTS / "clearaccept_netdata_5g_summary.csv", index=False)
    rows = [f"{label(r.algorithm)} & {100*r.optimal_rate:.1f}\\% & "
            f"{r.next_cochannel_load:.3f} & {100*r.deadline_miss:.1f}\\% " + ROW_END
            for r in trace_summary.sort_values("next_cochannel_load").itertuples()]
    write_table(TABLES / "clearaccept_netdata_5g.tex",
                "Method & Opt. & Next co-channel load & Miss " + ROW_END, rows, "lrrr")

    compiled = pd.read_csv(RESULTS / "clearaccept_compiled_resources.csv")
    c12 = compiled[compiled.n_vars == 12]
    rows = [f"{r.task.title()} & {r.topology.replace('line16','Line').replace('grid4x4','Grid')} & "
            f"{int(r.logical_interactions)} & {int(r.compiled_cx)} & "
            f"{int(r.routing_cx_overhead)} & {int(r.compiled_depth)} " + ROW_END
            for r in c12.itertuples()]
    write_table(TABLES / "clearaccept_compiled_resources.tex",
                "Task & Topology & Interactions & CX & Added CX & Depth " + ROW_END, rows, "llrrrr")

    abilene = pd.read_csv(RESULTS / "revision_abilene_trace.csv")
    abilene = abilene[(abilene.algorithm.isin(["local_search", "qaoa"]))
                      & ((abilene.algorithm != "qaoa") | (abilene.profile == "nominal"))]
    apiv = abilene.pivot_table(index="seed", columns="algorithm",
                               values="next_max_utilization").sort_index()
    abilene_block = block_bootstrap_difference(apiv.qaoa, apiv.local_search)

    # Fable-requested depth, transfer, penalty, staleness, trace, and timing audits.
    depth = pd.read_csv(RESULTS / "fable_depth_sensitivity.csv")
    depth_summary = depth.groupby("depth_p", as_index=False).agg(
        instances=("seed", "size"), optimum_probability=("optimum_probability", "mean"),
        hit_probability_1024=("hit_probability_1024", "mean"),
        expected_normalized_cost=("expected_normalized_cost", "mean"),
        median_optimizer_s=("optimizer_s", "median"))
    rows = [f"{int(r.depth_p)} & {int(r.instances)} & {r.optimum_probability:.4f} & "
            f"{r.hit_probability_1024:.3f} & {r.expected_normalized_cost:.4f} & "
            f"{r.median_optimizer_s:.3f} " + ROW_END for r in depth_summary.itertuples()]
    write_table(TABLES / "fable_depth_sensitivity.tex",
                "$p$ & Inst. & $P(x^*)$ & $P_{1024}(x^*)$ & $E[\\tilde C]$ & Fit s " + ROW_END,
                rows, "rrrrrr")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    for task, group in depth.groupby(["task", "depth_p"]).agg(
            optimum_probability=("optimum_probability", "mean"),
            expected_normalized_cost=("expected_normalized_cost", "mean")).reset_index().groupby("task"):
        axes[0].plot(group.depth_p, group.optimum_probability, marker="o", label=task.title())
        axes[1].plot(group.depth_p, group.expected_normalized_cost, marker="o", label=task.title())
    axes[0].set_ylabel("Exact optimum probability"); axes[1].set_ylabel("Expected normalized cost")
    for axis in axes:
        axis.set_xlabel("QAOA depth $p$"); axis.set_xticks([1, 2, 3]); axis.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "fable_depth_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    transfer = pd.read_csv(RESULTS / "fable_transfer_aggregation.csv")
    transfer_summary = transfer.groupby("aggregation", as_index=False).agg(
        optimum_probability=("optimum_probability", "mean"),
        hit_probability_1024=("hit_probability_1024", "mean"),
        expected_normalized_cost=("expected_normalized_cost", "mean"))
    display = {"circular_mean": "Circular mean", "component_circular_median": "Component median",
               "observed_circular_medoid": "Observed medoid"}
    rows = [f"{display[r.aggregation]} & {r.optimum_probability:.4f} & "
            f"{r.hit_probability_1024:.4f} & {r.expected_normalized_cost:.4f} " + ROW_END
            for r in transfer_summary.itertuples()]
    write_table(TABLES / "fable_transfer_aggregation.tex",
                "Aggregator & $P(x^*)$ & $P_{1024}(x^*)$ & $E[\\tilde C]$ " + ROW_END,
                rows, "lrrr")

    penalty = pd.read_csv(RESULTS / "fable_penalty_sensitivity.csv")
    penalty_summary = penalty.groupby("penalty_scale", as_index=False).agg(
        instances=("seed", "size"), any_feasible_ground_state=("any_feasible_ground_state", "mean"),
        all_ground_states_feasible=("all_ground_states_feasible", "mean"))
    # The two estimands coincide on every instance at these sizes (verified:
    # 0/600 rows differ), so one column is emitted; the text explains why.
    rows = [f"{r.penalty_scale:.2f} & {int(r.instances)} & "
            f"{100*r.any_feasible_ground_state:.1f}\\% " + ROW_END
            for r in penalty_summary.itertuples()]
    write_table(TABLES / "fable_penalty_sensitivity.tex",
                "Scale & Inst. & Feasible penalized ground state " + ROW_END, rows, "rrr")

    stale = pd.read_csv(RESULTS / "fable_staleness_sensitivity.csv")
    stale_summary = stale.groupby("staleness_multiplier", as_index=False).agg(
        utility_difference=("mean_qaoa_minus_local_utility", "mean"),
        qaoa_win_rate=("qaoa_win_rate", "mean"))
    rows = [f"{r.staleness_multiplier:.1f} & {r.utility_difference:+.3f} & "
            f"{100*r.qaoa_win_rate:.1f}\\% " + ROW_END for r in stale_summary.itertuples()]
    write_table(TABLES / "fable_staleness_sensitivity.tex",
                "Staleness multiplier & QAOA minus local utility & QAOA wins " + ROW_END,
                rows, "rrr")

    mip = pd.read_csv(RESULTS / "fable_mip_deadline_coverage.csv")
    rows = [f"{r.task.title()} & {int(r.n_vars)} & {100*r.optimal_rate:.0f}\\% & "
            f"{1e3*r.median_complete_s:.2f} & {1e3*r.p95_complete_s:.2f} & "
            f"{100*r.deadline_coverage:.0f}\\% " + ROW_END for r in mip.itertuples()]
    write_table(TABLES / "fable_mip_deadline.tex",
                "Task & $n$ & Opt. & Median ms & P95 ms & Deadline cov. " + ROW_END,
                rows, "lrrrrr")

    frontier = pd.read_csv(RESULTS / "fable_timing_frontier.csv")
    frontier_use = frontier[(frontier.profile == "nominal")
                            & frontier.queue_s.isin([0.0, 0.05, 0.10, 0.50])]
    rows = [f"{int(r.shots)} & {r.queue_s:.2f} & {int(r.max_rounds_for_90pct_ontime)} " + ROW_END
            for r in frontier_use.itertuples()]
    write_table(TABLES / "fable_timing_frontier.tex",
                "Shots & Queue s & Max rounds at $\\geq90\\%$ on time " + ROW_END,
                rows, "rrr")

    queue_replay = pd.read_csv(RESULTS / "fable_quantumqueue_replay.csv")
    queue_summary = queue_replay.groupby("algorithm", as_index=False).agg(
        deadline_miss=("deadline_miss", "mean"), mean_utility=("utility", "mean"),
        mean_delay_s=("delay_s", "mean"))
    queue_meta = json.loads((RESULTS / "fable_quantumqueue_provenance.json").read_text())
    queue_stats = queue_meta["statistics"]

    depth_pivot = depth.pivot_table(index=["task", "size_label", "seed"], columns="depth_p",
                                    values="optimum_probability")
    depth_delta = boot_ci(depth_pivot[3] - depth_pivot[1])

    macros = [
        f"\\newcommand{{\\ClearEqMean}}{{{eq[0]:+.4f}}}",
        f"\\newcommand{{\\ClearEqLo}}{{{eq[1]:+.4f}}}",
        f"\\newcommand{{\\ClearEqHi}}{{{eq[2]:+.4f}}}",
        f"\\newcommand{{\\ClearNetWindows}}{{{int(trace.seed.nunique())}}}",
        f"\\newcommand{{\\ClearAbileneBlockMean}}{{{abilene_block[0]:+.4f}}}",
        f"\\newcommand{{\\ClearAbileneBlockLo}}{{{abilene_block[1]:+.4f}}}",
        f"\\newcommand{{\\ClearAbileneBlockHi}}{{{abilene_block[2]:+.4f}}}",
        f"\\newcommand{{\\ClearQiskitVersion}}{{2.3.1}}",
        f"\\newcommand{{\\FableDepthDelta}}{{{depth_delta[0]:+.4f}}}",
        f"\\newcommand{{\\FableDepthDeltaLo}}{{{depth_delta[1]:+.4f}}}",
        f"\\newcommand{{\\FableDepthDeltaHi}}{{{depth_delta[2]:+.4f}}}",
        f"\\newcommand{{\\FableQueueMedianMin}}{{{queue_stats['queue_minutes_q50']:.1f}}}",
        f"\\newcommand{{\\FableQueuePninetyMin}}{{{queue_stats['queue_minutes_q90']:.1f}}}",
        f"\\newcommand{{\\FableQueueOverTwoHours}}{{{100*queue_stats['fraction_over_120_minutes']:.1f}\\%}}",
    ]
    (TABLES / "clearaccept_numbers.tex").write_text("\n".join(macros) + "\n")
    report = {
        "baseline_summary": summary.to_dict("records"),
        "hardness_n16": h16.to_dict("records"),
        "stratified_qaoa_minus_local": {"mean": eq[0], "ci": [eq[1], eq[2]],
                                         "margin_curve": margin_curve},
        "shot_replicates": shot_summary.to_dict("records"),
        "netdata_5g": trace_summary.to_dict("records"),
        "abilene_qaoa_minus_local_block_ci": abilene_block,
        "compiled_n12": c12.to_dict("records"),
        "depth_sensitivity": depth_summary.to_dict("records"),
        "depth_p3_minus_p1": {"mean": depth_delta[0], "ci": [depth_delta[1], depth_delta[2]]},
        "transfer_aggregation": transfer_summary.to_dict("records"),
        "penalty_sensitivity": penalty_summary.to_dict("records"),
        "staleness_sensitivity": stale_summary.to_dict("records"),
        "mip_deadline": mip.to_dict("records"),
        "public_queue_replay": queue_summary.to_dict("records"),
        "public_queue_statistics": queue_stats,
    }
    (RESULTS / "clearaccept_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
