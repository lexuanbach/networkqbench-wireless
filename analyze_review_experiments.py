#!/usr/bin/env python3
"""Analyze review-driven experiments and emit manuscript-ready assets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
TAB = ROOT / "generated"
ROW_END = r"\\"


def paired_cluster(frame: pd.DataFrame, value: str, a: str, b: str,
                   cluster: str, method: str = "algorithm",
                   seed: int = 20260902) -> tuple[float, float, float]:
    pivot = frame.pivot_table(index=cluster, columns=method, values=value, aggfunc="mean")
    diff = (pivot[a] - pivot[b]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    draws = diff[rng.integers(0, len(diff), (20_000, len(diff)))].mean(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def write_table(name: str, spec: str, header: str, rows: list[str]) -> None:
    (TAB / name.replace(".tex", ".txt")).write_text("\n".join([
        f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule", header,
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
    ]) + "\n")


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    constraint = pd.read_csv(RES / "review_constraint_aware.csv")
    csum = constraint.groupby(["task", "size_label", "algorithm"], as_index=False).agg(
        feasible_mass=("feasible_mass", "mean"), optimum_mass=("optimum_mass", "mean"),
        optimal_rate=("optimal", "mean"), optimizer_s=("optimizer_s", "median"))
    csum.to_csv(RES / "review_constraint_aware_summary.csv", index=False)
    rows = []
    labels = {"standard_x": "Standard X", "onehot_xy": "One-hot XY"}
    for r in csum[csum.size_label == 12].itertuples():
        rows.append(f"{r.task.title()} & {labels[r.algorithm]} & "
                    f"{r.feasible_mass:.3f} & {r.optimum_mass:.4f} & "
                    f"{100*r.optimal_rate:.1f}\\% & {r.optimizer_s:.3f} " + ROW_END)
    write_table("review_constraint_aware.tex", "llrrrr",
                r"Task & Mixer & Feas. mass & Opt. mass & Opt. return & Fit s \\", rows)

    indexed = constraint.assign(cluster=constraint.task + ":" +
                                constraint.size_label.astype(str) + ":" +
                                constraint.seed.astype(str))
    feasibility_gain = paired_cluster(indexed, "feasible_mass", "onehot_xy",
                                      "standard_x", "cluster")
    optimum_gain = paired_cluster(indexed, "optimum_mass", "onehot_xy",
                                  "standard_x", "cluster", seed=20260903)

    closed = pd.read_csv(RES / "review_closed_loop_5g.csv")
    closed_summary = closed.groupby("method", as_index=False).agg(
        stations=("station", "nunique"), decisions=("window", "size"),
        next_max_load=("next_max_effective_load", "mean"),
        next_cost=("next_cost", "mean"), deadline_miss=("deadline_miss", "mean"),
        switches=("switches", "mean"), median_delay_s=("delay_s", "median"))
    closed_summary.to_csv(RES / "review_closed_loop_summary.csv", index=False)
    order = ["milp", "rolling", "task_specific", "qaoa_transfer"]
    labels = {"milp": "HiGHS MILP", "rolling": "Rolling",
              "task_specific": "Task-specific",
              "qaoa_transfer": "Transferred QAOA"}
    rows = []
    for method in order:
        r = closed_summary.set_index("method").loc[method]
        delay = (f"{r.median_delay_s / 60:.2f} min" if method == "qaoa_transfer"
                 else (r"$<0.01$ ms" if 1e3 * r.median_delay_s < 0.005
                       else f"{1e3 * r.median_delay_s:.2f} ms"))
        rows.append(f"{labels[method]} & {r.next_max_load:.3f} & "
                    f"{r.next_cost:.3f} & {100*r.deadline_miss:.1f}\\% & "
                    f"{r.switches:.2f} & {delay} " + ROW_END)
    write_table("review_closed_loop.tex", "lrrrrr",
                r"Method & Next max load & Next cost & Miss & Switches & Median delay \\", rows)
    rolling_vs_qaoa = paired_cluster(closed, "next_max_effective_load", "qaoa_transfer",
                                     "rolling", "station", method="method")
    milp_vs_rolling = paired_cluster(closed, "next_max_effective_load", "rolling",
                                    "milp", "station", method="method", seed=20260904)

    resources = pd.read_csv(RES / "review_full_circuit_resources.csv")
    r12 = resources[(resources.n_vars == 12) & (resources.topology == "grid4x4")]
    variant_labels = {
        "quadratic_lower_bound": "Interaction lower bound",
        "exact_full_x": "Complete X",
        "exact_full_onehot": "Complete one-hot",
    }
    rows = [f"{r.task.title()} & {variant_labels[r.variant]} & {int(r.cx)} & "
            f"{int(r.depth)} & {r.transpile_s:.2f} " + ROW_END
            for r in r12.itertuples()]
    write_table("review_full_resources.tex", "llrrr",
                r"Task & Circuit & CX & Depth & Compile s \\", rows)

    emulation = pd.read_csv(RES / "review_backend_emulation.csv")
    esum = emulation.groupby("task", as_index=False).agg(
        instances=("seed", "size"), optimal_return=("optimal_return", "mean"),
        observed_optimum_frequency=("observed_optimum_frequency", "mean"),
        ideal_optimum_probability=("ideal_optimum_probability", "mean"),
        compiled_cx=("compiled_cx", "mean"), local_pipeline_s=("local_pipeline_s", "median"))
    esum.to_csv(RES / "review_backend_emulation_summary.csv", index=False)
    rows = [f"{r.task.title()} & {int(r.instances)} & {r.ideal_optimum_probability:.3f} & "
            f"{r.observed_optimum_frequency:.3f} & {100*r.optimal_return:.1f}\\% & "
            f"{r.compiled_cx:.0f} " + ROW_END for r in esum.itertuples()]
    write_table("review_backend_emulation.tex", "lrrrrr",
                r"Task & Inst. & Ideal opt. mass & Noisy frequency & Opt. return & CX \\", rows)

    macros = [
        f"\\newcommand{{\\ReviewFeasGain}}{{{feasibility_gain[0]:+.3f}}}",
        f"\\newcommand{{\\ReviewFeasGainLo}}{{{feasibility_gain[1]:+.3f}}}",
        f"\\newcommand{{\\ReviewFeasGainHi}}{{{feasibility_gain[2]:+.3f}}}",
        f"\\newcommand{{\\ReviewOptMassGain}}{{{optimum_gain[0]:+.4f}}}",
        f"\\newcommand{{\\ReviewOptMassGainLo}}{{{optimum_gain[1]:+.4f}}}",
        f"\\newcommand{{\\ReviewOptMassGainHi}}{{{optimum_gain[2]:+.4f}}}",
        f"\\newcommand{{\\ReviewClosedStations}}{{{closed.station.nunique()}}}",
        f"\\newcommand{{\\ReviewClosedTransitions}}{{{closed[['station','window']].drop_duplicates().shape[0]}}}",
        f"\\newcommand{{\\ReviewRollingVsQaoa}}{{{rolling_vs_qaoa[0]:+.3f}}}",
        f"\\newcommand{{\\ReviewRollingVsQaoaPrecise}}{{{rolling_vs_qaoa[0]:.4f}}}",
        f"\\newcommand{{\\ReviewRollingVsQaoaLo}}{{{rolling_vs_qaoa[1]:+.3f}}}",
        f"\\newcommand{{\\ReviewRollingVsQaoaHi}}{{{rolling_vs_qaoa[2]:+.3f}}}",
        f"\\newcommand{{\\ReviewMilpVsRolling}}{{{milp_vs_rolling[0]:+.3f}}}",
        f"\\newcommand{{\\ReviewEmuInstances}}{{{len(emulation)}}}",
    ]
    (TAB / "review_numbers.txt").write_text("\n".join(macros) + "\n")
    report = {
        "constraint_aware": csum.to_dict("records"),
        "constraint_aware_paired": {"feasible_mass_gain": feasibility_gain,
                                     "optimum_mass_gain": optimum_gain},
        "closed_loop": closed_summary.to_dict("records"),
        "closed_loop_station_clustered": {
            "qaoa_minus_rolling_next_max_load": rolling_vs_qaoa,
            "rolling_minus_milp_next_max_load": milp_vs_rolling,
        },
        "full_resources_n12_grid": r12.to_dict("records"),
        "backend_emulation": esum.to_dict("records"),
    }
    (RES / "review_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
