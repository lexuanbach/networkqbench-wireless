#!/usr/bin/env python3
"""Cartoon-style Figure 1, using frozen records and bundled drawing assets.
This local presentation revision does not alter the published v1.2.2 archive.
"""
import argparse, ast, hashlib, json, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ART = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--results', type=Path, default=ART/'results/v1_2/all_on')
ap.add_argument('--output', type=Path, default=ART/'output/figure1_cartoon')
ap.add_argument('--manuscript', type=Path)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ART/'figure1_assets'))
import energy_v12 as E
import comic as C
from paper_plot_style import apply_paper_font
PAL = C.PAL
SHOTS = 64
HORIZON_S = 60.0
INS = pd.read_csv(args.results / "instances.csv")
INS = INS[INS.seed >= 15]
CASE = INS[INS.incumbent_w - INS.optimum_w > 1e-7].sort_values(["size", "seed"]).iloc[0]   # same rule as render_v12
PI = pd.read_csv(args.results / "primary_instances.csv")
ALL_PI = PI.copy()
PI = PI[(PI["size"] == CASE["size"]) & (PI.seed == CASE.seed)]
S = pd.read_csv(args.results / "primary_summary.csv")
HELDOUT_Q = ALL_PI[(ALL_PI.seed >= 15) &
                   (ALL_PI.algorithm == "onehot_structured") &
                   (ALL_PI.shots == SHOTS)]
USEFUL_PERCENT = 100 * HELDOUT_Q.useful.mean()
APPLIED_PERCENT = 100 * HELDOUT_Q.accepted.mean()


def row(alg, shots=0):
    r = PI[(PI.algorithm == alg) & (PI.shots == shots)]
    assert len(r) == 1
    return r.iloc[0]


def summ(alg, shots):
    r = S[(S.algorithm == alg) & (S.shots == shots)]
    assert len(r) == 1
    return r.iloc[0]


# ---- the walkthrough case
inst = E.workload(int(CASE["size"]), int(CASE.seed))
m = inst.metadata
nn = m["nodes"]
mil, qs, qu = row("milp"), row("onehot_structured", SHOTS), row("quantum_uniform_onehot", SHOTS)
classical = pd.read_csv(args.results / "classical.csv")
frozen_milp = classical[(classical["size"] == CASE["size"]) &
                        (classical.seed == CASE.seed) & (classical.algorithm == "milp")]
assert len(frozen_milp) == 1
sol = int(frozen_milp.iloc[0].solution)
samples = pd.read_csv(args.results / "samples.csv")
case_samples = samples[(samples["size"] == CASE["size"]) & (samples.seed == CASE.seed) &
                       (samples.algorithm == "onehot_structured") & (samples.shots == SHOTS)]
assert len(case_samples) == 30 and case_samples["repeat"].nunique() == 30
optimal_batches = int(case_samples.optimal.sum())
prior = np.array(ast.literal_eval(CASE.prior_demands))


def watts(state, demands):
    a = E.decode(inst, state)
    load = np.bincount(a, weights=demands, minlength=nn)
    return a, load / m["capacities"], E.server_power(np.clip(load / m["capacities"], 0, 1))


inc_a, inc_u, inc_w = watts(int(CASE.incumbent), m["demands"])
opt_a, opt_u, opt_w = watts(sol, m["demands"])
assert abs(inc_w.sum() - CASE.incumbent_w) < 1e-6 and abs(opt_w.sum() - CASE.optimum_w) < 1e-6
assert abs(inst.deadline_s - CASE.deadline_s) < 1e-9
moved = [k for k in range(len(inc_a)) if inc_a[k] != opt_a[k]]
assert abs(mil.transition_j - 10.0 * len(moved)) < 1e-9
stay_j = CASE.incumbent_w * HORIZON_S
extra = qs.total_j - mil.total_j
assert abs(extra - qs.excess_j) < 1e-6
assert qs.timely == 1.0 and qs.quality == 1.0


def verify_consistency():
    """Check the drawn case using independent event integration and frozen inputs."""
    assert len(HELDOUT_Q) == 60 and HELDOUT_Q.groupby('size').size().eq(15).all()
    assert abs(USEFUL_PERCENT / 100 - summ('onehot_structured', SHOTS).useful) < 1e-9
    assert HELDOUT_Q[['useful', 'accepted']].ge(0).all().all()
    assert HELDOUT_Q[['useful', 'accepted']].le(1).all().all()
    # Independently replay every displayed deployment rate from sampled actions.
    heldout_samples = samples[(samples.seed >= 15) &
                              (samples.algorithm == 'onehot_structured') &
                              (samples.shots == SHOTS)]
    accepted_count = 0
    for r in HELDOUT_Q.itertuples():
        problem = E.workload(r.size, r.seed)
        instance_row = INS[(INS['size'] == r.size) & (INS.seed == r.seed)].iloc[0]
        batches = heldout_samples[(heldout_samples['size'] == r.size) &
                                  (heldout_samples.seed == r.seed)]
        assert len(batches) == 30 and batches['repeat'].nunique() == 30
        accepted = sum(E.event_energy(problem, [(r.ready_s, int(b.solution), 'QAOA')],
                                     E.Scenario(), incumbent=int(instance_row.incumbent))['accepted']
                       for b in batches.itertuples())
        assert abs(accepted / 30 - r.accepted) < 1e-9
        accepted_count += accepted
    assert abs(100 * accepted_count / len(heldout_samples) - APPLIED_PERCENT) < 1e-9
    preceding = E.workload(int(CASE['size']), int(CASE.seed), demands=prior)
    assert (prior >= m['demands']).all()
    assert preceding.feasible[int(CASE.incumbent)]
    assert abs(preceding.metadata['power_w'][int(CASE.incumbent)] -
               preceding.metadata['power_w'][preceding.feasible].min()) < 1e-8
    scenario = E.Scenario()
    replay = [E.event_energy(inst, [(qs.ready_s, int(r.solution), "QAOA")],
                            scenario, incumbent=int(CASE.incumbent))
              for r in case_samples.itertuples()]
    classical = E.event_energy(inst, [(mil.ready_s, sol, "HiGHS")],
                               scenario, incumbent=int(CASE.incumbent))
    fields = ['network_j', 'transition_j', 'delay_loss_j', 'quality_loss_j', 'fallback_loss_j']
    errors = {key: float(np.mean([r[key] for r in replay]) - qs[key]) for key in fields}
    assert max(abs(v) for v in errors.values()) < 1e-7
    for key in fields:
        assert abs(classical[key] - mil[key]) < 1e-7
    accepted = sum(r['accepted'] for r in replay)
    assert accepted == optimal_batches == 29
    assert sum(r['applied_action'] == int(CASE.incumbent) for r in replay) == 1
    assert qs.quality_loss_j == 0 and mil.quality_loss_j == 0
    assert case_samples.quality.all() and case_samples.feasible.all()
    for r in (mil, qs):
        component_sum = sum(r[key] for key in ['preparation_j', 'optimization_j',
                            'communication_j', 'postprocessing_j', 'offline_j', 'service_j'])
        assert abs(component_sum - r.compute_j) < 1e-8
        assert abs(r.compute_j + r.network_j + r.transition_j - r.total_j) < 1e-7
    assert abs(qs.compute_j - mil.compute_j + qs.network_j - mil.network_j +
               qs.transition_j - mil.transition_j - extra) < 1e-7
    for algorithm, shots in [('milp', 0), ('heuristic', 0), ('classical_uniform_onehot', 64),
                             ('quantum_uniform_onehot', 64), ('onehot_structured', 64),
                             ('onehot_transfer', 64)]:
        rows = ALL_PI[(ALL_PI.seed >= 15) & (ALL_PI.algorithm == algorithm) & (ALL_PI.shots == shots)]
        assert len(rows) == 60 and rows.groupby('size').size().eq(15).all()
        assert abs(rows.compute_j.mean() - summ(algorithm, shots).compute_j) < 1e-8
    optimal_incumbents = int(np.isclose(INS.incumbent_w, INS.optimum_w).sum())
    assert optimal_incumbents == 54 and len(INS) == 60
    report = dict(size=int(CASE['size']), seed=int(CASE.seed), batches=30,
                  optimal_batches=optimal_batches, accepted_batches=accepted,
                  retained_batches=1, current_optimal_incumbents=optimal_incumbents,
                  per_shot_exact_probability=float(qs.p_optimal),
                  per_shot_useful_probability=float(qs.p_quality),
                  extra_total_j=float(extra), network_gap_j=float(qs.network_j-mil.network_j),
                  delay_gap_j=float(qs.delay_loss_j-mil.delay_loss_j),
                  retention_gap_j=float(qs.fallback_loss_j-mil.fallback_loss_j),
                  applied_quality_loss_j=float(qs.quality_loss_j), event_errors_j=errors,
                  heldout_instances=len(HELDOUT_Q),
                  useful_candidate_percent=float(USEFUL_PERCENT),
                  beneficial_action_percent=float(APPLIED_PERCENT),
                  beneficial_action_replays=len(heldout_samples),
                  accepted_beneficial_actions=int(accepted_count),
                  aggregate_computation_values_match=True,
                  inputs={name: hashlib.sha256((args.results/name).read_bytes()).hexdigest()
                          for name in ['instances.csv', 'primary_instances.csv',
                                       'primary_summary.csv', 'classical.csv', 'samples.csv']})
    (args.output/'fig1_consistency.json').write_text(json.dumps(report, indent=2)+'\n')


H, FW = 3.65, 7.4
YB, YT = 1.62, 3.40
X = lambda i: i / FW
Y = lambda i: i / H


def build():
    fig = C.figure(FW, H)
    C.label(fig, 0.5, Y(H - 0.12),
            "A timely quantum candidate can still cost more energy than a classical decision",
            fs=9, bold=True)
    ws = [1.65, 1.65, 1.78, 1.87]
    gap = 0.12
    xs = [0.03]
    for w in ws[:-1]:
        xs.append(xs[-1] + w + gap)
    top = YT - 0.255 - 0.07
    bot = YB + 0.07
    meta = [("Demand drops", "held-out case, 4 servers", PAL["mustard"], "fc_electrical_threshold"),
            ("HiGHS MILP", "exact solver, 50 W host", PAL["slate"], "fc_services"),
            ("Structured QAOA", f"one-hot, p = 1, {SHOTS} shots", PAL["blue"], "fs_cloud"),
            ("Energy over 60 s", "mean of 30 sampling batches", PAL["red"], "fs_receipt")]
    for i, ((t, s_, col, hi), x, w) in enumerate(zip(meta, xs, ws)):
        C.card(fig, X(x), Y(YB), X(w), Y(YT - YB), n=i + 1, title=t, subtitle=s_, color=col, hicon=hi)
    # Stage markers using the shared illustration primitives: arrows between stages, a "vs" badge between the two
    # alternative controllers (HiGHS and QAOA), and a red arrow into the cost card.
    ymid = Y((YB + YT) / 2)
    C.step_arrow(fig, X(xs[0] + ws[0]), X(xs[1]), ymid)
    C.badge(fig, X(xs[1] + ws[1] + gap / 2), ymid, "vs", r=0.0125, z=9)
    C.step_arrow(fig, X(xs[2] + ws[2]), X(xs[3]), ymid, color=PAL["red"])
    L = 0.075

    def box(x, ytop, lines, w, **kw):
        cw, ch = C.code(fig, X(x), Y(ytop), lines, w=X(w), fs=kw.pop("fs", 6.2), **kw)
        return (X(x), Y(ytop) - ch, X(x) + cw, Y(ytop))

    def place(name, size, vis_left=None, vis_top=None, vis_cy=None, **kw):
        ab = C.icon(fig, name, 0.5, 0.5, size=size / FW, **kw)
        fig.canvas.draw()
        bb = ab.get_window_extent(fig.canvas.get_renderer())
        d = fig.dpi
        x0, y0, x1, y1 = bb.x0 / d, bb.y0 / d, bb.x1 / d, bb.y1 / d
        v = ab._comic_vis
        w, h = x1 - x0, y1 - y0
        vb = [x0 + v[0] * w, y0 + v[1] * h, x0 + v[2] * w, y0 + v[3] * h]
        dx = vis_left - vb[0]
        dy = (vis_top - vb[3]) if vis_top is not None else (vis_cy - (vb[1] + vb[3]) / 2)
        ab.remove()
        ic = C.icon(fig, name, 0.5 + dx / FW, 0.5 + dy / H, size=size / FW, **kw)
        return ic, [vb[0] + dx, vb[1] + dy, vb[2] + dx, vb[3] + dy]

    def actor(x, ytop, nm, size, caption, cy=None, **kw):
        if cy is None:
            ic, v = place(nm, size, vis_left=x + L, vis_top=ytop, **kw)
        else:
            ic, v = place(nm, size, vis_left=x + L + 0.02, vis_cy=cy, **kw)
        C.label(fig, X((v[0] + v[2]) / 2), Y(v[1] - 0.095), caption, fs=6.3, color=PAL["slate"], z=7)
        return ic, v

    def chart(x0, x1, ytop, ybot, vals, cols, tcol):
        ax = C.plot_axes(fig, [X(x0), Y(ybot), X(x1 - x0), Y(ytop - ybot)])
        ax.bar(range(nn), vals, color=cols, edgecolor=PAL["ink"], lw=0.8, width=0.66)
        for j, v_ in enumerate(vals):
            ax.text(j, v_ + 25, f"{v_:.0f}", ha="center", va="bottom", fontsize=5.6, color=PAL["ink"])
        ax.set_xticks(range(nn), [f"s{j + 1}" for j in range(nn)])
        ax.set_ylim(0, 1000); ax.set_xlim(-0.5, nn - 0.5)
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
        ax.tick_params(labelsize=6.0)
        ax.text(nn - 0.5, 990, f"total {vals.sum():,.0f} W", ha="right", va="top", fontsize=6.6, color=tcol, fontweight="bold")
        return ax

    def axbox(ax):
        fig.canvas.draw()
        bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        return (bb.x0, bb.y0, bb.x1, bb.y1)

    def stampbox(t):
        fig.canvas.draw()
        bb = t.get_bbox_patch().get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        return (bb.x0, bb.y0, bb.x1, bb.y1)

    fn = lambda a: " ".join(f"f{k + 1}->s{a[k] + 1}" for k in range(len(a)))
    # ---- card 1: demand drops, the previous optimum is now off
    x, w = xs[0], ws[0]
    c1 = box(x + L, top, [f"demand {prior[0]:.2f} {prior[1]:.2f} -> {m['demands'][0]:.0f} {m['demands'][1]:.0f}",
                          f"keep {fn(inc_a)}",
                          "all servers stay on",
                          f"deadline {1000 * inst.deadline_s:.0f} ms"], w - 2 * L, highlight={1: "mustard"})
    rt = c1[1] * H - 0.07
    ic, v = actor(x, rt - 0.167, "fs_gauge", 0.30, "SPEC curve")   # icon lowered ~9 pt to sit level with the bars
    ax = chart(x + w - L - 0.58, x + w - L, rt - 0.02, bot + 0.14, inc_w,
               [PAL["mustard"] if u > 0 else PAL["slate"] for u in inc_u], PAL["mustard"])
    C.link(fig, ic, (axbox(ax)[0], Y(v[1]), axbox(ax)[0], Y(v[3])), "prices")
    # ---- card 2: HiGHS consolidates
    x, w = xs[1], ws[1]
    k = moved[0]
    c2 = box(x + L, top, [f"move f{k + 1}: s{inc_a[k] + 1} -> s{opt_a[k] + 1}",
                          f"saves {CASE.incumbent_w - CASE.optimum_w:.2f} W",
                          f"time {1000 * mil.ready_s:.1f} ms",
                          f"cost {mil.compute_j:.2f} J"], w - 2 * L, highlight={0: "green"})
    rt = c2[1] * H - 0.07
    ic, v = actor(x, rt - 0.167, "fc_command_line", 0.30, "HiGHS")   # icon lowered ~9 pt to sit level with the bars
    ax = chart(x + w - L - 0.61, x + w - L, rt - 0.02, bot + 0.14, opt_w,
               [PAL["green"] if u > 0 else PAL["slate"] for u in opt_u], PAL["green"])
    C.link(fig, ic, (axbox(ax)[0], Y(v[1]), axbox(ax)[0], Y(v[3])), "places")
    # ---- card 3: structured one-hot QAOA against uniform one-hot sampling
    x, w = xs[2], ws[2]
    c3 = box(x + L, top, [f"{int(CASE.onehot_states)} one-hot states",
                          f"P(opt) QAOA    {100 * qs.p_optimal:.2f}%",
                          f"P(opt) uniform {100 * CASE.uniform_optimum_probability:.2f}%",
                          f"optimum: {optimal_batches}/30 batches",
                          f"cost {qs.compute_j:.2f} J"], w - 2 * L, fs=6.6, highlight={1: "red", 3: "green", 4: "red"})
    rt = c3[1] * H - 0.07
    ic, v = actor(x, None, "atom_symbol", 0.36, "Simulated Q", cy=(rt + bot + 0.14) / 2)
    st = C.stamp(fig, X((v[2] + 0.40 + x + w - L) / 2), Y((v[1] + v[3]) / 2), f"ON TIME\n{1000 * qs.ready_s:.1f} ms",
                 color=PAL["green"], fs=10, rot=-4)
    C.link(fig, ic, stampbox(st), "")
    # ---- card 4: ledger over the 60 s horizon
    x, w = xs[3], ws[3]
    f = lambda a: f"{a:>11,.2f}"   # wider columns so the two ledger columns read apart
    c4 = box(x + L, top, ["              HiGHS        QAOA",
                          f"compute {f(mil.compute_j)}  {f(qs.compute_j)}",
                          f"trans.  {f(mil.transition_j)}  {f(qs.transition_j)}",
                          f"servers {f(mil.network_j)}  {f(qs.network_j)}",
                          f"total   {f(mil.total_j)}  {f(qs.total_j)}",
                          f"saved   {f(stay_j - mil.total_j)}  {f(stay_j - qs.total_j)}",
                          f"server gap: {qs.network_j-mil.network_j:.2f} J"], w - 2 * L,
             fs=5.5, highlight={4: "red", 5: "green"})
    rt = c4[1] * H - 0.07
    ic, v = actor(x, None, "low_battery", 0.40, "QAOA path", cy=(rt + bot + 0.12) / 2, rotate=90)
    st = C.stamp(fig, X((v[2] + 0.40 + x + w - L) / 2), Y((v[1] + v[3]) / 2), f"+{extra:.2f} J\nvs HiGHS",
                 color=PAL["red"], fs=10, rot=-5)
    C.link(fig, ic, stampbox(st), "", color=PAL["red"])

    C.label(fig, .5, Y(1.50),
            f"Case n = {int(CASE['size'])}, seed {int(CASE.seed)}: {optimal_batches}/30 QAOA batches deploy the optimum. One retains the incumbent.",
            fs=6.3, color=PAL["slate"])
    bars = [("milp", 0, "HiGHS MILP", PAL["slate"]),
            ("heuristic", 0, "Greedy", PAL["slate"]),
            ("classical_uniform_onehot", SHOTS, "CPU\none-hot", PAL["plum"]),
            ("quantum_uniform_onehot", SHOTS, "Q\none-hot", PAL["mustard"]),
            ("onehot_structured", SHOTS, "Structured\nQAOA", PAL["blue"]),
            ("onehot_transfer", SHOTS, "Dense\nQAOA", PAL["red"])]
    ax = C.formal_axes(fig, [X(.55), Y(.22), X(3.95), Y(.96)],
                       title="(a) Mean computation energy")
    for j, (alg, sh, _, col) in enumerate(bars):
        value = summ(alg, sh).compute_j
        ax.bar(j, value, color=col, ec=PAL["ink"], lw=.6, width=.55)
        ax.text(j, value * 1.5, f"{value:.2f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_yscale("log"); ax.set_ylim(.003, 3000)
    ax.set_yticks([.01, 1, 100]); ax.set_ylabel("Modeled joules (log)")
    ax.set_xticks(range(len(bars)), [b[2] for b in bars], fontsize=6.5)
    C.formalize(ax)
    rates = C.formal_axes(fig, [X(4.95), Y(.22), X(2.30), Y(.96)],
                          title="(b) Structured QAOA outcomes")
    for y, value, label, color in [(1, USEFUL_PERCENT, "Useful candidate", PAL['blue']),
                                   (0, APPLIED_PERCENT, "Beneficial action applied", PAL['green'])]:
        rates.barh(y, 100, height=.27, color='#EDF0F2', zorder=2)
        rates.barh(y, value, height=.27, color=color, zorder=3)
        rates.text(0, y + .24, label, fontsize=6.8, va='bottom')
        rates.text(103, y, (f'{value:.2f}%' if y==0 else f'{value:.1f}%'), fontsize=6.8, va='center')
    rates.set_xlim(0, 125); rates.set_ylim(-.40, 1.65)
    rates.set_xticks([0, 50, 100]); rates.set_xlabel('Batches (%)', labelpad=2)
    rates.set_yticks([]); rates.spines['left'].set_visible(False)
    C.formalize(rates)
    rates.grid(False)
    C.label(fig, .5, Y(1.36), "Both plots: 60 held-out cases, 64 shots per sampler. Primary: 20 kW, u = 1. Keep incumbent: 0 J (omitted on log axis).", fs=6.2, color=PAL["slate"])
    apply_paper_font(fig, axes_only=True)
    return fig


def write_tex():
    cap = ("One positive-headroom case ($n=8$, seed 21) illustrates alternative HiGHS and "
           "structured-QAOA paths. Cards average 30 batches over 60~s. Plots show "
           "mean computation and structured-QAOA outcomes over 60 held-out cases at 64 shots. "
           "Primary timing and the base allocated-power scenario apply. Energy is modeled "
           "and quantum quality is simulated.")
    vals = {"FigOneExtraJ": f"{extra:.2f}", "FigOneSavedW": f"{CASE.incumbent_w - CASE.optimum_w:.2f}",
            "FigOneQaoaPopt": f"{100 * qs.p_optimal:.2f}", "FigOneUniformPopt": f"{100 * CASE.uniform_optimum_probability:.2f}",
            "FigOneQaoaJ": f"{qs.compute_j:.2f}", "FigOneMilpJ": f"{mil.compute_j:.2f}",
            "FigOneNetworkGapJ": f"{qs.network_j-mil.network_j:.2f}",
            "FigOneDelayGapJ": f"{qs.delay_loss_j-mil.delay_loss_j:.2f}",
            "FigOneRetentionJ": f"{qs.fallback_loss_j-mil.fallback_loss_j:.2f}",
            "FigOneUsefulPct": f"{USEFUL_PERCENT:.1f}",
            "FigOneAppliedPct": f"{APPLIED_PERCENT:.2f}"}
    with open(args.output / "fig1_cartoon.tex", "w") as fh:
        fh.write("% Written by artifact/render_fig1_cartoon.py; do not edit by hand.\n")
        fh.write("\\newcommand{\\FigOneCartoonCaption}{" + cap + "}\n")
        for k_, v_ in vals.items():
            fh.write(f"\\newcommand{{\\{k_}}}{{{v_}}}\n")


if __name__ == "__main__":
    verify_consistency()
    write_tex()
    C.save(build(), str(args.output / "fig1_cartoon.pdf"))
    if args.manuscript:
        import shutil
        for filename in ["fig1_cartoon.pdf", "fig1_cartoon_preview.png"]:
            shutil.copy2(args.output / filename, args.manuscript / "figures" / filename)
        shutil.copy2(args.output / "fig1_cartoon.tex", args.manuscript / "tables/fig1_cartoon.tex")
