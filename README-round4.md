# Local fourth-round evidence and manuscript reproduction

The title remains **NetworkQBench-Wireless: Deadline-Aware Energy Accounting for Quantum-Assisted Server Placement in Network Control**. Public v1.2.2 is unchanged. The additions in `results/round4_20261019/` and the current presentation are published in release v1.2.3. No QPU measurement or Zenodo deposit was performed.

Use Python 3.14 with `requirements-energy.txt`, a C++17 compiler for the existing validation suite, and LaTeX/latexmk with txfonts, type1cm/type1ec and dvipng. The supplied cartoon assets and fonts have their licenses under `figure1_assets/`.

## Reproduce from stored evidence

From `artifact/` in a fresh extraction:

```sh
python reproduce_current.py --output output/current --validate
```

See [README-current.md](README-current.md) for optional asset copying and PDF compilation when the local manuscript-source package is present. The standalone artifact requires no manuscript tree.

The renderer runs the existing renderer chain, finishes with the current cartoon, generates new tables/macros, and creates captioned supplementary copies with stable labels. It never modifies released measurements. The Figure 1 consistency report checks the 10 ms queue, simulated outcomes, event integration, and 94.9% useful versus 8.39% applied rates (the current round-five panel also displays 8.39%). Plot fonts match the article's Times fonts; cartoon lettering is retained. Figure 4 labels the crossing of empirical mean curves. The captioned supplement is regenerated from the original tables, not hand-edited numerical copies.

## Optional remeasurement

Use a **new output folder**, and run the timing campaigns sequentially without other compute-heavy jobs:

```sh
python run_round4.py --input results/v1_2 --previous results/v1_2_1 --output output/remeasured-round4 --stages transfer --seed 20261019
python run_round4.py --input results/v1_2 --previous results/v1_2_1 --output output/remeasured-round4 --stages scaling --seed 20261019 --deadlines 0.1 0.4 0.8 --reference-limit 30
python analyze_round4.py --campaign output/remeasured-round4 --output output/remeasured-round4-analysis
```

Completed transfer families and scaling rows are resumable. Do not combine different environments into one resumed campaign. The archived campaign snapshots retain exact execution code; the superseded pilot-training snapshot predates a corrected duplicate-dictionary-key error at the first scaling row. No scaling measurement was stored by that failed first invocation. The successful scaling campaign has its own start/end times and source hashes. No measured result is replaced by a timing rerun.

## Campaigns and inference

- **Round4Transfer**, 19 October 2026: 32 direct fits (two objectives × four sizes × four power families), training seeds 0–14 only; 960 held-out circuit implementations; 115,200 paired sampled batches. Existing circular angles are controls. A family objective costs 15 instance probability evaluations; family calls are capped at the minimum historical individual-fit evaluation count across training seeds, so every instance respects its own historical count. All fits record actual calls, iterations, stop reason, angles and measured offline time. Useful mass is explicitly optimum-informed on training data. No new useful-mass claim is made at 16 variables.
- **Round4Deadline**, 19 October: 600 sequential randomized controller measurements, 100 unique sparse instances, 3 deadlines, 2 controllers; 40 new offline references. Workload generation is excluded because physical inputs are available at the boundary. MILP construction, solve, and result validation are included. A remaining-budget solver call can overrun: actual time gates deployment and incurs 50 W host charges. Construction exhaustion skips solving. Greedy is also rejected if late. Both controllers use the same benefit gate and keep-incumbent fallback.
- **Template paired reanalysis**: ten stored Template rounds, paired within each instance; each round's event ledger is evaluated before averaging. Differences are bootstrapped across 60 instance averages, not 600 independent measurements.

All new intervals use 20,000 paired resamples, seed 20261017. Physical studies stratify by size; scaling inference stays within each ten-seed `(servers, functions, deadline)` group. Sampling repetitions are averaged first. Intervals condition on the training split, fitted angles, calibration and timing records, and scenario assumptions. Round-mean ranges are implementation variability, not facility uncertainty. Previous frozen estimates retain their historical bootstrap draws.

`transfer_summary.csv` covers all budgets and both boundaries. `transfer_paired_summary.csv` gives direct-minus-circular comparisons under common Primary charges and separately measured full pipelines. `transfer_resources.csv` retains compiled counts/depth/durations. `deadline_runs.csv` stores raw measured candidates and overruns; `deadline_instances.csv` adds offline certification/bound gaps and paired energy; `deadline_summary.csv` preserves configuration-level inference. Missing bounds are not zero. Higher proportional function counts do not imply constant utilization; the capacity enlargement and fixed 4 ms chain-latency bound are retained.

## Validation

Run only in a fresh extraction: older suites may regenerate validation summaries under historical result folders.

```sh
python check_artifact.py
python test_benchmark.py
python test_energy.py
python test_v12.py
python validate_v12.py
python test_v121.py
python test_v122.py
python test_round4.py
```

The new suite checks exhaustive small sparse optima, training membership and evaluation caps, objective reconstruction, common profiles, 11,264 independent quantum event replays, all 600 sparse event ledgers, exhausted deadlines, late-result rejection, no-computation semantics, missing bounds, paired intervals and campaign source hashes. Historical hashes are checked outside the disposable extraction. Tests do not turn host timings into hardware telemetry.

C19 (infeasible-incumbent demand growth) and C20 (second-host and live-hardware validation) remain explicitly deferred. The current boundary lacks overload/service-loss recovery accounting.

The first direct-family campaign used a summed-family evaluation cap. Its records are preserved under `pilot_family_total_budget/` and excluded from the final scientific comparison. The final campaign enforces the stricter per-instance cap and records every historical per-seed count. Its snapshot is `source_snapshots/round4_transfer`; the pilot snapshot is `round4_transfer_family_total_budget`.
