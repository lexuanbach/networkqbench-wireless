# Reproduction for version 1.2.2

This artifact accompanies **NetworkQBench-Wireless: Deadline-Aware Energy Accounting for Quantum-Assisted Server Placement in Network Control**. It contains the unchanged historical measurements, the round-two additions, and the round-three analyses and symbolic-template experiment. Quantum quality is simulated, host durations are measured, and electricity is modeled. Manuscripts and internal reviews are excluded.

## Regenerate publication outputs

Use Python 3.14 and the pinned dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-energy.txt
.venv/bin/python analyze_v122.py
.venv/bin/python render_v122.py
```

Figures, tables, and numeric macros appear in `output/v1_2_2/`. The renderer also regenerates historical supplementary tables from their stored inputs. It writes no historical measurements. `--manuscript PATH` copies generated files into an independently provided manuscript directory. Figure 1 uses bundled Matplotlib code and fonts.

The analyzers retain original primary and demand intervals where those results are unchanged. New paired contrasts resample 20,000 times within size, averaging sampling repetitions before resampling. Repeated timing rounds do not increase the number of problem instances.

## Template experiment

```sh
.venv/bin/python run_v122.py --input results/v1_2 --output results/template-rerun --rounds 10 --seed 20261018
.venv/bin/python analyze_v122.py --input results/v1_2 --previous results/v1_2_1 --output results/template-rerun
.venv/bin/python render_v122.py --results results/template-rerun --previous results/v1_2_1 --output output/template-rerun
```

Run the timing experiment without other compute-intensive jobs. It warms both implementations, randomizes case and implementation order, and measures 60 held-out cases in ten rounds. Symbolic templates parameterize assignment phases and mixer angles; numeric and symbolic variants use the same topology, compiler seed, and gate-duration model. Binding does not transpile. Template compilation is amortized over 100 decisions in addition to existing angle-fitting costs. The circuit structure retains zero-angle gates, so reduced preparation need not imply reduced allocated energy.

All 60 bound distributions are compared to the logical reference after undoing the physical layout. The run writes UTC timestamps, seed membership, platform, source hashes, timing rounds, compilation resources, and equivalence errors. New host timings will differ from shipped records.

## Timing campaigns and comparisons

| Campaign | Archived date | Scope |
|---|---|---|
| Primary | 16 October 2026 | Physical pipelines, all-powered/sleep families, admission and primary classical references |
| Stress | 15 October 2026 | Concentrated-incumbent experiment and matched classical timings |
| Follow-up | 17 October 2026 | Useful-mass pipeline, larger physical cases, sparse classical scaling |
| Demand | 17 October 2026 | Remeasured classical controllers under stronger preceding-demand changes |
| Repeated | 17 October 2026 | Ten randomized timing rounds |
| Template | 18 October 2026 | Matched numeric compilation and symbolic binding |

Earlier campaign dates follow archived file timestamps and execution records, not reconstructed exact start times. Template has explicit UTC start/end timestamps. The publication's primary table and five figures use Primary. Demand energy combines Primary quantum and Demand classical timings; the MILP has no warm start and its current workload is unchanged. Follow-up useful-mass and Template energy use Primary HiGHS. Template's two implementations are measured together; selection profiles and angle-fitting charges remain Primary.

The common-profile objective comparison fixes all Primary times and computation charges, including offline allocation, and changes only stored candidate outcomes. It isolates candidate effects rather than estimating a deployed useful-mass pipeline's complete cost. The latter remains separately reported with its actual fitting charges.

## New records

`results/v1_2_2/` includes:

- `keep_paired.csv`, `crossing.csv`: paired QAOA-minus-keep outcomes and the crossing found by replaying the full gate, not plot interpolation.
- `training_main.csv`, `below_uniform.csv`, `angle_diagnosis.csv`: per-size interpretation, heterogeneity, and noncausal angle diagnostics.
- `common_profile_*.csv`, `objective_paired.csv`: shared-profile replay and paired objective comparison.
- `demand_*.csv`: keep-incumbent and both classical references under stronger preceding demand.
- `occupancy_budgets.csv`, `scaling_quality*.csv`: unrounded budgets, exact greedy gaps, lower-bound guarantees, missing bounds, and timeout status.
- `sensitivity_*.csv`, `zero_transpilation_summary.csv`: complete scenario contrasts and explicitly idealized removal of numeric transpilation alone.
- `template_*.csv`, `template_manifest.json`: new measured timing rounds, offline costs, resources, simulated equivalence, and derived ledgers.
- `historical_sha256.json`, `analysis_manifest.json`, `validation.json`: preservation, analysis provenance, and independent checks.

## Validate

```sh
.venv/bin/python check_artifact.py
.venv/bin/python test_benchmark.py
.venv/bin/python test_energy.py
.venv/bin/python test_v12.py
.venv/bin/python validate_v12.py
.venv/bin/python test_v121.py
.venv/bin/python test_v122.py
```

The final test checks the shipped records and independently reconstructs representative templates and every common-profile action event. Its preservation audit applies to the shipped historical inputs, not independently remeasured replacements. Full previous experiment protocols remain in README-revision.md and README-energy.md. Public extension contracts and earlier entry-point defaults are unchanged.
