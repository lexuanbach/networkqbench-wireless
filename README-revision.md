# Revision 1.2.1: server-placement accounting and additional experiments

This locally prepared artifact accompanies **NetworkQBench-Wireless: Deadline-Aware Energy Accounting for Quantum-Assisted Server Placement in Network Control**. The current successor is version 1.2.2; see [README-round3.md](README-round3.md). The existing public 1.2.0 release contains the original experiment. This package includes the additional experiments and a standalone renderer for the revised article.

## Reproduce current publication outputs

Use Python 3.14 and the pinned `requirements-energy.txt` dependencies. Run commands from this artifact directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-energy.txt
.venv/bin/python analyze_v121.py
.venv/bin/python render_v121.py
```

Outputs go to `output/v1_2_1/`. Figure 1 is generated directly with Matplotlib; no sibling project, external font, or icon library is required. The renderer does not rewrite frozen measurements. Original primary HiGHS-reference intervals are retained for unchanged rows. New incumbent and greedy contrasts use 20,000 size-stratified paired bootstrap resamples. `render_v121.py --manuscript PATH` copies figures, tables, and macros to an explicitly chosen manuscript directory.

## New measurements

To rerun all additional studies, choose a new output directory so the shipped measurements remain available:

```sh
.venv/bin/python run_v121.py --output results/my-rerun
.venv/bin/python analyze_v121.py --output results/my-rerun
.venv/bin/python render_v121.py --results results/my-rerun --output output/my-rerun
```

`run_v121.py` supports `--stages training demand larger scaling timing`, `--training-objective useful_mass`, `--demand-range 0.2 0.5`, `--servers 20 50 100`, `--functions 4 8`, `--seeds 0 1 2 3 4 5 6 7 8 9`, and `--time-limit 30`. Defaults implement the manuscript protocol. `expectation` is an optional fitting-objective control; its stage manifest identifies the objective. The manuscript reports the useful-mass run only. Scaling checkpoints completed cases, so rerunning that stage resumes absent cases. Run timing after other compute-intensive studies have finished.

Fresh host timings need not reproduce frozen joules. The demand experiment reuses current quantum distributions and timings but remeasures incumbent-dependent classical controls. Useful-mass training recompiles its new angles and records fresh host preparation and selection profiles. The timing experiment uses ten randomized rounds, 60 primary held-out cases, and the same 30 sample streams within every round. All powers and joules remain modeled.

## Records

All additions are under `results/v1_2_1/`; originals under `results/v1_2/` remain unchanged.

| Records | Meaning |
|---|---|
| `primary_instances.csv`, `primary_summary.csv` | Added zero-computation incumbent and paired greedy contrasts; no candidate metrics for retaining the incumbent |
| `decomposition.csv`, `marginal_instances.csv` | Reconciled component ledgers at allocated power and marginal fraction 0.001 |
| `component_grid.csv`, `nominal_summary.csv` | Exact power-only interpolation to 18.6912 kW; nominal-queue replay |
| `headroom.csv`, `sleep_consolidation.csv`, `postprocessing.csv` | Operating headroom, empty-server/consolidation counts, budget-dependent selection timing |
| `training_diagnosis.csv`, `training_summary.csv` | Uniform, individually fitted, and transferred useful mass on identical training cases; held-out transfer separately |
| `aligned_fits.json`, `aligned_transfer.json`, `aligned_resources.csv`, `aligned_samples.csv` | Optimum-informed, training-only useful-mass fits; new compiled resources and 28,800 held-out sampling batches |
| `demand_incumbents.csv`, `demand_outcomes.csv`, `demand_summary.csv` | 20–50% preceding-demand increments, feasibility halving, matched controller replay |
| `larger_fits.json`, `larger_quality.csv`, `larger_resources.csv` | Physical 14/16-variable placement quality and resources; distinct from historical dimensionless cases |
| `scaling.csv`, `scaling_summary.csv` | 60 sparse classical cases; construction/solve times, returned incumbents, bounds, solver status, gaps, modeled compute energy |
| `timing.csv`, `timing_paired.csv`, `timing_rounds.csv` | 2,400 case/controller/round records; samplers average 30 draws within each record |
| `*_manifest.json` | Executed stage settings, seed membership, platform, Python version, source hashes and elapsed time |
| `validation.json` | Independent small-case, event, pairing, training separation, and timeout checks |

Large returned assignment IDs are serialized as decimal strings to avoid loss of integer precision. A timeout is not an optimum. A missing incumbent and missing bound remain unknown. The scaling study models computation headroom only; it does not claim deadline-admissible operation at those sizes or quantum performance there.

## Validation

```sh
.venv/bin/python check_artifact.py
.venv/bin/python test_benchmark.py
.venv/bin/python test_energy.py
.venv/bin/python test_v12.py
.venv/bin/python validate_v12.py
.venv/bin/python test_v121.py
```

The first five checks retain the original benchmark audit. The added check independently verifies sparse MILPs against exhaustive optima, forces a solver timeout, replays action events, checks no-computation semantics and paired references, and checks training separation and component reconciliation. The public five-adapter `extension_api.py` remains compatible.

The primary claim is conditional: no original QAOA configuration wins in the original scenario grid. The new useful-mass fit improves useful probability without reversing its primary energy comparison. Larger demand changes and classical scaling expose distinct forms of headroom, and repeated host timings characterize variability without claiming a second-host replication.
