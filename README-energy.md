# NetworkQBench-Wireless: operational-energy experiments (schema 1.2)

This is the artifact for **NetworkQBench-Wireless: Deadline-Aware Energy Accounting for Quantum-Assisted Server Placement in Network Control**, one original, unpublished manuscript. Version 1.2.0 is a software release; it does not imply a previously published paper.

The benchmark integrates channel assignment, routing, dimensionless placement, and physically dimensioned server placement. Only physical placement supports electricity conclusions. Quantum and server energy are modeled. Local phase times are measured; quantum quality is exact noiseless or exact noisy simulation. No live-QPU or network electricity measurement is claimed.

For the current article and version 1.2.2 package, use [README-round3.md](README-round3.md). The previous revision protocol remains in README-revision.md. The commands below reproduce the original 1.2 outputs.

## Reproduce the original outputs from frozen measurements

Python 3.14.6 was used. Install the pinned dependencies in an isolated environment. A C++17 compiler is needed for the exact-density validation and full 12-qubit noisy experiment. No external plotting package, font folder, or manuscript checkout is required.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-energy.txt
.venv/bin/python check_artifact.py
.venv/bin/python test_benchmark.py
.venv/bin/python test_energy.py
.venv/bin/python test_v12.py
.venv/bin/python validate_v12.py
.venv/bin/python analyze_v12.py
.venv/bin/python diagnose_v12.py
.venv/bin/python render_v12.py
```

Run the analyzers sequentially in that order. Outputs are written under `output/v1_2/`. The analysis uses stored phase times and stored `policy_timings.json`, including calibration and lookup costs; it does not silently remeasure them. `--skip-sweep` on `analyze_v12.py` skips only the unchanged 8,640-cell base sweep. Do not use it after changing accounting semantics. Figures contain no external absolute-path dependencies. PDF metadata can vary by generation time; numerical records and table text are the reproduction targets.

To copy regenerated publication outputs into an explicitly chosen manuscript directory:

```sh
.venv/bin/python render_v12.py --manuscript /path/to/manuscript
```

## Re-run experiments, with new timing measurements

Work in a copy of the release. `run_v12.py` regenerates the 120 base cases under four power variants, training-only fits, complete circuits, and 432,000 sampling batches. `run_v12_noise.py` produces 240 exact noisy distributions and 28,800 sampling batches. It caches individual distributions. Delete the relevant cached results only in the working copy when changing the noise model or circuit source.

```sh
.venv/bin/python run_v12.py
.venv/bin/python run_v12_noise.py
```

The noisy experiment takes substantially longer than frozen-output regeneration. Aer 0.17.2 evaluates 6/8/9-qubit exact density matrices. A bundled double-precision native kernel evaluates the same channels at 12 qubits, with zero-noise statevectors used as an exact reference. The kernel is built in the system temporary directory and uses full density matrices, not trajectories. Tests compare it against Aer and an analytic depolarizing-channel result. The manifest records the actual engine per condition. At most 12 qubits are simulated; enlarged QUBO circuits receive only a resource audit.

New host timings will change modeled joules. To collect fresh policy timings, remove each working copy's `policy_timings.json` before rerunning analysis. Preserve the released records and checksums as the reference run.

## Experimental contract

- Base identities: four sizes (6, 8, 9, 12), 30 seeds each; training 0–14, testing 15–29. The 60 held-out problem instances are the statistical units.
- Four power variants: all servers powered, plus hypothetical empty-server sleep draws of 0/10/50 W. Shared variants do not create independent instances.
- All 480 variant-specific MILP optima match independent exhaustive physical evaluation. Every QUBO audit verifies optimum equivalence.
- Budgets 16/64/256/1,024; 30 common-random-number repetitions. Classical and quantum uniform controls share ideal samples and selection semantics. Dense and structured one-hot depth-one circuits agree on the one-hot subspace.
- Primary incumbent: exact preceding placement under seeded increased demands, feasible after demands fall. Concentrated incumbents are separate stress cases.
- All controllers use the same event-based benefit gate, transition costs, deadline, and strict action-lifetime boundary. Concurrent backup charges both computations and every accepted transition.
- System-envelope power is a scenario grounded in an approximately 50-qubit reference architecture, not a measurement of the grid compilation target. Allocated and marginal accounting remain distinct. Queue waiting does not charge full QPU power.
- Offline training, calibration acquisition, and lookup are charged. Simulator research execution is identified separately from QPU service energy.
- Missing historical energy remains unknown. `EnergyLedger` rejects duplicate components; schema 1.2 adds workload/incumbent/encoding/depth/noise/controller/transition/event identities.

## Results and provenance map

| Records | Meaning |
|---|---|
| `results/v1_2/manifest.json` | Experiment environment, units, seed membership, source hashes |
| `*/instances.csv`, `classical.csv` | Physical states, preceding demand, co-optima, measured baseline timings |
| `*/fits.json`, `transfer.json` | Training membership, angles, dispersion, measured fitting effort |
| `*/circuits.csv`, `qubo_audit.csv` | Matched target resources, scheduled duration, exact encoding checks |
| `*/samples.csv`, `distributions.npz` | Repeated outcomes and exact logical distributions |
| `*/outcomes.csv`, `primary_instances.csv` | Within-instance outcome weights and component energy |
| `energy_ledger.csv` | Component joules, boundary, evidence, source, scenario, amortization |
| `*/primary_summary.csv`, `per_size.csv` | Paired 95% intervals, quality, timeliness, useful rates, W/T/L |
| `*/sensitivity.csv` | 8,640 all-powered base cells (including all three marginal fractions) |
| `*/policies.csv`, `admission_events.csv` | Training-only selections, predictions, computed/applied actions |
| `*/policy_per_size.csv`, `policy_summary.csv` | Controller comparisons and paired intervals |
| `*/sleep_sensitivity.csv`, `overheads.csv`, `migration_sensitivity.csv` | Separate physical and occupied-service sensitivities |
| `noise/` | 240 exact noisy distributions, resources, engine, and sampling records |
| `noise_instances.csv`, `noise_summary.csv` | Matched all-to-all energy and quality diagnostics |
| `service_compression.csv`, `*/thresholds.csv` | Piecewise service sensitivity and conditional power/reuse crossings |
| `utilizations.csv`, `illustrative_carbon.csv` | Placement distributions and declared carbon-intensity conversions |
| `stress_decomposition.csv` | Reproduction of the approximately 2,992-J concentrated-incumbent contrast |
| `validation.json` | Independent physical, sampling, event, and noise record audit |

`results/sustainability/` retains separately measured profiles used by the stress diagnostic. Its filenames describe software history, not a separate publication. The supporting network studies retain their original manifests and units. Do not join representative compilation rows to arbitrary physical cases without instance keys.

## Conclusions and limits

In the primary scenario, structured one-hot transfer improves resources over dense synthesis but still exceeds HiGHS total energy. Across the base grid no tested QAOA configuration has a negative mean contrast. A small marginal exception exists for low-quality zero-layer full-space sampling; its matched classical sampler is cheaper. This is a conditional finding for small synthetic instances, not a universal quantum-energy claim. Refer to generated `headline_numbers.json` and summary tables for exact values.

Full environmental lifecycle, radio power, transport power, measured migration protocols, and calibrated hardware noise remain outside the quantified boundary. Carbon conversion uses illustrative 50/400/800 gCO₂e/kWh. Do not apply PUE to cooling already included in a system envelope.

## Compatibility

The earlier dimensionless analyses remain available through the scripts listed in `README.md` and `REPRODUCE.md`. The separately recorded energy profiles regenerate with `analyze_energy.py`; `export_energy_manuscript.py --version 1.1` explicitly exports those profiles. The default export is 1.2. Supporting records are preserved, never replaced with invented zero-energy values.
