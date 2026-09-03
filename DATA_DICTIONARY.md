# NetworkQBench-Wireless data dictionary

All result files are comma-separated and keyed by `task`, `size_label` or
`n_vars`, `seed`, and `algorithm`. QAOA rows additionally use `profile` to
distinguish modeled deployment timing.

## Core result fields

| Field | Meaning |
|---|---|
| `task` | `channel`, `placement`, `routing`, or trace replay task. |
| `seed` | Deterministic instance or trace-window identifier. |
| `algorithm` | Exact, random, local search, annealing, mean field, or QAOA. |
| `cost`, `optimum` | Returned penalized objective and enumerated feasible optimum. |
| `feasible` | Whether the returned bit string satisfies the unpenalized constraints. |
| `gap` | Nonnegative normalized objective gap. |
| `runtime_s` | Measured local solver/statevector runtime; never a QPU measurement. |
| `evals`, `shots` | Charged objective or circuit evaluations and samples. |
| `success_prob` | Exact ideal statevector mass on feasible optima when available. |
| `profile` | `local`, `optimistic`, `nominal`, or `stressed`. This field selects a timing profile rather than declaring evidence. |
| `delay_s`, `deadline_s`, `deadline_miss` | Complete modeled/measured delay, decision budget, and miss indicator. |
| `utility` | Decomposable scenario utility defined in the manuscript. |

## Evidence labels

`extension_api.EvidenceKind` defines the canonical vocabulary:
`measured_local`, `exact_simulated`, `trace_derived`, `compiled_estimate`,
`calibration_snapshot_emulated`, `modeled_qpu`, `trace_replayed_qpu`, and
`measured_provider`. Evidence is attached to the quantity it qualifies:

- `timing_evidence` uses `measured_local` or `modeled_qpu`.
- `delay_evidence` uses `measured_local` or `trace_replayed_qpu`.
- `evidence` in `review_backend_emulation.csv` uses
  `calibration_snapshot_emulated`.
- Experiment manifests use `exact_simulated`, `trace_derived`, and
  `compiled_estimate` for output and input layers that do not share one row.

This separation prevents a trace-derived workload from being mistaken for a
measured QPU execution merely because both appear in the same experiment.

## Revision files

- `revision_hardness.csv`: 14/16-variable controlled-hardness extension.
- `revision_abilene_trace.csv`: decisions optimized at trace window `t` and
  scored at `t+1`; adds `next_max_utilization` and `next_overflow`.
- `revision_rate_intervals.csv`: Wilson 95% intervals for optimum-return rates.
- `revision_key_results.json`: equivalence interval, shot-confidence
  requirements, hardness summary, and trace summary.

The Abilene source files and checksums are under `data/abilene/`. They are
historical public traffic matrices, not current provider or wireless traces.

## Schema versioning

The record schema carries `SCHEMA_VERSION` (currently `1.0`), exported by
`extension_api.py`. Any field addition, removal, or unit change increments
the version; result CSVs produced by future schema versions must state the
version in their run manifests so downstream analyzers can refuse
mismatched inputs. Third-party adapters are validated by the typed
protocols and invariant checks in `extension_api.py`.
