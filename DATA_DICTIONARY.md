# Schema 1.2 additions

The integrated physical study is documented in `README-energy.md`. `energy_ledger.csv` identifies component, joules, accounting boundary, evidence, source, scenario, and amortization. Admission events include workload/size/seed, outcome, incumbent, encoding, depth, noise, controller, transition and event identities. `computed_action` and `applied_action` remain separate. Missing historical fields remain unknown; no backfilled zero values are used.

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

The record schema carries `SCHEMA_VERSION` (currently `1.1`; legacy records remain unchanged), exported by
`extension_api.py`. Any field addition, removal, or unit change increments
the version; result CSVs produced by future schema versions must state the
version in their run manifests so downstream analyzers can refuse
mismatched inputs. Third-party adapters are validated by the typed
protocols and invariant checks in `extension_api.py`.

## Sustainability extension: schema 1.1

The current `SCHEMA_VERSION` is 1.1. Historical result manifests retain their
original version and meanings. New energy outputs live only in
`results/sustainability/`; see `README-energy.md` for the file map.

Additional canonical evidence values are `modeled_energy`, `modeled_network`,
and `modeled_operation_durations`. An external measured server curve is used as
an input to a **modeled network**, not labeled as a measurement of that network.
No new output has `measured_provider` electricity evidence.

All `_s` fields use seconds, `_w` fields watts, and `_j` fields joules. `size`
means assignment binary variables; `seed` identifies the independent instance,
`repeat` the within-instance sampling trial, and `shots` the per-circuit budget.
`p_quality` is exact simulated probability of a feasible state within 5% of the
power optimum, normalized by optimum + 1 W. `quality_rate` is the empirical
fraction of sampled batches returning such a state. `useful_rate` also requires
on-time delivery. `optimal_rate` uses a 1e-8 W audit tolerance.

`compute_preparation_j`, `compute_optimization_j`, `compute_communication_j`,
`compute_postprocessing_j`, `compute_offline_j`, and `compute_service_j` are
disjoint components whose sum is `compute_j`. `total_j` additionally includes
`network_j` and `transition_j`. The `_vs_milp_j` saving is positive only when the
candidate's total is lower than MILP. The manuscript's delta E uses the opposite
sign. `boundary` is `allocated` or `marginal`; the latter changes the service
power fraction, not the host/server definitions. All numeric energy fields in
these results are modeled, as declared in their scenario manifest.

CSV missing values for unknown historical energy and absent lifetime/reuse
crossings are not zeros. Infinite energy-per-useful values mean that no useful
outcome was obtained; JSON headline summaries serialize these ratios as null,
with the zero useful rate retained. Scenario rows share outcomes and must not
be pooled as independent statistical observations.
