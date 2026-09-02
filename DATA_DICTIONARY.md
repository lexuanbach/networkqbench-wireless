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
| `evaluations`, `shots` | Charged objective/circuit evaluations and samples. |
| `success_prob` | Exact ideal statevector mass on feasible optima when available. |
| `profile` | `measured_local`, `optimistic`, `nominal`, or `stressed`. |
| `delay_s`, `deadline_s`, `deadline_miss` | Complete modeled/measured delay, decision budget, and miss indicator. |
| `utility` | Decomposable scenario utility defined in the manuscript. |

## Revision files

- `revision_hardness.csv`: 14/16-variable controlled-hardness extension.
- `revision_abilene_trace.csv`: decisions optimized at trace window `t` and
  scored at `t+1`; adds `next_max_utilization` and `next_overflow`.
- `revision_rate_intervals.csv`: Wilson 95% intervals for optimum-return rates.
- `revision_key_results.json`: equivalence interval, shot-confidence
  requirements, hardness summary, and trace summary.

The Abilene source files and checksums are under `data/abilene/`. They are
historical public traffic matrices, not current provider or wireless traces.
