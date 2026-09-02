# NetworkQBench-Wireless — standalone artifact

Simulator-first benchmark for quantum-assisted wireless network control.
Quantum evidence is exact statevector simulation or archived-calibration
noisy emulation. Every remote-QPU timing value is a labeled model or a labeled
public-telemetry replay, never a live-hardware measurement. No .tex/.pdf ships
here (manuscript lives outside this folder); `check_artifact.py` enforces that.

## Experiment map (script -> results)
| Stage | Run | Analyze | Key outputs |
|---|---|---|---|
| Main grid (360 inst.) | run_experiments.py | analyze_results.py | raw_results.csv, key_results.json, solver_summary table |
| Parameter transfer | run_recovery_experiments.py | analyze_results.py | recovery_*.csv |
| Revision wave (Abilene, hardness, mean-field) | run_revision_experiments.py | analyze_revision_experiments.py | revision_*.csv |
| Clear-accept wave (MIP/HiGHS, NetData 5G, shots, compiled resources) | run_clear_accept_experiments.py, compile_resources.py, import_provider_evidence.py | analyze_clear_accept.py | clearaccept_*.csv/json |
| Depth/transfer/penalty/timing + QuantumQueue replay | run_fable_experiments.py | analyze_clear_accept.py | fable_*.csv/json |
| Constraint-aware + closed-loop review wave | run_review_experiments.py | analyze_review_experiments.py | review_constraint_aware*.csv, review_closed_loop*.csv/json |
| Complete circuits + archived-calibration emulation | compile_full_resources.py, run_backend_emulation.py | analyze_review_experiments.py | review_full_circuit*.csv/json, review_backend_emulation*.csv/json |

## Headline numbers (regenerable; see REPRODUCE.md)
- Tuned depth-1 QAOA optimum rate 99.2% (best-of-1,024) vs local search 99.4%; mean optimal-state probability 3.2%.
- Paired QAOA-minus-local quality loss -0.0148, 95% CI [-0.0451, 0.0018] (inside the pre-declared ±0.05 equivalence margin).
- Mean 105.5 objective evaluations per tuned instance -> 100% modeled deadline misses in every profile; 1,024-shot transferred parameters: 90.0% optimal, 28.3% on time (nominal).
- Abilene next-window replay: local/mean-field 96.7% optimal vs QAOA 90.0%; n=16 hardness: 86.7% vs 56.7%.

Environment: Python >=3.13, numpy, scipy, pandas, scikit-learn (HiGHS via scipy).

## Public extension interface

`extension_api.py` defines the stable five-adapter contract for instances,
solvers, service evidence, next-state outcomes, and provenance. Extensions
return the typed records in that module and call `apply_deadline_policy`.
This preserves the rule that a late computed action is recorded while the
previous action is applied. The interface rejects negative phase times and
requires an explicit evidence label and provenance fields.
