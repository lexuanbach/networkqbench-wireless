# NetworkQBench-Wireless — reproducible benchmark artifact

This repository accompanies **NetworkQBench-Wireless: Deadline-Aware Energy Accounting for Quantum-Assisted Network Control**, an original manuscript. Software release `networkqbench-wireless-v1.2.0` integrates solution quality, complete execution resources, applied actions, and operational energy. See [README-energy.md](README-energy.md) for the complete schema-1.2 experiment, validation, and reproduction commands.

The supporting network studies below retain their execution protocols and result files. Quantum evidence is exact noiseless/noisy simulation or declared calibration-snapshot emulation. Remote service and electricity are explicit models or trace replay, not live-QPU measurements. Internal reviews and manuscript source are excluded from this artifact. Generated PDFs and table fragments are written only under `output/`.

## Experiment map (script to results)

All scripts resolve inputs and outputs relative to this repository. Raw and
summary records remain under `results/`. Analyzers write regenerated figures
and manuscript-ready table fragments under ignored `output/` directories.

| Stage | Run | Analyze | Key outputs |
|---|---|---|---|
| Main grid (360 inst.) | run_experiments.py | analyze_results.py | raw_results.csv, key_results.json, solver_summary table |
| Parameter transfer | run_recovery_experiments.py | analyze_results.py | recovery_*.csv |
| Abilene, hardness, and mean-field | run_revision_experiments.py | analyze_revision_experiments.py | revision_*.csv |
| HiGHS, NetData 5G, shots, compiled resources | fetch_netdata.py, run_clear_accept_experiments.py, compile_resources.py | analyze_clear_accept.py | clearaccept_*.csv/json |
| Depth/transfer/penalty/timing + QuantumQueue replay | run_fable_experiments.py | analyze_clear_accept.py | fable_*.csv/json |
| Constraint-aware and closed-loop experiments | run_review_experiments.py | analyze_review_experiments.py | review_constraint_aware*.csv, review_closed_loop*.csv/json |
| Complete circuits + archived-calibration emulation | compile_full_resources.py, run_backend_emulation.py | analyze_review_experiments.py | review_full_circuit*.csv/json, review_backend_emulation*.csv/json |

## Headline numbers (regenerable; see REPRODUCE.md)
- Tuned depth-1 QAOA optimum rate 99.2% (best-of-1,024) vs local search 99.4%; mean optimal-state probability 3.2%.
- Paired QAOA-minus-local quality loss -0.0148, 95% CI [-0.0448, 0.0018] (inside the declared ±0.05 equivalence margin).
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

`EvidenceKind` is the single vocabulary used by extension records, CSV
evidence columns, and manifests. The data dictionary maps each evidence
column to the relevant values.

## External data and license

The benchmark code is released under the MIT license in `LICENSE`.
Third-party datasets retain their original terms. The NetData source does not
declare a redistribution license, so its 5G CSV is not shipped. Run
`python3 fetch_netdata.py` to obtain the pinned input and verify its checksum.
Clone QuantumQueue into `data/QuantumQueue` or pass its location through
`run_fable_experiments.py --queue-dir PATH`.
