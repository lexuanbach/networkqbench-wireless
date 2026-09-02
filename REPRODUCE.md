# Reproduce every number

```bash
python3 test_benchmark.py                      # smoke gate
python3 run_experiments.py --seeds 30          # main grid (long)
python3 run_recovery_experiments.py
python3 run_revision_experiments.py
python3 run_clear_accept_experiments.py
python3 compile_resources.py                   # logical-resource ledger
python3 import_provider_evidence.py --csv data/QuantumQueue   # public telemetry
python3 run_fable_experiments.py
python3 run_review_experiments.py
python3 compile_full_resources.py
python3 run_backend_emulation.py
python3 analyze_results.py
python3 analyze_revision_experiments.py
python3 analyze_clear_accept.py                # emits all summary tables/JSONs
python3 analyze_review_experiments.py          # review-wave tables/JSON
```

Stored `results/` lets you rerun only the analyzers and obtain every
reported number without re-running the experiment stages. Outputs are
deterministic given the recorded seeds (see manifests in results/).
