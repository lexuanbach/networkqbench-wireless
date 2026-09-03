# Reproduce every number

```bash
python3 test_benchmark.py                      # smoke gate
python3 run_experiments.py --seeds 30          # main grid (long)
python3 run_recovery_experiments.py
python3 run_revision_experiments.py
python3 fetch_netdata.py                       # pinned external 5G trace
python3 run_clear_accept_experiments.py
python3 compile_resources.py                   # logical-resource ledger
# Clone https://github.com/rgokulsm/QuantumQueue into data/QuantumQueue,
# or provide another local clone with --queue-dir.
python3 run_fable_experiments.py --queue-dir data/QuantumQueue
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
deterministic given the recorded seeds (see manifests in `results/`).
Analyzers write regenerated figures and table fragments to ignored `output/`
directories.

`import_provider_evidence.py` is not part of the QuantumQueue replay. It
validates a future provider-job CSV with the required schema. For example:

```bash
python3 import_provider_evidence.py provider_jobs.csv \
  --manifest results/provider_jobs_manifest.json
```

```bash
python3 fetch_quantumqueue.py   # optional: QuantumQueue at the pinned commit
python3 emit_host_manifest.py  # records the timing host
python3 analyze_tnsm_revision.py  # SLA weight grid, seed-cluster SE, block sensitivity
```
