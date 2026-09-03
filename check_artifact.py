#!/usr/bin/env python3
"""Leak gate + presence check for the standalone artifact."""
import sys
from pathlib import Path
root = Path(__file__).resolve().parent
bad = [p for p in root.rglob("*")
       if p.suffix in (".tex", ".pdf")
       and "output" not in p.relative_to(root).parts
       and ".git" not in p.relative_to(root).parts]
assert not bad, f"manuscript leak: {bad[:3]}"
required = ["networkqbench.py", "extension_api.py", "run_experiments.py",
            "analyze_results.py", "analyze_clear_accept.py",
            "run_review_experiments.py", "analyze_review_experiments.py",
            "compile_full_resources.py", "run_backend_emulation.py",
            "results/raw_results.csv", "results/key_results.json",
            "results/review_constraint_aware.csv",
            "results/review_closed_loop_5g.csv",
            "results/review_full_circuit_resources.csv",
            "results/review_backend_emulation.csv", "DATA_DICTIONARY.md",
            "LICENSE", "fetch_netdata.py"]
missing = [r for r in required if not (root / r).exists()]
assert not missing, f"missing: {missing}"
sys.path.insert(0, str(root))
import networkqbench  # noqa: F401  import smoke
import pandas as pd
assert len(pd.read_csv(root / "results/raw_results.csv")) > 1000
for script in root.glob("*.py"):
    if script.name == Path(__file__).name:
        continue
    source = script.read_text()
    assert ' / "submission" / ' not in source, f"external submission path: {script.name}"
    assert "02-QPU-Aware-CoOptimization" not in source, f"sibling-project path: {script.name}"
print("artifact check: OK")
