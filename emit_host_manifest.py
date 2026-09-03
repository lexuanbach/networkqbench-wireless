#!/usr/bin/env python3
"""Emit a machine-readable manifest of the host used for measured timings."""
import json, platform, subprocess, sys
from importlib import metadata
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "host_manifest.json"

def main() -> int:
    pkgs = {}
    for name in ("numpy", "scipy", "pandas", "networkx", "qiskit", "highspy"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    cpu = ""
    if platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "platform": platform.platform(),
        "processor": cpu or platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": pkgs,
    }, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
