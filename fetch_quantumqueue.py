#!/usr/bin/env python3
"""Fetch the QuantumQueue dataset at the pinned commit used by the paper."""
import subprocess, sys
from pathlib import Path

REPO = "https://github.com/rgokulsm/QuantumQueue"
COMMIT = "a614f916"
DEST = Path(__file__).resolve().parent / "data" / "QuantumQueue"

def main() -> int:
    if DEST.exists() and any(DEST.iterdir()):
        print(f"already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPO, str(DEST)], check=True)
    subprocess.run(["git", "-C", str(DEST), "checkout", COMMIT], check=True)
    print(f"checked out {COMMIT} into {DEST}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
