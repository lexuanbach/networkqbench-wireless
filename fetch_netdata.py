#!/usr/bin/env python3
"""Fetch the pinned NetData input without redistributing the upstream CSV."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "data" / "netdata" / "Performance_5G_Weekday.csv"
COMMIT = "093e13ebcbee4f18a3fc4d6a6f3aeab9ac1283a3"
URL = ("https://media.githubusercontent.com/media/tsinghua-fib-lab/"
       f"NetData/{COMMIT}/Performance_5G_Weekday.csv")
SHA256 = "33a4a1530d6a871807ffae0a498a803c5631fd5edc33138369fdf88bf3a9f0a0"


def main() -> int:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    payload = urlopen(URL, timeout=120).read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"NetData checksum mismatch: {digest}")
    DESTINATION.write_bytes(payload)
    print(f"Wrote {len(payload)} bytes to {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
