#!/usr/bin/env python3
"""Gate 0: preserve a learned nonlinear function while the observation basis moves."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moving_problem import summarize_gate0


def main() -> None:
    summary = summarize_gate0(seeds=20)
    print(json.dumps(summary, indent=2, sort_keys=True))
    out = Path(__file__).resolve().parents[1] / "results" / "gate0_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
