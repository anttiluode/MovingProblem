#!/usr/bin/env python3
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moving_problem_gate3 import summarize_gate3

summary = summarize_gate3(seeds=8)
print(json.dumps(summary, indent=2, sort_keys=True))
out = ROOT / "results" / "gate3_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {out}")
