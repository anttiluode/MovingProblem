#!/usr/bin/env python3
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moving_problem_gate2 import summarize_gate2

summary = summarize_gate2(seeds=12)
print(json.dumps(summary, indent=2, sort_keys=True))
out = ROOT / "results" / "gate2_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {out}")
