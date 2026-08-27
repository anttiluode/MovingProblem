#!/usr/bin/env python3
import json
from pathlib import Path
from moving_problem_gate1 import summarize_gate1

summary = summarize_gate1(seeds=12)
print(json.dumps(summary, indent=2, sort_keys=True))
out = Path(__file__).resolve().parents[1] / "results" / "gate1_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {out}")
