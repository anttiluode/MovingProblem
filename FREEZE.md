# Frozen at Gate 4

**MovingProblem is intentionally frozen here. No Gate 5 is planned in this repository.**

The gate ladder did its job: it narrowed a large "non-backprop / drifting representation" exploration to one small failure mode that is worth carrying forward independently.

## The result worth preserving

A continuously matched real frame can return with the wrong **oriented sign** after a closed parameter path even though:

- the eigengap remained large the entire time;
- the local frame was statistically identifiable everywhere;
- every gap/confidence health check remained green;
- the endpoint operator returned to its starting value.

In Gate 4's 2×2 laboratory:

```text
minimum gap                     2.0000
low-confidence fraction        0.0000
final oriented alignment      -1.0000
```

A non-enclosing loop returns +1. Two enclosing loops return +1. Small symmetric operator noise leaves the one-loop result at approximately -0.9999.

The local health check therefore misses a **path/topology failure**.

For the toy 2-D case, one bit of winding parity repairs the externally grounded orientation.

## What this is NOT

- not new fiber-bundle mathematics;
- not a new Berry-phase result;
- not a better source separator;
- not a replacement for Procrustes;
- not evidence that brains implement holonomy trackers;
- not evidence that this failure occurs often in real sensor/neural data.

The topology is old. The open practical question is whether adaptive frame/BSS systems encounter such loops in real drifting streams often enough to matter.

## Why stop here

Several broader claims died or became ordinary prior art:

- dimensionality reduction explains the Gate 1 compression;
- Stiefel / Grassmann drift is established mathematics and current neuroscience prior art;
- Euclidean / Procrustes alignment wins when repeated paired anchors are available;
- exact degeneracy remains genuinely unidentifiable;
- local gap monitoring is useful but incomplete.

Adding another synthetic gate would make the repository less clear.

## The next empirical question belongs elsewhere

> **On a real long multichannel stream, does the tracked identifying frame accumulate nonzero winding / sign holonomy around degeneracies?**

If yes, the bug is live.

If no, Gate 4 is an elegant synthetic failure mode and should remain exactly that.

That real-data question is intentionally left for the next repository.

## Demo

Open [holonomy_demo.html](holonomy_demo.html).

It shows the entire claim visually:

```text
parameter path closes
health light stays green
semantic sign flips
```

with a null control and the one-bit winding-parity repair.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments/gate4_topological_holonomy.py
```

The full history remains in the README and individual gate receipts. This file is the stop marker.
