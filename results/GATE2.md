# Gate 2 — a downstream learner behind a representation that never stops moving

CI-verified 12-seed result.

## Question

Can a downstream computation remain useful when the upstream network itself stays plastic and its individual coordinates continuously drift?

This is not the Gate 0 button-rotation world.

The drift is produced by another learning system.

## Provenance

Gate 2 is inspired directly by Qin et al., **Coordinated drift of receptive fields during noisy representation learning** (2021):

https://doi.org/10.1101/2021.08.30.458264

Their linear similarity-matching network has the fixed-point neural map

```text
y = M^-1 W x
```

corresponding to dynamics

```text
dy/dt = W x - M y
```

and local feedforward/lateral updates of the form

```text
Delta W = eta (y x^T - W)
Delta M = eta (y y^T - M)
```

with synaptic noise during continued learning.

The paper's key observation is that individual representations can drift through a degenerate family of equally good solutions while population representational similarity remains stable.

Gate 2 uses that architecture, but is **not an exact replication**.

## Deliberate modification

The paper's example principal eigenvalues are not used here.

Gate 2 gives the top three latent directions:

```text
equal zero-lag variance
different temporal autocorrelation

rho = [0.95, 0.65, 0.25]
```

and separates them from low-variance nuisance dimensions.

Why?

Because equal zero-lag variance makes the useful 3-D subspace rotationally degenerate from the point of view of instantaneous covariance. PCA can identify the subspace, but there is no zero-lag reason to call one direction "source 1".

Temporal statistics do contain that information.

This makes the experiment an attack on the exact distinction MovingProblem cares about:

> **stable geometry is not the same thing as stable freedom identity.**

## Upstream learner

The 8-D input is compressed to 3 outputs by the Hebbian/anti-Hebbian similarity-matching network.

There is a burn-in phase with no explicit synaptic noise.

Then the network remains plastic for 50,000 more samples while Gaussian synaptic update noise is applied.

Nothing tells the representation to rotate.

The drift emerges from ongoing local updates inside a degenerate useful subspace.

## Did the representation actually drift?

Yes.

Across 12 seeds, comparing the same 200 fixed probe stimuli at the start and end:

```text
raw coordinate change        0.7280 ± 0.2917
Gram/similarity change       0.0266 ± 0.0105
rotation drift metric        1.2615 ± 0.5081
mean PSP error               0.1332 ± 0.0049
```

So individual coordinates move substantially while pairwise population geometry remains far more stable.

That reproduces the qualitative phenomenon Gate 2 needed from the paper.

## Two downstream tasks

### Task A — geometry is enough

The task depends only on whether consecutive latent states are more or less similar than an initial threshold.

A decoder can use the current output inner product

```text
y(t) dot y(t-1)
```

without naming any individual axis.

Result:

```text
geometry-only similarity task   0.9909 ± 0.0005
```

There is no reason to run AMUSE here.

The invariant is already the computation.

### Task B — identity/orientation matters

Now the downstream task is:

```text
is latent source z0 positive or negative?
```

The three useful latent sources have equal instantaneous variance, so zero-lag geometry cannot tell which current axis is z0.

Methods:

1. **Frozen coordinate decoder**  
   Fit once on the first 1,000 samples and never update.

2. **Zero-lag PCA tracker**  
   Re-estimate a PCA basis every 1,000 samples, resolve the initial sign with 16 labels, then use continuity.

3. **Rotation-invariant feature readout**  
   Fit on norms and lagged inner products only.

4. **AMUSE tracker, calibrate once**  
   Every 1,000 samples, use lag covariance to recover axes ordered by temporal signature. Resolve z0's sign once with 16 labels in the first window. Thereafter choose signs only by continuity with the previous demixer.

5. **AMUSE, recalibrate every window**  
   Same temporal separator, but spend another 16 labels in every window.

## Result

| method | oriented-task accuracy |
|---|---:|
| **AMUSE tracker, 16 labels once** | **0.9323 ± 0.0068** |
| AMUSE recalibrated every window | 0.9243 ± 0.0151 |
| frozen coordinate decoder | 0.8442 ± 0.0466 |
| zero-lag PCA tracker | 0.5020 ± 0.0521 |
| rotation-invariant features | 0.4985 ± 0.0131 |

Late drift makes the frozen failure clearer:

```text
last-quarter frozen decoder    0.7720 ± 0.0965
last-quarter AMUSE tracker     0.9323 ± 0.0167
```

Calibration cost:

```text
track continuously after one calibration    16 labels
recalibrate every 1,000-sample window       800 labels
```

The once-calibrated tracker is not merely cheaper in this world; it is slightly more accurate than repeatedly spending labels, because continuity avoids noisy repeated sign decisions.

## What Gate 2 earns

### 1. Do not align what the task does not care about

When the downstream task depends only on preserved relational geometry:

```text
99.1%
```

with no coordinate recovery.

Trying to identify axes would be wasted work.

### 2. Geometry cannot name an oriented freedom

On z0 sign:

```text
PCA tracker          50.2%
invariant features   49.9%
```

The useful subspace is present, but source identity is not available from zero-lag geometry.

### 3. Temporal diversity can break the continuous ambiguity

Different AR signatures identify the three source axes.

AMUSE raises the oriented task to:

```text
93.2%
```

### 4. Smooth drift turns repeated calibration into tracking

The remaining sign ambiguity is paid for once.

After that the current demixer stays oriented by continuity with the previous one:

```text
16 labels once
versus
800 labels when recalibrating every window
```

This is the strongest useful result in MovingProblem so far.

## The architecture that survived

```text
UPSTREAM NETWORK
continues local Hebbian/anti-Hebbian learning
             |
             v
representation coordinates drift
but relational geometry remains useful
             |
             +-----------------------------+
             |                             |
             v                             v
task only needs geometry          task needs named freedom
use invariant directly            use temporal statistics
no recalibration                  to recover axis identity
                                             |
                                             v
                                  initial scalar consequence
                                  fixes sign once
                                             |
                                             v
                                  continuity maintains lock
```

## What this does NOT establish

- the brain uses AMUSE;
- Qin et al.'s biological model contains this downstream solution;
- all representational drift is rotational;
- real neural/sensor drift has stable temporal signatures;
- this beats adaptive BSS;
- this beats Euclidean/Riemannian alignment on tasks where geometry alone suffices;
- a new source-separation algorithm;
- general deep credit assignment without backprop.

The result instead establishes a clean **division of labor**:

> **Use invariants when the task is invariant. Spend temporal/statistical identification and consequence only on the residual degrees of freedom that action actually requires.**

## Next attacks

The next experiments should attack this result, not decorate it:

1. add Euclidean alignment / Procrustes on the geometry-sufficient task;
2. add adaptive BSS on the oriented task;
3. make temporal signatures crowd and cross, so identity can become ambiguous;
4. let one source disappear and another appear;
5. make the useful subspace itself slowly deform rather than merely rotate;
6. measure when continuity loses lock and how much consequence is needed to reacquire it.

Raw summary: [gate2_summary.json](gate2_summary.json).
