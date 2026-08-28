# Gate 3 — identifiability stress map

CI-verified, 8 seeds.

## Why this gate

Gate 2 produced a clean result:

- geometry-only tasks can ignore coordinate drift;
- axis-sensitive tasks can use temporal diversity to recover identity;
- a small amount of consequence fixes the remaining orientation once;
- continuity can maintain lock.

Gate 3 asks what happens when the equivalence manifold stops being friendly.

The stress harness is controlled rather than biological. Gate 2 already established that the paper-inspired Hebbian/anti-Hebbian learner can generate genuine representational drift. Gate 3 isolates the downstream identification problem so we can force specific failures.

## Scenarios

Every world contains three independent Gaussian temporal sources. The observed coordinates move smoothly between 1,000-sample windows.

Attacks:

1. **separated signatures**
   `rho = [0.95, 0.65, 0.25]`

2. **crowded signatures**
   `rho = [0.95, 0.91, 0.87]`

3. **crossing signatures**
   source 0 moves `0.95 -> 0.55` while source 1 moves `0.55 -> 0.95`

4. **sustained exact degeneracy**
   `rho = [0.95, 0.95, 0.25]`

5. **source disappearance**
   source 0 fades to zero, remains absent for 12 windows, then returns

6. **non-orthogonal mixing**
   the observation map rotates while independently stretching/shrinking the source axes

7. **subspace deformation**
   the useful 3-D source subspace itself bends through a 5-D ambient observation space

## The tiny new rule

Gate 2 trusted lag-eigenvalue order.

That fails when temporal signatures cross: the eigenvalues exchange order, so a downstream semantic identity silently swaps from one source to another.

Gate 3 keeps a trusted frame and aligns each new AMUSE frame using both:

- row / coordinate continuity;
- temporal-signature continuity.

Then it asks whether the current operator is temporarily less identifiable than its recent history.

If either the signature gap collapses abruptly or the tracked row loses geometric continuity:

```text
DO NOT UPDATE SEMANTIC IDENTITY
```

For that window the tracker coasts with the last trusted map.

Once identifiability returns, it re-locks to the moving frame.

This is not a new source separator. It is a conservative identity-tracking policy.

## Results

### Friendly separated world

```text
guarded temporal tracker        0.9330 ± 0.0128
EA-style covariance recentering 0.6741 ± 0.1333
```

The EA-style baseline symmetrically whitens every window from zero-lag covariance, then applies one frozen readout. It helps relative to a totally frozen raw coordinate system, but cannot determine the individual temporal freedom in this isotropic useful subspace.

This is **not** a full Riemannian Procrustes implementation.

### Crossing signatures

```text
trust eigenvalue order          0.7226 ± 0.0065
guarded continuity              0.9507 ± 0.0052
guarded, final 10 windows       0.9649 ± 0.0053
recalibrate from 16 labels
    every window                0.9360 ± 0.0115
```

The guarded tracker marks about

```text
25.9% ± 3.3%
```

of crossing windows as low confidence and refuses to let those windows rewrite semantic identity.

Calibration cost:

```text
guarded tracker                 16 labels once
recalibrate every window       640 labels
```

The key result is not merely better classification:

> **a temporary collapse of statistical identifiability does not require forgetting a previously grounded identity.**

If the ambiguous interval is short relative to coordinate drift, old identity can bridge it.

### Explicit paired correspondences

Ordinary orthogonal Procrustes is given 16 repeated paired anchor stimuli in the reference and current coordinate system in **every** window:

```text
paired Procrustes               0.9935 ± 0.0005
```

It crushes the temporal tracker.

Good.

If stable paired correspondences are available, use them.

The information cost is different:

```text
temporal tracker       16 task labels once
Procrustes             16 paired anchors every window
```

This is a boundary, not a victory claim.

### Sustained exact degeneracy

```text
guarded tracker overall         0.4731 ± 0.1895
last 10 windows                 0.4592 ± 0.1882
```

Failure.

With two statistically identical Gaussian sources, no temporal statistic distinguishes their individual axes. Continuity may coast by accident for a while, but there is no observation-driven correction within the degenerate plane.

This is the expected identifiability wall.

### Source disappearance and return

During the interval in which the target source has exactly zero amplitude:

```text
0.4990 ± 0.0068
```

Chance. Correct.

After the same source returns:

```text
0.9332 ± 0.0151
```

without spending a new task-label budget.

So absence is not confused with evidence about a new semantic identity.

### Non-orthogonal mixing

The observation map both rotates and changes axis scales:

```text
0.9328 ± 0.0127
```

The clean orthogonal-Q assumption from Gate 0 was therefore not load-bearing in this invertible linear stress test.

### Useful subspace itself deforms

The three-dimensional source subspace slowly rotates into an additional ambient dimension in a 5-D observation space:

```text
0.9321 ± 0.0122
```

So the result is not confined to pure internal rotations inside one fixed observed subspace.

## What survived

The current picture is now:

```text
if task needs only invariant geometry:
    use the invariant directly

if task needs a named freedom
and statistics identify it:
    track it

if identifiability temporarily collapses:
    preserve the last grounded identity
    do not learn through the ambiguity

if identifiability is absent indefinitely:
    admit failure / request new information

if paired correspondences are available:
    ordinary Procrustes is better
```

## The mathematical object that is emerging

A useful subspace can move while a basis within that subspace also rotates.

Those are different motions.

The downstream problem is not merely "estimate a matrix." It is:

> **transport task meaning along a moving family of representational frames, updating only when the observations actually identify the frame.**

This language is standard differential geometry / subspace tracking territory, not a new branch of mathematics.

The possible residual is the interface between:

- statistical identifiability;
- geometric continuity;
- task consequence;
- reacquisition cost.

## Important prior-art direction

A 2026 paper discovered while Gate 3 was running is extremely close to the geometric framing:

**Lee et al., Stiefel Manifold Dynamical Systems for Tracking Representational Drift (2026).**

They explicitly model orthonormal emission matrices moving smoothly on the Stiefel manifold. Their skew-symmetric displacement separates:

- rotations **within** the latent subspace;
- motion that changes the subspace itself on the Grassmann manifold.

They also report that more neurally and behaviorally important dimensions drift less.

That paper means "representational drift as motion on Stiefel/Grassmann manifolds" is absolutely **not** ours.

MovingProblem's narrower question is downstream:

> when a task needs identities that the moving geometry does not itself preserve, can temporal statistics + sparse consequence maintain those identities cheaply?

See `PAPERS.md`.

## Next mathematical attack

The clean next quantity is not another accuracy number.

It is the **lock condition**.

For a time-varying identifying operator `L(t)`, identity should remain trackable while estimator error plus operator motion is small relative to the current eigenvalue/signature gap.

This points toward a bound of the rough form

```text
estimation error
+ movement during one update
< some fraction of signature gap
```

and a corresponding reacquisition rate when the trajectory approaches the degeneracy set.

That is a more defensible mathematical route than assuming random-matrix level-repulsion exponents apply automatically.

Raw summary: [gate3_summary.json](gate3_summary.json).
