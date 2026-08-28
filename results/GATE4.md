# Gate 4 — topology breaks the gap guard

CI-verified, 16 seeds for noisy quantities.

## Why this gate

Gate 3's strongest rule was:

> when the identifying gap collapses temporarily, do not let that interval rewrite a previously grounded semantic identity.

That rule survives true crossings, source disappearance, non-orthogonal mixing and subspace deformation.

But it only watches **local identifiability**.

Gate 4 asks whether semantic identity can change after a path that is locally safe everywhere.

## The two-mode normal form

Near a real-symmetric two-mode degeneracy, remove the irrelevant trace and write

```text
L(a,b) =
    [ a   b ]
    [ b  -a ]
```

with eigenvalue gap

```text
g = 2 sqrt(a^2 + b^2).
```

The degeneracy is the single point

```text
(a,b) = (0,0).
```

The principal eigenLINE is perfectly well defined anywhere away from the origin.

The experiment transports the principal eigenvector continuously by choosing its sign at each step to maximize overlap with the previous vector.

That is essentially the local sign-continuity rule used by the Gate 3 tracker.

## Attack 1 — circle the degeneracy at a safe gap

Take

```text
a = cos(phi)
b = sin(phi)
phi: 0 -> 2 pi
```

The gap is constant:

```text
g = 2
```

so a gap guard with threshold 0.5 never fires.

Yet after one full loop:

```text
final oriented-eigenvector alignment with start = -1.0000
low-confidence fraction                        =  0.0000
minimum gap                                    =  2.0000
winding                                        =  1
```

The eigenLINE returned to itself.

The **oriented eigenvector did not**.

This is the real-symmetric sign holonomy / Longuet-Higgins geometric phase.

So Gate 3 had a genuine blind spot:

> **distance from degeneracy is not enough to preserve oriented semantic meaning.**

A path can remain safely far from the degeneracy and still return with a sign inversion.

## Attack 2 — topology, not path length

A loop of the same radius that does **not** enclose the origin:

```text
final alignment = +1
winding         =  0
```

Two complete loops around the degeneracy:

```text
final alignment = +1
winding         =  2
```

So the effect is Z2:

```text
odd winding  -> sign flip
even winding -> no net sign flip
```

It is not caused by long travel or accumulated numerical drift.

## Minimal repair — remember winding parity

Gate 4 does not pretend to solve topology in a large unknown system.

In this 2-D laboratory the degeneracy location is known, so the tracker can accumulate the winding of

```text
(a(t), b(t))
```

around the origin.

The semantic readout stores one additional global bit:

```text
(-1)^(winding parity).
```

Then:

```text
one enclosing loop

local continuity only        -1.0000
+ winding-parity memory       +1.0000
```

With small symmetric operator noise:

```text
local continuity only        -0.999924 ± 0.000113
+ winding-parity memory       +0.999924 ± 0.000113
```

This is not "a new topology algorithm."

It demonstrates that a purely local confidence rule is insufficient for an oriented semantic task.

## Important qualification

The sign of a real eigenvector is gauge.

If the downstream task depends only on the **eigenline**, projector, energy or other sign-invariant object, nothing failed.

The failure exists only when an external task has grounded an **orientation**:

```text
+ axis = left
- axis = right

or

+ freedom = open
- freedom = close
```

Then a topological sign holonomy can be a semantic error even though the statistical line returned correctly.

This is exactly MovingProblem's distinction:

```text
statistical identity
is not automatically
task semantic identity.
```

## Finite-sample crossing confound

Claude pointed out another problem with Gate 3's exact crossing.

At the population level, the independent-source lag operator can be exactly diagonal:

```text
b = 0
```

and two diagonal signatures can genuinely cross.

But a finite-sample estimate has off-diagonal noise of order

```text
1 / sqrt(N).
```

Therefore an observed sample operator will almost surely show an apparent avoided crossing even when the population operator is exactly degenerate.

Gate 4 models the estimated 2×2 operator at the exact population crossing as a symmetric Gaussian perturbation with entry scale

```text
sigma / sqrt(N).
```

Measured mean apparent gap:

```text
N = 128     0.19077 ± 0.00140
N = 4096    0.03373 ± 0.00035
```

Log-log fit over

```text
N = 128, 256, 512, 1024, 2048, 4096
```

gives

```text
gap ~ N^(-0.49934 ± 0.00359).
```

Almost exactly the expected `N^-1/2`.

So future avoided-crossing sweeps must quote coupling relative to the estimator noise floor:

```text
delta * sqrt(N) / sigma
```

rather than just quoting `delta`.

Gate 4 also checks two reference points at `N=1024`:

```text
delta / noise = 0.25
measured gap / noise ≈ 2.18
    -> estimator floor dominates

delta / noise = 4.0
measured gap / noise ≈ 8.14
    -> physical coupling dominates; population gap is 2*delta
```

## Landau-Zener connection — provenance, not prediction

The same matrix normal form

```text
[ vt   delta ]
[ delta -vt  ]
```

is the classical two-level Landau-Zener Hamiltonian.

That literature gives a precise adiabatic / diabatic transition law for **Schrodinger evolution**.

MovingProblem's nearest-frame tracker is not a quantum state obeying the Schrodinger equation.

Therefore the Landau-Zener probability is **not** a quantitative prediction for this algorithm.

The useful inheritance is conceptual:

```text
coupling scale
versus
rate of passage through a crossing
```

controls whether it is natural to follow an instantaneous eigenmode or preserve a bare/source identity.

That motivates a later source-separation experiment, but Gate 4 does not fake that experiment.

## What changed after Gate 4

Before:

```text
gap healthy:
    update identity

gap bad:
    freeze identity
```

After:

```text
LOCAL QUESTION:
is the current frame statistically identifiable?

GLOBAL QUESTION:
what path has the frame taken through the space
of identifiable operators?

TASK QUESTION:
does the downstream computation care about a line,
an orientation, an eigenmode, or a persistent source?
```

Those are different questions.

## Relation to the Stiefel EKF paper

Figueras, Persson & Viitasaari (2026), **Extended Kalman filtering on Stiefel manifolds**, gives a principled predict/update filter when the state or measurement is an orthonormal frame.

That is relevant to replacing Gate 3's hard 0/1 confidence guard with a continuous manifold-valued filter.

But it does not remove Gate 4's topological issue:

- a local Kalman gain can downweight an uncertain measurement;
- it does not by itself tell an oriented semantic readout that a closed path accumulated odd sign holonomy.

The paper also assumes a restrictive isotropic-noise setting and known anti-symmetric system dynamics in its presented algorithm. MovingProblem's near-degeneracy uncertainty is strongly direction-dependent, so a direct drop-in claim would be wrong.

See [PAPERS.md](../PAPERS.md) and [MATH.md](../MATH.md).

## Current residual

The machine now appears to need three kinds of memory:

```text
1. statistical memory
   what freedom does the current stream identify?

2. geometric memory
   how has the representational frame moved?

3. semantic memory
   what externally grounded meaning was attached to that freedom?
```

A matrix snapshot contains none of that history by itself.

That may be the most relevant connection back to the original biological intuition.

Raw summary: [gate4_summary.json](gate4_summary.json).
