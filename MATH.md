# MovingProblem math note — moving subspaces, moving frames, and semantic lock

This note separates established mathematics from the possible residual that MovingProblem is actually testing.

## 1. The geometry is not new

Let the task-relevant population representation occupy a k-dimensional subspace of an n-dimensional observation space.

The **subspace itself** is a point on the Grassmann manifold:

```text
S(t) in Gr(k,n)
```

Choose an orthonormal basis for that subspace:

```text
Q(t) in St(n,k)
```

on the Stiefel manifold.

But

```text
Q(t)
```

and

```text
Q(t) R(t),   R(t) in O(k)
```

span the same Grassmann subspace.

That internal rotation is an equivalence / gauge freedom whenever the downstream computation only depends on the subspace or its relational geometry.

This is standard differential geometry and standard subspace-tracking language.

A directly relevant 2026 prior-art example is Lee et al., **Stiefel Manifold Dynamical Systems for Tracking Representational Drift**, which explicitly separates:

- rotation within the latent subspace;
- motion that changes the subspace itself on the Grassmann manifold.

See [PAPERS.md](PAPERS.md).

## 2. Where PCA / ICA / AMUSE enter

Suppose the current representation has an internal statistical operator

```text
L(t)
```

such as:

- zero-lag covariance;
- lag covariance;
- a jointly diagonalized family of lag covariances;
- a higher-order cumulant;
- a skew / antisymmetric temporal operator.

Different statistics break different parts of the frame ambiguity.

A useful schematic is:

```text
subspace only
    continuous O(k) frame freedom remains

+ identifying temporal / non-Gaussian statistics
    individual axes become identifiable
    except permutation / sign residues

+ cross-view coupling
    ties those residues across views

+ task consequence
    assigns semantic orientation where statistics cannot
```

This is not a claim that a brain literally runs PCA, ICA, AMUSE, IVA or IVE.

It is the weaker statement that local dynamics can produce effective operators whose invariant subspaces / axes are described by the same mathematics.

## 3. The discriminant set

Let the eigenvalues / temporal signatures of L(t) be

```text
lambda_1(t), ..., lambda_k(t).
```

Individual eigenvectors are well-defined only while their relevant gaps are nonzero.

Define the degeneracy / discriminant set informally as

```text
Delta = { L : lambda_i = lambda_j for some required i != j }.
```

Away from Delta, statistics can select a frame.

At Delta, the individual axes inside the degenerate eigenspace are not statistically identifiable.

Gate 3 measured exactly this distinction:

```text
brief crossing:
    bridge it by continuity

sustained exact degeneracy:
    individual identity fails
```

The important engineering consequence is:

> **Do not let an observation interval that contains no identity information overwrite a previously grounded identity.**

## 4. Semantic tracking as frame transport

MovingProblem's current tracker can be interpreted as a crude frame-transport rule.

When L(t) is well identified:

```text
estimate current frame
align to previous trusted frame
update trusted identity
```

When the gap collapses:

```text
do not update trusted identity
coast using the previous grounded frame
```

When the gap returns:

```text
re-lock by geometric / temporal continuity
```

This resembles parallel transport in spirit, but the current implementation is **not** a differential-geometric parallel-transport algorithm and should not be described as one.

The interesting object is a **task-conditioned moving frame**:

- geometry determines what transformations are equivalent;
- stream statistics break the equivalence where they can;
- task consequence fixes the residual meaning;
- continuity carries that meaning forward.

## 5. A candidate lock condition

Let:

```text
g(t)      current identifying signature gap
epsilon   statistical estimation error
delta     operator movement over one tracking update
```

Davis-Kahan-style perturbation results suggest eigenvector error scales roughly like

```text
angle error ~ epsilon / g
```

when the gap is nonzero.

A plausible identity-lock condition therefore has the form

```text
estimation error
+ movement during one update
< c * current gap
```

for some constant and a precisely specified operator norm / matching rule.

This is not yet a theorem for MovingProblem.

It is the next theorem-shaped question.

A useful version would prove that nearest-frame matching preserves permutation/sign identity under a quantitative gap + motion + estimation bound.

## 6. The D^(1/4) window law

Suppose a window contains N samples.

A simple model of tracking error is

```text
e(N)
    ~ a / (g sqrt(N))
      + b sqrt(D N)
```

where:

- the first term is finite-sample estimation error;
- g is the identifying gap;
- D is a diffusive frame-drift coefficient;
- the second term is how far the frame moves while the window is being collected.

Optimizing gives

```text
N* = a / (b g sqrt(D))
```

and

```text
e* = 2 sqrt(a b) D^(1/4) / sqrt(g).
```

The quarter-power is therefore a straightforward consequence of this two-term model.

That type of estimation-vs-tracking tradeoff is standard adaptive-filter / subspace-tracking mathematics.

What would be useful is to test whether MovingProblem's measured optimum actually follows it.

If not, the assumed drift or estimator-error model is wrong.

## 7. Reacquisition cost as distance to degeneracy

The more interesting quantity for a product is not average angle error.

It is:

```text
how often does semantic lock become unsafe?
```

One possible mathematical program is:

1. define an unsafe tube around Delta using the finite-sample and drift bound;
2. follow the trajectory L(t);
3. count excursions into that tube;
4. measure how long each excursion lasts;
5. determine whether continuity can bridge it;
6. request task consequence only when the excursion exceeds the bridgeable duration.

Then:

```text
label / calibration rate
    =
reacquisition cost
    x
rate of unbridgeable visits to the unsafe region.
```

This is a more direct route to a useful "labels per hour" law than accuracy alone.

## 8. Caution on random-matrix level repulsion

A tempting conjecture is that lock-loss rates could be predicted from random-matrix level-repulsion exponents.

That may eventually be useful, but it is **not established here**.

In particular, taking a real signal, forming its analytic/Hilbert representation, and obtaining a complex Hermitian covariance does **not automatically prove** that the relevant estimator belongs to a GUE-like beta=2 ensemble or that near-degeneracies become quadratically rarer.

The data remain strongly structured and the real/imaginary parts are not arbitrary independent complex samples.

So:

> **real-vs-analytic crossing-rate differences should be measured before invoking Wigner-Dyson exponents.**

If a robust exponent appears, then derive the ensemble assumptions afterward.

## 9. The user's "geometry changes as the signal passes" intuition

A fixed matrix W is only a complete description for a fixed linear map.

A stateful nonlinear population is better described locally by something like

```text
J(x, h, t) = partial output / partial input
```

where h contains recurrent / adaptive state.

As the signal and state evolve, so does the local effective operator and therefore its active subspaces.

In biological hardware, candidate causes include:

- conductance state;
- dendritic nonlinearities;
- recurrent inhibition/excitation;
- adaptation;
- propagation delays;
- short- and long-term plasticity.

So the matrix can be our **local external description of wet computation**, not a literal matrix stored by the tissue.

This is established mathematical language (state-dependent Jacobians, tangent spaces, invariant subspaces), not evidence that neurons implement our specific algorithms.

## 10. What might actually be worth proving

A realistic small theorem would be something like:

> **Given a smoothly varying identifying operator with a finite spectral gap, bounded estimation error, and bounded inter-update motion, a continuity-matched frame preserves semantic axis identity without repeated labels. If the path crosses a degenerate set, identity can still be preserved without labels provided the ambiguous interval is short enough relative to the accumulated frame motion.**

Then attack it numerically with:

- gap sweeps;
- sample-length sweeps;
- drift-rate sweeps;
- exact crossings;
- sustained degeneracy;
- source disappearance;
- Grassmann subspace motion.

That would not be "new mathematics" in the grand sense.

It could, however, be a new useful theorem about the specific interface MovingProblem cares about:

```text
statistical identifiability
    +
moving representation geometry
    +
semantic / task grounding
    +
calibration cost.
```
