# Papers and prior art map

This file exists to prevent MovingProblem from rediscovering old machinery and calling it new.

## AMUSE / temporal blind source separation

AMUSE is a second-order blind source separation method. In the square stationary case it whitens the observations and diagonalizes a lagged autocovariance. Distinct lag signatures can identify source axes that zero-lag covariance cannot.

MovingProblem uses one-lag AMUSE in Gate 0 because the synthetic Gaussian latent sources have equal instantaneous variance but different AR(1) dynamics.

This is **old signal processing**, not a MovingProblem invention.

## Adaptive / nonstationary blind source separation

Time-varying mixtures have a substantial literature.

Examples include:

- V. Koivunen, M. Enescu & E. Oja, **Adaptive algorithm for blind separation from noisy time-varying mixtures**, *Neural Computation* 13(10), 2001. DOI: https://doi.org/10.1162/089976601750541822
- J.-T. Chien & H.-L. Hsieh, **Nonstationary Source Separation Using Sequential and Variational Bayesian Learning**, *IEEE TNNLS* 24(5), 2013. DOI: https://doi.org/10.1109/TNNLS.2013.2242090

Therefore:

> **"the mixing matrix moves and the separator adapts" is not new.**

## Naumann, Keijser & Sprekeler — eLife (2022)

**Invariant neural subspaces maintained by feedback modulation**

DOI: https://doi.org/10.7554/eLife.76096

This is especially close prior art.

They construct a dynamic blind-source-separation task

```text
x(t) = A(t) s(t) + noise
```

in which the mixing/context changes over time. A recurrent modulatory system uses input/output history to change feedforward gains so that the population maintains a context-invariant representation.

Important distinction from Gate 0:

- their modulator is trained supervised;
- Gate 0 directly estimates the changing coordinate system from temporal statistics and uses no backpropagation.

Important non-distinction:

- both are trying to preserve useful/invariant internal coordinates despite a moving observation map.

So Gate 0 should be read as a **different operational solution to an old class of problem**, not as proof of novelty.

## Hinton — Forward-Forward (2022)

Geoffrey Hinton, **The Forward-Forward Algorithm: Some Preliminary Investigations**

arXiv: https://arxiv.org/abs/2212.13345

Forward-Forward replaces the usual forward/backward training sequence with positive and negative forward passes and local layer objectives.

It is mandatory prior art for any claim of "neural network learning without backpropagation."

MovingProblem Gate 0 is not Forward-Forward: it does not train hidden layers at all. Gate 1 must compare against it if hidden learning is attempted.

## Random features / extreme-learning-machine family

Gate 0 uses a fixed random `tanh` feature bank and a ridge-trained output. This is an old and intentionally boring architecture family.

The point of Gate 0 is not the random-feature learner. The point is whether the **already learned nonlinear function remains usable after the raw coordinate system changes**.





## Qin et al. — coordinated representational drift (2021)

Shanshan Qin, Shiva Farashahi, David Lipshutz, Anirvan M. Sengupta, Dmitri B. Chklovskii & Cengiz Pehlevan, **Coordinated drift of receptive fields during noisy representation learning**.

bioRxiv: https://doi.org/10.1101/2021.08.30.458264

This paper is the direct provenance for MovingProblem Gate 2.

Its core linear model uses neural dynamics

```text
dy/dt = W x - M y
```

and local Hebbian / effective anti-Hebbian updates

```text
Delta W = eta (y x^T - W)
Delta M = eta (y y^T - M)
```

with synaptic noise added during continued learning.

The important observation is not merely that receptive fields drift. The network can continue to move through a degenerate family of equally good representations while preserving population representational similarity. In the linear case, the drift behaves approximately like a coordinated rotation of the represented cloud rather than independent motion of every unit.

The paper also explicitly notes that stable downstream behavior may require an adaptive readout when drift is not confined to a coding-null space, and leaves the mechanism of such adaptation open.

Gate 2 is **not an exact replication**. It deliberately changes the top-three input spectrum so those three directions have equal zero-lag variance but distinct temporal autocorrelation. That makes the rotation within the useful subspace impossible to resolve with zero-lag PCA alone, while AMUSE can still identify the temporal freedoms. This is a stress test of the paper's drift geometry, not a claim about the authors' original simulation.




## Lee et al. — Stiefel Manifold Dynamical Systems for representational drift (2026)

Hyun Dong Lee, Aditi Jha, Stephen E. Clarke, Michael P. Silvernagel, Paul Nuyujukian & Scott W. Linderman, **Stiefel Manifold Dynamical Systems for Tracking Representational Drift**.

Preprint / PMC record: https://pmc.ncbi.nlm.nih.gov/articles/PMC13060931/

This is very close prior art for the geometric language that emerged around Gate 2 and Gate 3.

SMDS treats the observation/emission matrix as an orthonormal frame that evolves smoothly across trials on the **Stiefel manifold**, while the underlying latent dynamics remain shared.

Its skew-symmetric displacement matrix separates two kinds of motion:

```text
W block
    rotations within the current latent subspace
    -> basis/frame changes that leave the Grassmann subspace fixed

V block
    motion orthogonal to the current subspace
    -> actual subspace drift on the Grassmann manifold
```

That distinction is exactly the one MovingProblem informally reached after Gate 2:

```text
basis drift inside a useful space
versus
the useful space itself moving
```

The paper also reports gradual within-session representational drift in macaque and rodent neural recordings and finds that dimensions carrying more neural/behavioral variance tend to drift less.

Its supplementary analysis explicitly shows that PCA can recover overall subspace drift while failing to recover **per-dimension** drift because individual principal directions can flip or swap. That is directly relevant to Gate 3's identity-crossing attack.

Therefore:

> **"model representational drift on Stiefel/Grassmann manifolds" is not a MovingProblem contribution.**

MovingProblem's residual question is narrower: can a downstream task preserve a **named/oriented freedom** through such drift using temporal identifiability, sparse task consequence, and conservative continuity, while spending new labels only when the observations cease to identify the freedom?

SMDS is also a future attacker: it uses a probabilistic state-space model with variational EM / extended Kalman smoothing and is much more sophisticated than the current windowed AMUSE tracker.


## Oja / Sanger — the hidden learner in Gate 1 is old

Gate 1's hidden update is not a new learning rule.

- E. Oja, **Simplified neuron model as a principal component analyzer**, *Journal of Mathematical Biology* 15, 267–273 (1982). DOI: https://doi.org/10.1007/BF00275687
- T. D. Sanger, **Optimal unsupervised learning in a single-layer linear feedforward neural network**, *Neural Networks* 2(6), 459–473 (1989). DOI: https://doi.org/10.1016/0893-6080(89)90044-0

Gate 1 uses Sanger's generalized Hebbian rule explicitly:

```text
y = W u
Delta w_i = eta y_i (u - sum_(j<=i) y_j w_j)
```

The only MovingProblem-specific move is what is fed into that mature learner: columns of per-sequence **skew lag matrices**. Their covariance is proportional to an unsigned skew-energy operator, so exact forward/reverse pairs reinforce the same rotation subspace instead of cancelling.

Therefore:

> **"Sanger learns hidden directions without backprop" is old. "Use Sanger on skew-energy samples, then let delayed consequence select among the resulting directed planes" is the tested recombination here, not yet a novelty claim.**

## Eligibility / three-factor credit is also prior art

The Gate 1 downstream rule has the ordinary shape

```text
local activity -> eligibility trace
later scalar consequence -> modulate the trace
```

This belongs to the broad family of eligibility-trace / three-factor learning rules rather than being a MovingProblem invention.

A relevant biological/computational example is:

- E. M. Izhikevich, **Solving the distal reward problem through linkage of STDP and dopamine signaling**, *Cerebral Cortex* 17(10), 2443–2452 (2007). DOI: https://doi.org/10.1093/cercor/bhl152

MovingProblem's implementation is much simpler and not an STDP model. The citation is here to mark the occupied conceptual territory: delayed global modulation meeting a locally retained trace is established prior art.


## Ji et al. — Current Biology (2025)

Z. Ji, T. Chu, S. Wu & N. Burgess, **A systems model of alternating theta sweeps via firing rate adaptation**.

DOI: https://doi.org/10.1016/j.cub.2024.08.059

This belongs to the biological provenance inherited through KyberDyyni1. It motivated continuous internally generated sampling, not Gate 0's AMUSE mathematics.

## Vollan et al. — bioRxiv (2026)

A. Z. Vollan, M. F. Schellenberger, M.-B. Moser & E. I. Moser, **Attention-like regulation of theta sweeps in the brain's spatial navigation circuit**.

DOI: https://doi.org/10.64898/2026.01.27.702083

Reported sweep direction, width and frequency are dynamically modulated with behavioral relevance. KyberDyyni1 tested a naive engineering translation of narrower/faster sampling and found that the literal rule did not improve its search task.

MovingProblem carries forward only the broader abstraction:

```text
sampling direction
sampling extent
sampling rate
```

may be separately controlled properties of an active system.

The biology does not specify the artificial learning rule.

## Novelty rule

Before calling a future MovingProblem mechanism new:

1. state the actual mathematical update;
2. compare it algebraically to backprop, feedback alignment, node perturbation, REINFORCE, SPSA/ES, Hebbian/Oja-family rules, target propagation, predictive coding and Forward-Forward as appropriate;
3. search the adaptive filtering / BSS / reservoir / continual-learning literature;
4. run matched attackers;
5. claim only the residual that survives.
