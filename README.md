# MovingProblem — give the system a moving problem

> **Static weights throw together at least three problems that can be separated: what to compute now, what evidence to gather next, and what should persist.**

This repository is an attempt to turn a long biological / signal-processing detour into a useful artificial machine.

The rule is simple:

> **Do not fake novelty. Do not count `no autograd` as a result. If the mechanism collapses to a known adaptive filter, random-feature model, source separator, node-perturbation method, Forward-Forward network, or ordinary optimizer, say so and keep the stronger ordinary method.**

## Provenance

The shortest useful lineage is:

```text
Perception-map accident
    ↓
geometric-neuron / FunctionalArbors
    ↓
Monday
persistent structure as computation
    ↓
Tuesday
ICA / AMUSE / SOBI / IVA and temporal identifiability
    ↓
yrotisopeRweN
continuous points, finite structural allocation, recurrent growth
    ↓
Twensday
what operators does the growing population actually produce?
    ↓
T-800NNP
continuous temporal traffic + delayed local consequence
    ↓
KyberDyyni1
fast internal sampling + slow consolidation
cross-basis transfer + temporal alignment + selective consequence
    ↓
MovingProblem
can any of this make a useful non-backprop machine?
```

Direct links:

- [FunctionalArbors](https://github.com/anttiluode/FunctionalArbors)
- [Monday](https://github.com/anttiluode/Monday)
- [Tuesday](https://github.com/anttiluode/Tuesday)
- [yrotisopeRweN](https://github.com/anttiluode/yrotisopeRweN)
- [Twensday](https://github.com/anttiluode/Twensday)
- [T-800NNP](https://github.com/anttiluode/T-800NNP)
- [KyberDyyni1](https://github.com/anttiluode/KyberDyyni1)

Biology is provenance, not specification. The project drifted away from a literal point-neuron model. That is intentional.

## What counts as "without backpropagation"

Allowed:

- matrix multiplication;
- eigendecomposition / SVD / whitening;
- local or closed-form regression;
- recurrent state;
- local eligibility / consequence;
- explicit zeroth-order probes, if identified honestly as such.

Not allowed when making a non-backprop claim:

- autograd;
- reverse-mode differentiation through layers;
- backpropagation through time;
- silently computing the same chain-rule error under another name.

A backprop-trained model is always allowed as an **attacker**.

# Gate 0 — preserve a nonlinear function while the observed basis moves

Gate 0 is deliberately modest.

There is one persistent latent process

```text
z(t) in R^4
```

whose components are independent Gaussian AR(1) processes with distinct temporal signatures.

The useful task is nonlinear:

```text
y = sign(
      z0*z1
      + 0.55*z2
      - 0.25*z3
      + 0.20*sin(2*z0)
    )
```

But each session renders the same latent process through a new unknown orthogonal basis:

```text
x(t) = Q_session z(t) + noise
```

The network learns the nonlinear task once in session 0.

After that the observation basis changes completely.

## Candidate machine

```text
unlabeled temporal stream x(t)
            │
            ▼
     one-lag AMUSE
  covariance + lag covariance
            │
            ▼
 reusable latent axes
   (up to global signs)
            │
            ▼
 tiny scalar-consequence
      sign calibration
            │
            ▼
 fixed tanh random-feature network
            │
            ▼
       frozen readout
```

There is no backward pass.

The first effective network operator changes because the temporal adapter changes. The nonlinear network downstream does **not** relearn its task after each basis change.

The sources are Gaussian on purpose. Their temporal statistics can identify axes, but not the orientation `s -> -s`. Gate 0 therefore does not hide the ambiguity. At rank 4 it explicitly tests all 16 sign maps and receives one scalar MSE consequence for each candidate map on a 32-example calibration batch.

## Result

20 seeds, four unseen basis changes per seed:

| method | accuracy after unseen basis changes |
|---|---:|
| temporal adapter + frozen nonlinear network | **0.9003 ± 0.0344** |
| zero-lag/PCA adapter + same sign search | 0.6262 ± 0.0996 |
| raw frozen random-feature network | 0.5432 ± 0.1039 |
| raw random-feature network, output refit on 32 labels | 0.7298 ± 0.0560 |
| oracle latent coordinates | 0.9578 ± 0.0060 |

This is a positive result, but **not yet a new learning algorithm**.

What it establishes is narrower:

> **A nonlinear function learned once can remain useful across large observation-basis changes when temporal structure reconstructs a reusable internal coordinate system and scalar consequence resolves the small ambiguity left by blind separation.**

The cleanest comparison is temporal versus zero-lag: both receive the same moving observations and the same sign-calibration machinery. Only the temporal adapter has the information needed to recover the individual Gaussian freedoms.

## Why this is not a novelty claim

Almost every ingredient is old:

- AMUSE / second-order blind source separation;
- adaptive blind source separation under changing mixtures;
- fixed random nonlinear features / extreme-learning-machine style readouts;
- ridge regression;
- brute-force sign calibration;
- context-invariant neural representations.

A particularly close prior problem is Naumann, Keijser & Sprekeler (eLife 2022), **Invariant neural subspaces maintained by feedback modulation**. They explicitly study dynamic blind source separation with a time-varying mixing matrix and use a learned recurrent modulator to maintain an invariant population representation.

Gate 0 differs operationally — it uses temporal source statistics directly and trains no recurrent modulator — but that is not enough to declare a new field or even a new algorithm.

See [PAPERS.md](PAPERS.md).

## What Gate 0 is good for

It gives the repo a working target that is more meaningful than another pytest demonstration:

```text
learn something once
        ↓
the representation moves
        ↓
do not retrain the whole network
        ↓
recover the useful freedoms from ongoing dynamics
        ↓
spend only a little consequence on what remains ambiguous
        ↓
keep using the learned nonlinear computation
```

That is an actual non-backprop use case.

# Gate 1 — hidden temporal structure learns without backprop

Gate 1 resurrects a problem we had already solved elsewhere, then removes the old optimizer.

The ancestry is unusually clean:

- [GeometricNeuronV9](https://github.com/anttiluode/GeometricNeuronV9) identified direction with the skew half of a lag operator.
- [RecurrentGeometricNet](https://github.com/anttiluode/RecurrentGeometricNet) learned a time-arrow classifier, but inspection of the actual code shows that its hidden filters/edges were trained with PyTorch `loss.backward()`.
- [GeoNeuronX Gate 5](https://github.com/anttiluode/GeoNeuronX/blob/main/results/GATE5.md) had an explicit NumPy Sanger/Oja learner that specialized temporal axes without source labels.
- [yrotisopeRweN Gates 8–10](https://github.com/anttiluode/yrotisopeRweN/blob/main/results/GATE8.md) had delayed eligibility, scalar consequence, and finite structural allocation.

Gate 1 asks whether those latter two pieces can replace the backprop hidden learner on a direction task.

## Harder moving-basis world

Each example is a 32-channel stream:

```text
one coherent 2-D chirped rotation
        +
30 independent nuisance processes
        ↓
fresh unknown 32-D orthogonal observation basis
```

The opposite class is the **exact time reverse**, including noise.

Therefore ordinary power is identically matched.

## Hidden learning

After zero-lag whitening (still a global convenience), each unlabeled sequence produces a skew lag matrix:

```text
A_i = 1/2 E[
      x_t x_(t-lag)^T
    - x_(t-lag) x_t^T
]
```

Time reversal flips `A_i -> -A_i`, but leaves `A_i A_i^T` unchanged.

Columns of the per-sequence skew matrices are presented to an explicit Sanger generalized-Hebbian population. Six hidden axes self-organize using only forward activity and lateral competition.

No labels.

No autograd.

No backward error.

The six axes expose 15 antisymmetric pair features. Each pair receives two possible orientations, and all 30 candidates compete for one conserved positive structural budget. Only 16 calibration sequences receive task consequence. The consequence arrives three events after the local feature, so an eligibility packet is required.

## Result

12 independently rotated 32-D observation bases:

| method | held-out accuracy |
|---|---:|
| **learned skew-energy axes + delayed structural consequence** | **0.8800 ± 0.0476** |
| random six hidden axes + same consequence learner | 0.5500 ± 0.0950 |
| shuffle time only while hidden axes learn | 0.5400 ± 0.0852 |
| erase eligibility | 0.5000 |
| power-only on same learned axes | 0.5000 |
| full 496-coordinate skew field + ridge on same 16 labels | **0.8783 ± 0.0506** |

This is the first gate here where **hidden structure itself learns without backprop and beats the matched fixed-random hidden representation**.

It is not an optimizer victory. A boring supervised ridge readout over all 496 skew coordinates is in the same performance range.

The earned statement is narrower:

> **Unlabeled temporal self-organization can compress a 32-D moving representation into a small directed feature repertoire on which delayed scalar consequence matches supervised fitting over the full skew field to within this development gate's noise.**

See [results/GATE1.md](results/GATE1.md).

## What Gate 1 still does not solve

- global batch whitening remains;
- arbitrary deep credit assignment remains unsolved;
- the sequence is still chunked rather than truly endless;
- the structural selector is shallow;
- ordinary adaptive BSS / supervised matrix methods remain mandatory attackers;
- recombining old mechanisms is not evidence of novelty.

The next clean attack is to make the observation basis change **during one uninterrupted stream**, let the hidden skew axes reorganize online, and ask whether the task selector can persist instead of being relearned.


# Gate 2 — stable function behind a drifting representation

Gate 2 replaces the artificial basis jump with another learner.

The upstream network is a paper-inspired Hebbian/anti-Hebbian similarity-matching population from Qin et al. (2021). After burn-in it remains plastic under noisy local updates, so its coordinates drift inside a useful representational equivalence class.

The deliberate stress-test modification is:

```text
three useful latent directions
same zero-lag variance
different temporal autocorrelation
```

Therefore instantaneous covariance can recover the useful subspace but cannot say which current axis is which. Temporal statistics can.

## Did it really drift?

Across 12 seeds, the same fixed probe bank changed substantially in raw coordinates while its relational geometry barely moved:

```text
raw coordinate change     0.7280 ± 0.2917
Gram/similarity change    0.0266 ± 0.0105
```

So this is no longer "press a button and rotate Q". The upstream learner itself keeps changing.

## Two tasks expose the boundary

### A. Task only needs geometry

Classify whether consecutive latent states are relatively similar, using only the current output inner product.

```text
geometry-only accuracy    0.9909 ± 0.0005
```

No axis recovery is needed. The invariant is already enough.

### B. Task needs one named/oriented freedom

Classify the sign of the latent source with the strongest temporal autocorrelation.

| method | accuracy |
|---|---:|
| **AMUSE track continuously; 16 labels once** | **0.9323 ± 0.0068** |
| AMUSE recalibrate every window | 0.9243 ± 0.0151 |
| frozen coordinate decoder | 0.8442 ± 0.0466 |
| zero-lag PCA tracker | 0.5020 ± 0.0521 |
| rotation-invariant features | 0.4985 ± 0.0131 |

The frozen decoder degrades further as drift accumulates:

```text
last-quarter frozen       0.7720 ± 0.0965
last-quarter AMUSE        0.9323 ± 0.0167
```

And the calibration budget changes from:

```text
track after one calibration    16 labels
recalibrate every window       800 labels
```

So Gate 2 earns a cleaner sentence than "always realign":

> **Use invariants when the task is invariant. Spend temporal/statistical identification and consequence only on the residual degrees of freedom that action actually requires.**

This is also the first gate where smooth representational drift turns repeated calibration into **continuous lock maintenance**.

See [results/GATE2.md](results/GATE2.md).

## What Gate 2 does not claim

- the brain runs AMUSE;
- the Qin et al. model contains this downstream solution;
- real neural or sensor drift is always rotational;
- real temporal signatures stay separated;
- superiority to adaptive BSS;
- superiority to Euclidean/Riemannian alignment on geometry-sufficient tasks;
- a new BSS algorithm.

Those are now the attackers.



# Gate 3 — do not learn through ambiguity

Gate 3 stress-tests the continuous-lock idea instead of adding more machinery.

The controlled harness forces:

- crowded temporal signatures;
- two signatures crossing;
- sustained exact degeneracy;
- source disappearance and return;
- non-orthogonal mixing;
- the useful 3-D subspace itself bending through a 5-D ambient space.

The tiny rule is:

> **When the current statistics become temporarily non-identifying, do not let that interval rewrite a previously grounded semantic identity.**

The tracker keeps a trusted frame, aligns new temporal frames by geometry plus signature continuity, and temporarily holds the old map when the signature gap collapses abruptly.

## Result

| attack / baseline | accuracy |
|---|---:|
| separated temporal tracker | **0.9330 ± 0.0128** |
| EA-style zero-lag recenter + frozen readout | 0.6741 ± 0.1333 |
| crossing, trust eigenvalue order | 0.7226 ± 0.0065 |
| **crossing, guarded continuity** | **0.9507 ± 0.0052** |
| guarded after crossing (last 10 windows) | **0.9649 ± 0.0053** |
| recalibrate from 16 labels every window | 0.9360 ± 0.0115 |
| paired Procrustes, 16 anchors every window | **0.9935 ± 0.0005** |
| sustained exact degeneracy | 0.4731 ± 0.1895 |
| source exactly absent | 0.4990 ± 0.0068 |
| same source after return | **0.9332 ± 0.0151** |
| non-orthogonal mixing | **0.9328 ± 0.0127** |
| deforming 3-D subspace in 5-D | **0.9321 ± 0.0122** |

Calibration / correspondence cost in the 40-window crossing world:

```text
guarded tracker                    16 task labels once
task recalibration                 640 labels
paired Procrustes                  16 paired anchors every window
```

Procrustes is the right answer when repeated paired correspondences exist.

Sustained exact degeneracy is the right failure when no statistic distinguishes two freedoms.

The result that survived is therefore not "AMUSE always wins". It is:

> **Preserve identity through brief intervals in which identification becomes impossible; only replace that identity when the observations again contain enough information to do so.**

See [results/GATE3.md](results/GATE3.md).

A very relevant 2026 prior-art result is Lee et al., **Stiefel Manifold Dynamical Systems for Tracking Representational Drift**. It explicitly separates rotations *inside* a latent subspace from motion of the subspace itself on the Grassmann manifold. So that geometric framing is established prior art, not a MovingProblem invention. See [PAPERS.md](PAPERS.md).


## Run

```bash
python -m pip install -r requirements.txt
python experiments/gate0_moving_basis.py
python experiments/gate1_local_temporal_learning.py
python experiments/gate2_drifting_representation.py
python experiments/gate3_stress_map.py
python -m unittest discover -s tests -v
```

The experiment runners write `results/gate0_summary.json` through `results/gate3_summary.json`.

## Current sentence

> **MovingProblem asks what must actually remain stable when representations keep moving. The current answer is: use relational invariants where they suffice; recover named freedoms only when action requires them; preserve grounded identity through temporary ambiguity; request new information only when identifiability is genuinely lost.**
