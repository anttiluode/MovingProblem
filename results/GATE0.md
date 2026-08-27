# Gate 0 receipt — nonlinear task under moving observation bases

## Question

Can a nonlinear computation learned once remain usable after the observed coordinates change, without backpropagating or retraining the whole network?

## World

Four independent unit-variance Gaussian AR(1) latent sources:

```text
rho = [0.95, 0.80, 0.60, 0.35]
```

Nonlinear target:

```text
y = sign(z0*z1 + 0.55*z2 - 0.25*z3 + 0.20*sin(2*z0))
```

Five sessions. Every session applies a fresh random orthogonal observation basis plus Gaussian observation noise:

```text
x = Q_session z + 0.05 * noise
```

The nonlinear task is learned only in session 0.

For every later session:

- first 2000 samples are used without task labels to estimate a coordinate adapter;
- 32 calibration examples are used for residual sign calibration / raw-readout attacker;
- the remainder is held out for evaluation.

## Methods

### Temporal adapter

One-lag AMUSE:

```text
zero-lag whitening
        ↓
lag-1 covariance eigendecomposition
        ↓
source axes ordered by temporal signature
```

Because Gaussian AR sources are invariant under `s -> -s`, AMUSE cannot identify orientation. At rank 4 all 16 sign maps are explicitly tested. Each candidate receives one scalar MSE consequence on the calibration batch.

The downstream network is a fixed 128-unit `tanh` random-feature bank with a ridge-trained output learned in session 0. It is frozen after that.

### Attackers

- zero-lag PCA/whitening + the **same** 16-way sign search;
- raw frozen random-feature network;
- raw random-feature network with its output refit by ridge on the 32 calibration labels;
- oracle latent coordinates.

## 20-seed result

Four unseen basis changes per seed, therefore 80 moving-session evaluations:

```text
temporal adapter + frozen nonlinear network   0.9003 ± 0.0344
zero-lag/PCA + same sign search               0.6262 ± 0.0996
raw frozen network                            0.5432 ± 0.1039
raw network + output refit on 32 labels       0.7298 ± 0.0560
oracle latent coordinates                     0.9578 ± 0.0060
```

## Interpretation

Positive:

> **Temporal structure can reconstruct a reusable coordinate system well enough that a nonlinear computation learned once survives otherwise destructive observation-basis changes.**

The zero-lag control is the important one. Both methods whiten the data and both receive the same sign-calibration search. Only the temporal adapter has enough information to identify the individual Gaussian freedoms.

The raw-refit attacker is also useful: with only 32 new task labels, relearning the output on the changed raw representation reaches about 0.73 rather than 0.90.

## What this does NOT establish

- a new blind-source-separation algorithm;
- a new random-feature network;
- hidden-layer learning;
- superiority to all adaptive BSS systems;
- superiority to backpropagation;
- biological plausibility;
- a general solution to representation drift.

Gate 0 is a working non-backprop architecture and a landing strip for harder attacks.

## Next wall

The fixed random hidden bank is now the loophole.

Gate 1 must ask whether hidden computation itself can adapt/learn from forward local activity plus delayed or scalar consequence, and it must survive comparison to known non-backprop and zeroth-order learners.
