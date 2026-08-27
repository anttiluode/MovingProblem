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

# Gate 1 — the real wall

Gate 0 still uses a fixed random hidden feature bank and a closed-form trained readout.

That is not the end goal.

The next question is:

> **Can hidden structure itself learn a useful nonlinear computation from local forward activity and delayed/scalar consequence, without reverse credit propagation, while beating matched node-perturbation / random-feature / Forward-Forward attackers?**

Mandatory attackers:

- fixed random features + ridge / RLS;
- LMS / delta readout;
- node perturbation / REINFORCE-style scalar learning;
- SPSA / evolution-strategy style perturbation;
- Forward-Forward;
- feedback alignment where applicable;
- ordinary backprop MLP;
- matched temporal source-separation front ends.

Kill conditions:

- if the hidden update is algebraically just backprop, kill the claim;
- if it is ordinary node perturbation with a new name, call it node perturbation;
- if fixed random features perform as well, hidden learning did not earn a role;
- if a standard adaptive BSS front-end solves the moving representation problem more simply, use it;
- if backprop wins and the non-backprop method has no compensating online / locality / hardware advantage, do not pretend otherwise.

## Run

```bash
python -m pip install -r requirements.txt
python experiments/gate0_moving_basis.py
python -m unittest discover -s tests -v
```

The experiment writes `results/gate0_summary.json`.

## Current sentence

> **MovingProblem is not trying to prove that brains secretly contain our matrices. It is asking whether temporal self-calibration, local consequence and persistent computation can produce an artificial learner that keeps working while the representation itself moves — and whether any genuinely useful non-backprop learning rule survives the attackers.**
