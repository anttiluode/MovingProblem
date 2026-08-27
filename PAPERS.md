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
