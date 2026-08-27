# Gate 1 — local hidden temporal learning, no backprop

Development receipt. Positive result, **not a novelty claim**.

## Why this gate

Gate 0 kept a learned nonlinear function alive while the raw observation basis moved, but it used a fixed random hidden feature bank.

The archaeology then exposed three older pieces that had never been combined:

1. **GeometricNeuronV9 / RecurrentGeometricNet** had the right task and the right temporal object: direction lives in the skew half of a lag operator, and exact time reversal flips its sign.
2. **GeoNeuronX Gate 5** had an explicit NumPy Sanger/Oja population update that learns temporal structure without source labels.
3. **yrotisopeRweN Gates 8–10** had delayed eligibility plus scalar consequence plus finite positive structural allocation.

One correction matters: the old V9 / RecurrentGeometricNet "learned" hidden structures were trained with PyTorch autograd. Gate 1 does **not** inherit that optimizer.

## World

Each example is a 32-channel temporal stream.

Hidden inside it is:

```text
one coherent 2-D chirped rotation     <- carries direction
30 independent AR nuisance processes <- do not
```

Then a fresh unknown orthogonal matrix renders the 32 latent coordinates into the observed basis.

For every forward sequence, the opposite-class example is its **exact time reverse**, including noise:

```text
down(t) = up(T - 1 - t)
```

Therefore every time-order-blind power statistic is exactly matched.

Each development seed uses a different 32-D observation basis.

## Hidden learner

First, the unlabeled stream is zero-lag whitened. This is still a global batch convenience and is **not** claimed as local.

For every sequence, form the antisymmetric lag matrix

```text
A_i = 1/2 E_t[
        x_t x_(t-lag)^T
      - x_(t-lag) x_t^T
    ]
```

Exact reversal gives approximately / algebraically:

```text
A_reverse = -A_forward
```

so directly averaging A would cancel.

But:

```text
A_i A_i^T
```

does not care about that sign.

The columns of every A_i are therefore fed as samples into an explicit Sanger generalized-Hebbian population. Their covariance is proportional to the skew-energy operator

```text
sum_i A_i A_i^T
```

and the population learns six hidden axes with the ordinary forward update

```text
y = W u

Delta w_i =
    eta * y_i *
    (u - sum_(j<=i) y_j w_j)
```

No labels enter this stage.

No autograd enters this stage.

No error is propagated backward through this stage.

## Task features

The six learned axes make only

```text
6 choose 2 = 15
```

candidate directed plane features:

```text
L_ij =
    E_t[
        q_j(t) q_i(t-lag)
      - q_i(t) q_j(t-lag)
    ]
```

Each L flips sign on time reversal.

For finite positive structure, every feature receives two candidate orientations:

```text
+L_ij
-L_ij
```

so there are 30 structural candidates sharing one conserved nonnegative mass budget.

## Delayed consequence

Only 16 calibration sequences receive task consequence.

There is one pass.

At event time:

```text
eligibility_c = feature_c * current_mass_c
```

Three events later the scalar target error arrives:

```text
evidence_c = error * eligibility_c
growth_c   = max(evidence_c, 0)
```

Positive growth is followed by global renormalization of mass. There is no explicit signed anti-growth command.

This is deliberately the old yrotisopeRweN mechanism, not a renamed gradient.

## 12-basis development result

Each seed uses an independently rotated 32-D observation basis.

| method | held-out accuracy |
|---|---:|
| **Sanger skew-energy hidden axes + delayed structural consequence** | **0.8800 ± 0.0476** |
| random six hidden axes + same downstream learner | 0.5500 ± 0.0950 |
| learn hidden axes after shuffling time | 0.5400 ± 0.0852 |
| erase eligibility before consequence | 0.5000 |
| power-only on the same learned axes | 0.5000 |
| full 496-coordinate skew field + supervised ridge on same 16 labels | **0.8783 ± 0.0506** |

The learned six-axis subspace captures about

```text
0.2506 ± 0.0020
```

of the unlabeled skew energy, versus

```text
0.1887 ± 0.0041
```

for six random axes.

The power feature difference between each exact forward/reverse pair is at floating-point noise (~2e-15), so the 0.5 power result is structural, not a failed optimizer.

## What survived

### 1. Hidden learning earned a role against the matched random-hidden control

With the same six-axis budget, same 15 pair features, same 16 scalar-consequence examples, and same positive-mass learner:

```text
learned hidden axes   0.8800
random hidden axes    0.5500
```

So in this deliberately high-dimensional nuisance world, the hidden learner is not decorative.

### 2. Temporal order is load-bearing

Destroy temporal order only while the hidden axes learn:

```text
0.8800 -> 0.5400
```

The learner needs the skew temporal structure, not merely the frame marginals.

### 3. Delayed eligibility is load-bearing

Erase the local packet before scalar consequence arrives:

```text
0.8800 -> 0.5000
```

Consequence without an address does not select useful structure.

### 4. This is not an optimizer victory

A conventional supervised ridge model that receives the **entire 496-dimensional skew feature field** and the same 16 calibration labels reaches:

```text
0.8783
```

which is statistically in the same neighborhood.

That attacker is simpler if global supervised matrix fitting is allowed.

The narrower positive result is:

> **Unlabeled local-ish temporal self-organization can compress a 32-D moving representation into a six-axis, 15-pair directed feature repertoire on which a one-pass delayed scalar-consequence allocator performs essentially the same held-out accuracy as a supervised readout over the full 496-coordinate skew field in this development gate.**

That is useful enough to continue, but not enough to call the learning rule new.

## Why this is closer to the original point-neuron question

The hidden update is not told the task gradient.

Instead:

```text
ONGOING LOCAL TRAFFIC
        |
        v
discover temporal operator directions
        |
        v
small local repertoire
        |
        v
eligibility
        |
 delayed scalar consequence
        |
        v
finite persistent allocation
```

This is the first MovingProblem gate where **hidden structure itself changes without backpropagation** and where the learned hidden structure beats a matched fixed-random hidden representation.

## Remaining cheats / walls

- whitening is still batch/global;
- each sequence is presented as a bounded object rather than one endless stream;
- Sanger lateral competition is local-ish, not a biological synapse claim;
- task consequence is supplied;
- the final positive-mass selector is shallow;
- arbitrary deep credit assignment is still unsolved;
- full-skew ridge remains a strong ordinary attacker;
- no hardware/locality advantage is measured yet;
- no novelty claim survives prior art simply because these pieces were recombined.

## Next

Do **not** add more biological machinery yet.

The next attacks are obvious:

1. replace batch whitening with streaming decorrelation / homeostasis;
2. make the basis change **during** one uninterrupted stream rather than between development sessions;
3. retain the learned task selector while the hidden skew axes reorganize under that basis change;
4. compare consequence/sample cost against adaptive BSS + linear readout;
5. only then try a second nonlinear hidden stage.

Raw summary: [gate1_summary.json](gate1_summary.json).
