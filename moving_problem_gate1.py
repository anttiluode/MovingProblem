"""MovingProblem Gate 1 — learn a compact temporal hidden basis without backprop.

This deliberately recombines three older repository results:
- GeometricNeuronV9 / RecurrentGeometricNet: direction lives in a skew lag operator.
- GeoNeuronX Gate 5: explicit Sanger/Oja population learning.
- yrotisopeRweN Gates 8-10: delayed eligibility + finite positive structure.

No autograd package is imported. No reverse error derivative reaches the hidden
learner. Batch whitening remains a stated global convenience.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

Array = np.ndarray


def random_orthogonal(rng: np.random.Generator, dim: int) -> Array:
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _ar1(rng: np.random.Generator, n: int, rho: float) -> Array:
    x = np.zeros(n, dtype=float)
    e = rng.normal(size=n)
    x[0] = e[0]
    scale = np.sqrt(max(1.0 - rho * rho, 1e-12))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + scale * e[t]
    return x


def directional_pair(
    seed: int,
    *,
    n_steps: int = 256,
    dim: int = 32,
    noise_std: float = 0.03,
) -> Tuple[Array, Array]:
    """One 32-D sequence and its exact time reverse.

    One two-dimensional chirped rotation carries direction. The remaining
    coordinates are independent AR(1) nuisance processes. Noise is added before
    reversal so all time-order-blind statistics remain exactly paired.
    """
    if dim < 4:
        raise ValueError("dim must be >= 4")
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=float)
    cycles_per_step = 0.05 + 0.20 * t / n_steps
    phase = 2.0 * np.pi * np.cumsum(cycles_per_step)
    directed = np.column_stack([np.cos(phase), np.sin(phase)])

    rhos = np.linspace(0.05, 0.95, dim - 2)
    nuisance = np.column_stack([_ar1(rng, n_steps, rho) for rho in rhos])
    z = np.column_stack([directed, nuisance])
    z = (z - z.mean(axis=0, keepdims=True)) / (
        z.std(axis=0, keepdims=True) + 1e-12
    )
    z += noise_std * rng.normal(size=z.shape)
    return z, z[::-1].copy()


def make_session(
    n_pairs: int,
    seed: int,
    *,
    dim: int = 32,
    basis: Array | None = None,
) -> Tuple[Array, Array, Array]:
    """Render exact reversal pairs through one unknown orthogonal basis."""
    rng = np.random.default_rng(seed + 91_337)
    if basis is None:
        basis = random_orthogonal(rng, dim)

    sequences: List[Array] = []
    labels: List[float] = []
    for i in range(n_pairs):
        up, down = directional_pair(seed * 10_000 + i, dim=dim)
        sequences.extend([up @ basis.T, down @ basis.T])
        labels.extend([1.0, -1.0])
    return np.stack(sequences), np.asarray(labels), basis


@dataclass
class Whitener:
    mean: Array
    matrix: Array

    def transform(self, x: Array) -> Array:
        return (x - self.mean) @ self.matrix


def fit_whitener(x: Array, eps: float = 1e-6) -> Whitener:
    """Batch zero-lag whitening; global convenience, not a locality claim."""
    flat = np.asarray(x, dtype=float).reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0)
    xc = flat - mean
    cov = (xc.T @ xc) / len(xc)
    cov = 0.5 * (cov + cov.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    matrix = vectors @ np.diag(1.0 / np.sqrt(values + eps))
    return Whitener(mean=mean, matrix=matrix)


def sequence_skew_matrices(x: Array, lag: int = 3) -> Array:
    """Per-sequence antisymmetric lag operator."""
    current = x[:, lag:, :]
    past = x[:, :-lag, :]
    forward = np.einsum("bti,btj->bij", current, past)
    backward = np.einsum("bti,btj->bij", past, current)
    return (forward - backward) / (2.0 * (x.shape[1] - lag))


def sanger_population(
    samples: Array,
    n_outputs: int,
    *,
    lr: float = 0.25,
    epochs: int = 5,
    seed: int = 0,
) -> Array:
    """Explicit generalized Hebbian / Sanger update, no autograd."""
    data = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(data.shape[1], n_outputs)))
    weights = q[:, :n_outputs].T.copy()

    for _ in range(int(epochs)):
        for u in data:
            y = weights @ u
            for i in range(n_outputs):
                reconstruction = np.sum(
                    y[: i + 1, None] * weights[: i + 1],
                    axis=0,
                )
                weights[i] += lr * y[i] * (u - reconstruction)
    return weights.T


def learn_skew_energy_axes(
    x_whitened: Array,
    *,
    n_axes: int = 6,
    lag: int = 3,
    seed: int = 0,
    shuffle_time: bool = False,
) -> Array:
    """Learn high skew-energy axes without class labels.

    Exact reversal changes A_i to -A_i. Presenting columns of all A_i matrices
    to Sanger makes their covariance proportional to sum A_i A_i^T, so forward
    and reverse reinforce the same unsigned rotation subspace rather than cancel.
    """
    x_used = np.asarray(x_whitened).copy()
    if shuffle_time:
        rng = np.random.default_rng(seed + 700_001)
        for sequence in x_used:
            rng.shuffle(sequence, axis=0)

    skew = sequence_skew_matrices(x_used, lag=lag)
    column_samples = np.concatenate([a.T for a in skew], axis=0)
    return sanger_population(
        column_samples,
        n_axes,
        lr=0.25,
        epochs=5,
        seed=seed,
    )


def random_axes(dim: int, n_axes: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, n_axes)))
    return q[:, :n_axes]


def pair_arrow_features(x_whitened: Array, axes: Array, lag: int = 3) -> Array:
    """Signed antisymmetric cross-time feature for every axis pair."""
    q = x_whitened @ axes
    current = q[:, lag:, :]
    past = q[:, :-lag, :]
    skew = np.einsum("bti,btj->bij", current, past)
    skew -= np.einsum("bti,btj->bij", past, current)
    skew /= 2.0 * (q.shape[1] - lag)
    idx = np.triu_indices(q.shape[2], 1)
    return skew[:, idx[0], idx[1]]


def power_features(x_whitened: Array, axes: Array) -> Array:
    q = x_whitened @ axes
    return np.mean(q * q, axis=1)


@dataclass
class FeatureStats:
    mean: Array
    std: Array

    @classmethod
    def fit(cls, x: Array) -> "FeatureStats":
        return cls(x.mean(axis=0), x.std(axis=0) + 1e-6)

    def transform(self, x: Array) -> Array:
        return (x - self.mean) / self.std


def delayed_consequence_mass(
    calibration_features: Array,
    calibration_target: Array,
    stats: FeatureStats,
    *,
    delay: int = 3,
    gain: float = 0.5,
    learning_rate: float = 0.25,
    reserve: float = 1e-5,
    keep_eligibility: bool = True,
) -> Array:
    """One-pass positive-only allocation under delayed scalar consequence."""
    f = stats.transform(calibration_features)
    candidates = np.concatenate([f, -f], axis=1)
    mass = np.full(candidates.shape[1], 1.0 / candidates.shape[1])

    queue: List[Tuple[int, Array, float, float]] = []
    for t in range(len(candidates) + delay):
        if t < len(candidates):
            prediction = float(gain * (candidates[t] @ mass))
            eligibility = (
                candidates[t] * mass
                if keep_eligibility
                else np.zeros_like(mass)
            )
            queue.append(
                (t + delay, eligibility.copy(), float(calibration_target[t]), prediction)
            )

        due = [item for item in queue if item[0] == t]
        queue = [item for item in queue if item[0] != t]
        for _, eligibility, target, old_prediction in due:
            error = target - old_prediction
            growth = np.maximum(error * eligibility, 0.0)
            mass += learning_rate * growth
            mass = np.maximum(mass, reserve)
            mass /= mass.sum() + 1e-12
    return mass


def mass_predict(
    features: Array,
    stats: FeatureStats,
    mass: Array,
    *,
    gain: float = 0.5,
) -> Array:
    f = stats.transform(features)
    candidates = np.concatenate([f, -f], axis=1)
    return gain * (candidates @ mass)


def accuracy(prediction: Array, target: Array) -> float:
    labels = np.where(prediction >= 0.0, 1.0, -1.0)
    return float(np.mean(labels == target))


def full_skew_features(x_whitened: Array, lag: int = 3) -> Array:
    skew = sequence_skew_matrices(x_whitened, lag=lag)
    idx = np.triu_indices(x_whitened.shape[2], 1)
    return skew[:, idx[0], idx[1]]


def ridge_attacker(
    unlabeled_features: Array,
    calibration_features: Array,
    calibration_target: Array,
    test_features: Array,
    *,
    ridge: float = 10.0,
) -> Array:
    stats = FeatureStats.fit(unlabeled_features)
    a = stats.transform(calibration_features)
    b = stats.transform(test_features)
    alpha = np.linalg.solve(
        a @ a.T + ridge * np.eye(len(a)),
        calibration_target,
    )
    return b @ (a.T @ alpha)


def power_ridge_attacker(
    unlabeled_features: Array,
    calibration_features: Array,
    calibration_target: Array,
    test_features: Array,
) -> Array:
    stats = FeatureStats.fit(unlabeled_features)
    a = stats.transform(calibration_features)
    b = stats.transform(test_features)
    coef = np.linalg.solve(
        a.T @ a + np.eye(a.shape[1]),
        a.T @ calibration_target,
    )
    return b @ coef


def run_gate1_seed(
    seed: int,
    *,
    dim: int = 32,
    n_axes: int = 6,
    unlabeled_pairs: int = 40,
    calibration_pairs: int = 8,
    test_pairs: int = 50,
    lag: int = 3,
) -> Dict[str, float]:
    """One fresh observation basis; only 16 consequence examples are labeled."""
    unlabeled, _, basis = make_session(unlabeled_pairs, seed, dim=dim)
    calibration, y_cal, _ = make_session(
        calibration_pairs, seed + 100, dim=dim, basis=basis
    )
    test, y_test, _ = make_session(
        test_pairs, seed + 200, dim=dim, basis=basis
    )

    whitener = fit_whitener(unlabeled)
    u = whitener.transform(unlabeled)
    c = whitener.transform(calibration)
    t = whitener.transform(test)

    learned_axes = learn_skew_energy_axes(u, n_axes=n_axes, lag=lag, seed=seed)
    learned_u = pair_arrow_features(u, learned_axes, lag)
    learned_c = pair_arrow_features(c, learned_axes, lag)
    learned_t = pair_arrow_features(t, learned_axes, lag)
    learned_stats = FeatureStats.fit(learned_u)
    learned_mass = delayed_consequence_mass(learned_c, y_cal, learned_stats)
    learned_pred = mass_predict(learned_t, learned_stats, learned_mass)

    fixed_axes = random_axes(dim, n_axes, seed + 333)
    random_u = pair_arrow_features(u, fixed_axes, lag)
    random_c = pair_arrow_features(c, fixed_axes, lag)
    random_t = pair_arrow_features(t, fixed_axes, lag)
    random_stats = FeatureStats.fit(random_u)
    random_mass = delayed_consequence_mass(random_c, y_cal, random_stats)
    random_pred = mass_predict(random_t, random_stats, random_mass)

    shuffled_axes = learn_skew_energy_axes(
        u, n_axes=n_axes, lag=lag, seed=seed, shuffle_time=True
    )
    shuffled_u = pair_arrow_features(u, shuffled_axes, lag)
    shuffled_c = pair_arrow_features(c, shuffled_axes, lag)
    shuffled_t = pair_arrow_features(t, shuffled_axes, lag)
    shuffled_stats = FeatureStats.fit(shuffled_u)
    shuffled_mass = delayed_consequence_mass(
        shuffled_c, y_cal, shuffled_stats
    )
    shuffled_pred = mass_predict(shuffled_t, shuffled_stats, shuffled_mass)

    noelig_mass = delayed_consequence_mass(
        learned_c, y_cal, learned_stats, keep_eligibility=False
    )
    noelig_pred = mass_predict(learned_t, learned_stats, noelig_mass)

    p_u = power_features(u, learned_axes)
    p_c = power_features(c, learned_axes)
    p_t = power_features(t, learned_axes)
    power_pred = power_ridge_attacker(p_u, p_c, y_cal, p_t)

    f_u = full_skew_features(u, lag)
    f_c = full_skew_features(c, lag)
    f_t = full_skew_features(t, lag)
    ridge_pred = ridge_attacker(f_u, f_c, y_cal, f_t)

    skew = sequence_skew_matrices(u, lag)
    skew_energy = np.mean(
        np.einsum("bij,bkj->bik", skew, skew),
        axis=0,
    )
    learned_capture = float(
        np.trace(learned_axes.T @ skew_energy @ learned_axes)
        / (np.trace(skew_energy) + 1e-12)
    )
    random_capture = float(
        np.trace(fixed_axes.T @ skew_energy @ fixed_axes)
        / (np.trace(skew_energy) + 1e-12)
    )

    return {
        "seed": float(seed),
        "learned_local": accuracy(learned_pred, y_test),
        "random_hidden": accuracy(random_pred, y_test),
        "shuffled_hidden_time": accuracy(shuffled_pred, y_test),
        "no_eligibility": accuracy(noelig_pred, y_test),
        "power_only": accuracy(power_pred, y_test),
        "full_skew_ridge": accuracy(ridge_pred, y_test),
        "learned_skew_energy_capture": learned_capture,
        "random_skew_energy_capture": random_capture,
        "power_pair_max_difference": float(
            np.max(np.abs(p_t[0::2] - p_t[1::2]))
        ),
        "structural_effective_candidates": float(
            1.0 / np.sum(learned_mass * learned_mass)
        ),
        "structural_largest_mass": float(np.max(learned_mass)),
    }


def summarize_gate1(seeds: int = 12) -> Dict[str, Dict[str, float]]:
    rows = [run_gate1_seed(seed) for seed in range(seeds)]
    summary: Dict[str, Dict[str, float]] = {}
    for key in rows[0]:
        if key == "seed":
            continue
        values = np.asarray([row[key] for row in rows])
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return summary
