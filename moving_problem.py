"""Core machinery for MovingProblem Gate 0.

Everything here is NumPy matrix algebra. There is no autograd, no backward
pass through a network, and no backpropagation through time.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np


def random_orthogonal(rng: np.random.Generator, dim: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def generate_ar_latents(
    rng: np.random.Generator, n: int, rhos: np.ndarray
) -> np.ndarray:
    """Stationary unit-variance independent Gaussian AR(1) sources."""
    rhos = np.asarray(rhos, dtype=float)
    z = np.zeros((n, len(rhos)), dtype=float)
    z[0] = rng.normal(size=len(rhos))
    innovation_scale = np.sqrt(1.0 - rhos**2)
    for t in range(1, n):
        z[t] = rhos * z[t - 1] + innovation_scale * rng.normal(size=len(rhos))
    return z


def nonlinear_target(z: np.ndarray) -> np.ndarray:
    """A genuinely nonlinear binary target in the persistent latent basis."""
    if z.shape[1] < 4:
        raise ValueError("Gate 0 currently expects at least four latent dimensions")
    score = (
        z[:, 0] * z[:, 1]
        + 0.55 * z[:, 2]
        - 0.25 * z[:, 3]
        + 0.20 * np.sin(2.0 * z[:, 0])
    )
    return np.where(score >= 0.0, 1.0, -1.0)


@dataclass
class MovingStream:
    x: np.ndarray
    z: np.ndarray
    y: np.ndarray
    bases: List[np.ndarray]
    segment_length: int


def make_moving_stream(
    seed: int,
    *,
    dim: int = 4,
    segments: int = 5,
    segment_length: int = 6000,
    observation_noise: float = 0.05,
) -> MovingStream:
    """Create one latent task observed through a new orthogonal basis each session."""
    if dim != 4:
        raise ValueError("Gate 0 is frozen at dim=4 so the sign search stays explicit")
    rng = np.random.default_rng(seed)
    n = segments * segment_length
    rhos = np.array([0.95, 0.80, 0.60, 0.35], dtype=float)
    z = generate_ar_latents(rng, n, rhos)
    y = nonlinear_target(z)
    bases = [random_orthogonal(rng, dim) for _ in range(segments)]
    x = np.zeros_like(z)
    for j, q in enumerate(bases):
        sl = slice(j * segment_length, (j + 1) * segment_length)
        x[sl] = z[sl] @ q.T
        x[sl] += observation_noise * rng.normal(size=(segment_length, dim))
    return MovingStream(x=x, z=z, y=y, bases=bases, segment_length=segment_length)


@dataclass
class LinearAdapter:
    mean: np.ndarray
    transform_matrix: np.ndarray
    signature: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.transform_matrix.T


def fit_amuse(x: np.ndarray, *, lag: int = 1, eps: float = 1e-6) -> LinearAdapter:
    """One-lag AMUSE: whiten, then diagonalize lagged covariance."""
    mean = x.mean(axis=0)
    xc = x - mean
    c0 = (xc.T @ xc) / len(xc)
    c0 = 0.5 * (c0 + c0.T)
    eigvals, eigvecs = np.linalg.eigh(c0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    whitening = np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
    xw = xc @ whitening.T
    ctau = (xw[lag:].T @ xw[:-lag]) / (len(xw) - lag)
    ctau = 0.5 * (ctau + ctau.T)
    lagvals, lagvecs = np.linalg.eigh(ctau)
    order = np.argsort(lagvals)[::-1]
    lagvals = lagvals[order]
    lagvecs = lagvecs[:, order]
    demixer = lagvecs.T @ whitening
    return LinearAdapter(mean=mean, transform_matrix=demixer, signature=lagvals)


def fit_zero_lag(x: np.ndarray, *, eps: float = 1e-6) -> LinearAdapter:
    """PCA/whitening attacker using only instantaneous covariance."""
    mean = x.mean(axis=0)
    xc = x - mean
    c0 = (xc.T @ xc) / len(xc)
    c0 = 0.5 * (c0 + c0.T)
    eigvals, eigvecs = np.linalg.eigh(c0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    whitening = np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
    return LinearAdapter(mean=mean, transform_matrix=whitening, signature=eigvals)


@dataclass
class RandomFeatureNetwork:
    hidden_matrix: np.ndarray
    hidden_bias: np.ndarray
    readout: np.ndarray

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
        *,
        hidden: int = 128,
        ridge: float = 0.1,
    ) -> "RandomFeatureNetwork":
        dim = x.shape[1]
        hidden_matrix = rng.normal(scale=1.0 / np.sqrt(dim), size=(hidden, dim))
        hidden_bias = rng.uniform(-1.0, 1.0, size=hidden)
        h = np.tanh(x @ hidden_matrix.T + hidden_bias)
        hb = np.column_stack([h, np.ones(len(h))])
        gram = hb.T @ hb + ridge * np.eye(hidden + 1)
        readout = np.linalg.solve(gram, hb.T @ y)
        return cls(hidden_matrix=hidden_matrix, hidden_bias=hidden_bias, readout=readout)

    def features(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.hidden_matrix.T + self.hidden_bias)

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = self.features(x)
        hb = np.column_stack([h, np.ones(len(h))])
        return hb @ self.readout

    def refit_readout(self, x: np.ndarray, y: np.ndarray, *, ridge: float = 1.0) -> None:
        """Strong local/closed-form attacker: retrain only the output on calibration labels."""
        h = self.features(x)
        hb = np.column_stack([h, np.ones(len(h))])
        gram = hb.T @ hb + ridge * np.eye(hb.shape[1])
        self.readout = np.linalg.solve(gram, hb.T @ y)

    def copy(self) -> "RandomFeatureNetwork":
        return RandomFeatureNetwork(
            hidden_matrix=self.hidden_matrix.copy(),
            hidden_bias=self.hidden_bias.copy(),
            readout=self.readout.copy(),
        )


def scalar_sign_calibration(
    separated: np.ndarray,
    calibration_target: np.ndarray,
    network: RandomFeatureNetwork,
) -> Tuple[np.ndarray, float]:
    """Resolve only the global +/- ambiguity left by Gaussian temporal separation.

    The search receives one scalar MSE consequence per candidate sign map. It does
    not receive gradients or a hidden-state target. At dim=4 this is 16 explicit
    hypotheses, intentionally kept small and auditable.
    """
    dim = separated.shape[1]
    best_loss = np.inf
    best_sign = np.ones(dim)
    for bits in product((-1.0, 1.0), repeat=dim):
        sign = np.asarray(bits)
        prediction = network.predict(separated * sign)
        loss = float(np.mean((prediction - calibration_target) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_sign = sign
    return best_sign, best_loss


def classification_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.where(prediction >= 0.0, 1.0, -1.0) == target))


def run_gate0_seed(
    seed: int,
    *,
    separator_window: int = 2000,
    calibration_examples: int = 32,
    hidden: int = 128,
) -> List[Dict[str, float]]:
    stream = make_moving_stream(seed)
    x, z, y = stream.x, stream.z, stream.y
    seglen = stream.segment_length
    rng = np.random.default_rng(100_000 + seed)

    # Learn the nonlinear task once in session 0.
    amuse0 = fit_amuse(x[:separator_window])
    pca0 = fit_zero_lag(x[:separator_window])
    train_slice = slice(separator_window, seglen - 500)

    temporal_net = RandomFeatureNetwork.fit(
        amuse0.transform(x[train_slice]), y[train_slice], rng, hidden=hidden
    )
    pca_net = RandomFeatureNetwork.fit(
        pca0.transform(x[train_slice]), y[train_slice], rng, hidden=hidden
    )
    raw_net = RandomFeatureNetwork.fit(x[train_slice], y[train_slice], rng, hidden=hidden)
    oracle_net = RandomFeatureNetwork.fit(z[train_slice], y[train_slice], rng, hidden=hidden)

    rows: List[Dict[str, float]] = []
    for segment in range(5):
        start = segment * seglen
        stop = (segment + 1) * seglen
        amuse = fit_amuse(x[start : start + separator_window])
        pca = fit_zero_lag(x[start : start + separator_window])
        s = amuse.transform(x[start:stop])
        p = pca.transform(x[start:stop])

        if segment == 0:
            sign_t = np.ones(4)
            sign_p = np.ones(4)
            offset = separator_window
            raw_adapt = raw_net
        else:
            cal0 = separator_window
            cal1 = cal0 + calibration_examples
            sign_t, _ = scalar_sign_calibration(
                s[cal0:cal1], y[start + cal0 : start + cal1], temporal_net
            )
            sign_p, _ = scalar_sign_calibration(
                p[cal0:cal1], y[start + cal0 : start + cal1], pca_net
            )
            raw_adapt = raw_net.copy()
            raw_adapt.refit_readout(
                x[start + cal0 : start + cal1], y[start + cal0 : start + cal1]
            )
            offset = cal1

        target = y[start + offset : stop]
        rows.append(
            {
                "seed": float(seed),
                "segment": float(segment),
                "temporal": classification_accuracy(
                    temporal_net.predict(s[offset:] * sign_t), target
                ),
                "zero_lag": classification_accuracy(
                    pca_net.predict(p[offset:] * sign_p), target
                ),
                "raw_frozen": classification_accuracy(
                    raw_net.predict(x[start + offset : stop]), target
                ),
                "raw_refit_32": classification_accuracy(
                    raw_adapt.predict(x[start + offset : stop]), target
                ),
                "oracle_latent": classification_accuracy(
                    oracle_net.predict(z[start + offset : stop]), target
                ),
            }
        )
    return rows


def summarize_gate0(seeds: int = 20) -> Dict[str, Dict[str, float]]:
    rows = []
    for seed in range(seeds):
        rows.extend(run_gate0_seed(seed))
    moving = [row for row in rows if int(row["segment"]) > 0]
    keys = ["temporal", "zero_lag", "raw_frozen", "raw_refit_32", "oracle_latent"]
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in moving])
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary
