"""MovingProblem Gate 2 — one plastic network downstream of another plastic network.

Upstream is a paper-inspired linear Hebbian/anti-Hebbian similarity-matching
network. It learns a principal subspace, then continues updating with synaptic
noise so its coordinates drift inside an equivalence class while population
geometry remains approximately stable.

Downstream compares three strategies:
1. frozen coordinate decoder;
2. geometry-only invariant readout;
3. temporal self-calibration (AMUSE) with one initial sign calibration, then
   sign continuity only.

The deliberate modification from Qin et al. is that the top three latent
directions have equal zero-lag variance but distinct AR(1) temporal signatures.
This removes a cheap PCA identity cue while preserving the paper's rotational
degeneracy idea. The experiment is therefore paper-inspired, not an exact
replication.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

Array = np.ndarray


def _orthogonal(rng: np.random.Generator, dim: int) -> Array:
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _orthogonal_part(a: Array) -> Array:
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    return u @ vt


def _ar_stream(length: int, rhos: Array, rng: np.random.Generator) -> Array:
    rhos = np.asarray(rhos, dtype=float)
    z = np.zeros((length, len(rhos)), dtype=float)
    z[0] = rng.normal(size=len(rhos))
    scale = np.sqrt(1.0 - rhos * rhos)
    for t in range(1, length):
        z[t] = rhos * z[t - 1] + scale * rng.normal(size=len(rhos))
    return z


@dataclass
class DriftRun:
    x: Array
    y: Array
    z: Array
    snapshots: Array
    snapshot_times: Array
    principal_basis: Array


def simulate_similarity_matching_drift(
    seed: int,
    *,
    input_dim: int = 8,
    output_dim: int = 3,
    burn_steps: int = 30_000,
    drift_steps: int = 50_000,
    eta: float = 0.05,
    synaptic_noise: float = 0.015,
    snapshot_every: int = 500,
) -> DriftRun:
    """Paper-inspired local Hebbian/anti-Hebbian drift model."""
    if input_dim <= output_dim:
        raise ValueError("input_dim must exceed output_dim")
    if output_dim != 3:
        raise ValueError("Gate 2 is frozen at output_dim=3")

    rng = np.random.default_rng(seed)
    q = _orthogonal(np.random.default_rng(seed + 101), input_dim)
    u = q[:, :output_dim]
    v = q[:, output_dim:]
    rhos = np.array([0.95, 0.65, 0.25], dtype=float)

    def make_input(length: int, stream_seed: int) -> Tuple[Array, Array]:
        local = np.random.default_rng(stream_seed)
        z = _ar_stream(length, rhos, local)
        nuisance = local.normal(size=(length, input_dim - output_dim))
        x = (
            np.sqrt(2.0) * z @ u.T
            + np.sqrt(0.1) * nuisance @ v.T
        )
        return x, z

    x_burn, _ = make_input(burn_steps, seed + 1_000)
    w = 0.1 * rng.normal(size=(output_dim, input_dim))
    m = np.eye(output_dim)

    def stabilize_lateral(matrix: Array) -> Array:
        matrix = 0.5 * (matrix + matrix.T)
        minimum = float(np.linalg.eigvalsh(matrix).min())
        if minimum < 0.05:
            matrix = matrix + (0.05 - minimum) * np.eye(output_dim)
        return matrix

    for x_t in x_burn:
        y_t = np.linalg.solve(m + 1e-5 * np.eye(output_dim), w @ x_t)
        w += eta * (np.outer(y_t, x_t) - w)
        m += eta * (np.outer(y_t, y_t) - m)
        m = stabilize_lateral(m)

    x, z = make_input(drift_steps, seed + 2_000)
    y = np.zeros((drift_steps, output_dim), dtype=float)
    snapshots: List[Array] = []
    snapshot_times: List[int] = []

    for t, x_t in enumerate(x):
        y_t = np.linalg.solve(m + 1e-5 * np.eye(output_dim), w @ x_t)
        y[t] = y_t

        w += (
            eta * (np.outer(y_t, x_t) - w)
            + np.sqrt(eta) * synaptic_noise * rng.normal(size=w.shape)
        )
        noise_m = np.sqrt(eta) * synaptic_noise * rng.normal(size=m.shape)
        m += eta * (np.outer(y_t, y_t) - m) + 0.5 * (noise_m + noise_m.T)
        m = stabilize_lateral(m)

        if t % snapshot_every == 0:
            f = np.linalg.solve(m + 1e-5 * np.eye(output_dim), w)
            snapshots.append(f.copy())
            snapshot_times.append(t)

    return DriftRun(
        x=x,
        y=y,
        z=z,
        snapshots=np.asarray(snapshots),
        snapshot_times=np.asarray(snapshot_times),
        principal_basis=u,
    )


def fit_amuse(y: Array, lag: int = 1, eps: float = 1e-6) -> Tuple[Array, Array, Array]:
    """Zero-lag whitening plus symmetric lag covariance eigendecomposition."""
    mean = y.mean(axis=0)
    yc = y - mean
    c0 = (yc.T @ yc) / len(yc)
    c0 = 0.5 * (c0 + c0.T)
    values, vectors = np.linalg.eigh(c0)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    whitening = np.diag(1.0 / np.sqrt(values + eps)) @ vectors.T

    yw = yc @ whitening.T
    ct = (yw[lag:].T @ yw[:-lag]) / (len(yw) - lag)
    ct = 0.5 * (ct + ct.T)
    lag_values, lag_vectors = np.linalg.eigh(ct)
    order = np.argsort(lag_values)[::-1]
    lag_values = lag_values[order]
    lag_vectors = lag_vectors[:, order]
    demixer = lag_vectors.T @ whitening
    return mean, demixer, lag_values


def apply_demixer(y: Array, mean: Array, demixer: Array) -> Array:
    return (y - mean) @ demixer.T


def _binary_accuracy(score: Array, target: Array) -> float:
    pred = np.where(np.asarray(score) >= 0.0, 1.0, -1.0)
    return float(np.mean(pred == target))


def _orient_first_axis_from_labels(
    source_estimate: Array,
    target: Array,
    labels: int,
) -> float:
    pred = np.where(source_estimate[:labels, 0] >= 0.0, 1.0, -1.0)
    return 1.0 if float(np.mean(pred == target[:labels])) >= 0.5 else -1.0


def temporal_axis_tracker(
    y: Array,
    target: Array,
    *,
    window: int = 1_000,
    initial_labels: int = 16,
    recalibrate_every_window: bool = False,
) -> Array:
    """Track the highest-autocorrelation latent freedom across drifting windows."""
    previous: Array | None = None
    outputs: List[Array] = []
    n_windows = len(y) // window

    for index in range(n_windows):
        start = index * window
        stop = start + window
        block = y[start:stop]
        block_target = target[start:stop]

        mean, demixer, _ = fit_amuse(block)
        sources = apply_demixer(block, mean, demixer)

        if index == 0 or recalibrate_every_window:
            sign = _orient_first_axis_from_labels(
                sources,
                block_target,
                initial_labels,
            )
            demixer[0] *= sign
            for row in range(1, demixer.shape[0]):
                pivot = int(np.argmax(np.abs(demixer[row])))
                if demixer[row, pivot] < 0.0:
                    demixer[row] *= -1.0
        else:
            signs = np.sign(np.sum(demixer * previous, axis=1))
            signs[signs == 0.0] = 1.0
            demixer = signs[:, None] * demixer

        sources = apply_demixer(block, mean, demixer)
        outputs.append(np.where(sources[:, 0] >= 0.0, 1.0, -1.0))
        previous = demixer.copy()

    return np.concatenate(outputs)


def zero_lag_pca_tracker(
    y: Array,
    target: Array,
    *,
    window: int = 1_000,
    initial_labels: int = 16,
) -> Array:
    """Zero-lag control in a deliberately degenerate covariance world."""
    previous: Array | None = None
    outputs: List[Array] = []
    n_windows = len(y) // window

    for index in range(n_windows):
        start = index * window
        stop = start + window
        block = y[start:stop]
        yc = block - block.mean(axis=0)

        c0 = (yc.T @ yc) / len(yc)
        c0 = 0.5 * (c0 + c0.T)
        values, vectors = np.linalg.eigh(c0)
        vectors = vectors[:, np.argsort(values)[::-1]]
        demixer = vectors.T
        sources = yc @ demixer.T

        if previous is None:
            sign = _orient_first_axis_from_labels(
                sources,
                target[start:stop],
                initial_labels,
            )
            demixer[0] *= sign
            for row in range(1, demixer.shape[0]):
                pivot = int(np.argmax(np.abs(demixer[row])))
                if demixer[row, pivot] < 0.0:
                    demixer[row] *= -1.0
        else:
            signs = np.sign(np.sum(demixer * previous, axis=1))
            signs[signs == 0.0] = 1.0
            demixer = signs[:, None] * demixer

        sources = yc @ demixer.T
        outputs.append(np.where(sources[:, 0] >= 0.0, 1.0, -1.0))
        previous = demixer.copy()

    return np.concatenate(outputs)


def run_gate2_seed(
    seed: int,
    *,
    burn_steps: int = 30_000,
    drift_steps: int = 50_000,
    window: int = 1_000,
    initial_labels: int = 16,
) -> Dict[str, float]:
    run = simulate_similarity_matching_drift(
        seed,
        burn_steps=burn_steps,
        drift_steps=drift_steps,
    )
    y = run.y
    z = run.z
    snapshots = run.snapshots
    u = run.principal_basis

    latent_similarity = np.sum(z[1:] * z[:-1], axis=1)
    latent_threshold = float(np.median(latent_similarity[: window - 1]))
    geometry_target = np.where(latent_similarity >= latent_threshold, 1.0, -1.0)

    output_similarity = np.sum(y[1:] * y[:-1], axis=1)
    output_threshold = float(np.median(output_similarity[: window - 1]))
    geometry_score = output_similarity - output_threshold
    geometry_accuracy = _binary_accuracy(geometry_score, geometry_target)

    oriented_target = np.where(z[:, 0] >= 0.0, 1.0, -1.0)

    design = np.column_stack([y[:window], np.ones(window)])
    coefficient = np.linalg.solve(
        design.T @ design + 1e-3 * np.eye(design.shape[1]),
        design.T @ z[:window, 0],
    )
    frozen_score = np.column_stack([y, np.ones(len(y))]) @ coefficient
    frozen_prediction = np.where(frozen_score >= 0.0, 1.0, -1.0)

    tracked_prediction = temporal_axis_tracker(
        y,
        oriented_target,
        window=window,
        initial_labels=initial_labels,
        recalibrate_every_window=False,
    )
    recalibrated_prediction = temporal_axis_tracker(
        y,
        oriented_target,
        window=window,
        initial_labels=initial_labels,
        recalibrate_every_window=True,
    )
    pca_prediction = zero_lag_pca_tracker(
        y,
        oriented_target,
        window=window,
        initial_labels=initial_labels,
    )

    invariant = np.column_stack(
        [
            np.sum(y * y, axis=1),
            np.r_[0.0, np.sum(y[1:] * y[:-1], axis=1)],
            np.r_[0.0, 0.0, np.sum(y[2:] * y[:-2], axis=1)],
        ]
    )
    inv_design = np.column_stack([invariant[:window], np.ones(window)])
    inv_coef = np.linalg.solve(
        inv_design.T @ inv_design + 1e-2 * np.eye(inv_design.shape[1]),
        inv_design.T @ z[:window, 0],
    )
    inv_score = np.column_stack([invariant, np.ones(len(invariant))]) @ inv_coef

    rng = np.random.default_rng(seed + 555)
    latent_probe = rng.normal(size=(200, 3))
    input_probe = latent_probe @ u.T
    y0 = input_probe @ snapshots[0].T
    y_last = input_probe @ snapshots[-1].T

    gram0 = y0 @ y0.T
    gram_last = y_last @ y_last.T
    gram_error = float(
        np.linalg.norm(gram_last - gram0, ord="fro")
        / (np.linalg.norm(gram0, ord="fro") + 1e-12)
    )
    coordinate_error = float(
        np.linalg.norm(y_last - y0, ord="fro")
        / (np.linalg.norm(y0, ord="fro") + 1e-12)
    )

    projector = u @ u.T
    psp_errors = [
        np.linalg.norm(f.T @ f - projector, ord="fro")
        / (np.linalg.norm(projector, ord="fro") + 1e-12)
        for f in snapshots
    ]

    start_rotation = _orthogonal_part(snapshots[0] @ u)
    last_rotation = _orthogonal_part(snapshots[-1] @ u)
    rotation_drift = float(np.linalg.norm(last_rotation - start_rotation, ord="fro"))

    usable = (len(y) // window) * window
    after_initial = slice(window, usable)
    last_quarter = slice(int(0.75 * usable), usable)
    n_windows = usable // window

    return {
        "seed": float(seed),
        "geometry_similarity_task": geometry_accuracy,
        "oriented_frozen_after_initial": float(
            np.mean(frozen_prediction[after_initial] == oriented_target[after_initial])
        ),
        "oriented_frozen_last_quarter": float(
            np.mean(frozen_prediction[last_quarter] == oriented_target[last_quarter])
        ),
        "oriented_amuse_track_once": float(
            np.mean(tracked_prediction[after_initial] == oriented_target[after_initial])
        ),
        "oriented_amuse_track_last_quarter": float(
            np.mean(tracked_prediction[last_quarter] == oriented_target[last_quarter])
        ),
        "oriented_amuse_recal_each": float(
            np.mean(recalibrated_prediction[after_initial] == oriented_target[after_initial])
        ),
        "oriented_pca_track_once": float(
            np.mean(pca_prediction[after_initial] == oriented_target[after_initial])
        ),
        "oriented_invariant_features": _binary_accuracy(
            inv_score[window:usable],
            oriented_target[window:usable],
        ),
        "labels_track_once": float(initial_labels),
        "labels_recal_each": float(initial_labels * n_windows),
        "gram_error_final": gram_error,
        "coordinate_error_final": coordinate_error,
        "psp_error_mean": float(np.mean(psp_errors)),
        "rotation_drift_final": rotation_drift,
    }


def summarize_gate2(seeds: int = 12) -> Dict[str, Dict[str, float]]:
    rows = [run_gate2_seed(seed) for seed in range(seeds)]
    summary: Dict[str, Dict[str, float]] = {}
    for key in rows[0]:
        if key == "seed":
            continue
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return summary
