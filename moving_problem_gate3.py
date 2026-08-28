"""MovingProblem Gate 3 — stress the continuous-lock idea.

Gate 2 established that a drifting upstream representation can preserve geometry
while a temporal tracker maintains one oriented freedom with one initial label
budget.

Gate 3 isolates the downstream tracking problem and attacks it under:
- crowded temporal signatures;
- signatures that cross;
- sustained exact degeneracy;
- source disappearance and reappearance;
- non-orthogonal mixing;
- useful-subspace deformation in a larger ambient space.

The new mechanism is deliberately tiny: when the current operator is
temporarily non-identifying, preserve the last trusted identity map instead of
updating through the ambiguity.

This is a tracking heuristic, not a novelty claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Tuple

import numpy as np

Array = np.ndarray


def _orthogonal(rng: np.random.Generator, dim: int) -> Array:
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _rotation3(axis: Array, angle: float) -> Array:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis) + 1e-12
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    eye = np.eye(3)
    return eye + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


@dataclass
class StressWorld:
    y: Array
    z: Array
    signatures: Array
    amplitudes: Array
    mixing: List[Array]
    window: int


def make_stress_world(
    seed: int,
    scenario: str,
    *,
    windows: int = 40,
    window: int = 1_000,
) -> StressWorld:
    rng = np.random.default_rng(seed)
    total = windows * window

    rho = np.zeros((windows, 3), dtype=float)
    amp = np.ones((windows, 3), dtype=float)
    obs_dim = 3

    if scenario == "separated":
        rho[:] = [0.95, 0.65, 0.25]
    elif scenario == "crowded":
        rho[:] = [0.95, 0.91, 0.87]
    elif scenario == "degenerate":
        rho[:] = [0.95, 0.95, 0.25]
    elif scenario == "crossing":
        s = np.linspace(0.0, 1.0, windows)
        rho[:, 0] = 0.95 - 0.40 * s
        rho[:, 1] = 0.55 + 0.40 * s
        rho[:, 2] = 0.25
    elif scenario == "dropout":
        rho[:] = [0.95, 0.65, 0.25]
        amp[12, 0] = 0.66
        amp[13, 0] = 0.33
        amp[14:26, 0] = 0.0
        amp[26, 0] = 0.33
        amp[27, 0] = 0.66
    elif scenario == "nonorthogonal":
        rho[:] = [0.95, 0.65, 0.25]
    elif scenario == "deforming_subspace":
        rho[:] = [0.95, 0.65, 0.25]
        obs_dim = 5
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    z = np.zeros((total, 3), dtype=float)
    z[0] = rng.normal(size=3)
    for t in range(1, total):
        w = min(t // window, windows - 1)
        r = rho[w]
        z[t] = (
            r * z[t - 1]
            + np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0)) * rng.normal(size=3)
        )

    y = np.zeros((total, obs_dim), dtype=float)
    mixing: List[Array] = []

    if obs_dim == 3:
        base = _orthogonal(np.random.default_rng(seed + 100), 3)
        axis = np.random.default_rng(seed + 101).normal(size=3)

        for w in range(windows):
            rotation = _rotation3(axis, 0.035 * w) @ base
            scale = np.ones(3)
            if scenario == "nonorthogonal":
                scale = np.array(
                    [
                        1.0 + 0.50 * np.sin(0.17 * w),
                        1.8 + 0.40 * np.cos(0.11 * w),
                        0.55 + 0.15 * np.sin(0.13 * w),
                    ]
                )
            b = rotation @ np.diag(scale)
            mixing.append(b)
            sl = slice(w * window, (w + 1) * window)
            y[sl] = (
                (z[sl] * amp[w]) @ b.T
                + 0.02 * rng.normal(size=(window, 3))
            )
    else:
        q = _orthogonal(np.random.default_rng(seed + 100), 5)
        useful = q[:, :3]
        spare = q[:, 3]
        axis = np.random.default_rng(seed + 101).normal(size=3)

        for w in range(windows):
            theta = 0.65 * w / max(windows - 1, 1)
            first = np.cos(theta) * useful[:, 0] + np.sin(theta) * spare
            basis = np.column_stack([first, useful[:, 1], useful[:, 2]])
            b = basis @ _rotation3(axis, 0.03 * w)
            mixing.append(b)
            sl = slice(w * window, (w + 1) * window)
            y[sl] = (
                (z[sl] * amp[w]) @ b.T
                + 0.04 * rng.normal(size=(window, 5))
            )

    return StressWorld(
        y=y,
        z=z,
        signatures=rho,
        amplitudes=amp,
        mixing=mixing,
        window=window,
    )


def fit_amuse(block: Array, *, components: int = 3, lag: int = 1) -> Tuple[Array, Array, Array]:
    mean = block.mean(axis=0)
    xc = block - mean

    c0 = (xc.T @ xc) / len(xc)
    c0 = 0.5 * (c0 + c0.T)
    values, vectors = np.linalg.eigh(c0)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    whitening = np.diag(1.0 / np.sqrt(values + 1e-6)) @ vectors.T

    xw = xc @ whitening.T
    ct = (xw[lag:].T @ xw[:-lag]) / (len(xw) - lag)
    ct = 0.5 * (ct + ct.T)
    lag_values, lag_vectors = np.linalg.eigh(ct)
    order = np.argsort(lag_values)[::-1]
    lag_values = lag_values[order]
    lag_vectors = lag_vectors[:, order]
    demixer = lag_vectors.T @ whitening

    return mean, demixer[:components], lag_values[:components]


def _normalized_rows(a: Array) -> Array:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def _align_joint(
    current: Array,
    current_signature: Array,
    reference: Array,
    reference_signature: Array,
) -> Tuple[Array, Array, Array]:
    """Small-k assignment using both coordinate continuity and signature continuity."""
    cur = _normalized_rows(current)
    ref = _normalized_rows(reference)
    k = len(cur)

    best_score = -np.inf
    best_perm = tuple(range(k))
    for perm in permutations(range(k)):
        score = 0.0
        for i in range(k):
            row_score = abs(float(cur[perm[i]] @ ref[i]))
            signature_score = np.exp(
                -abs(float(current_signature[perm[i]] - reference_signature[i])) / 0.08
            )
            score += row_score + signature_score
        if score > best_score:
            best_score = score
            best_perm = perm

    aligned = current[list(best_perm)].copy()
    signature = current_signature[list(best_perm)].copy()
    similarities = np.zeros(k, dtype=float)

    for i in range(k):
        row = aligned[i] / (np.linalg.norm(aligned[i]) + 1e-12)
        similarity = float(row @ ref[i])
        if similarity < 0.0:
            aligned[i] *= -1.0
            similarity = -similarity
        similarities[i] = similarity

    return aligned, signature, similarities


def _initial_orientation(
    block: Array,
    mean: Array,
    demixer: Array,
    target: Array,
    labels: int,
) -> Array:
    sources = (block - mean) @ demixer.T
    predicted = sources[:labels, 0] >= 0.0
    truth = target[:labels] >= 0.0
    if float(np.mean(predicted == truth)) < 0.5:
        demixer[0] *= -1.0

    for row in range(1, len(demixer)):
        pivot = int(np.argmax(np.abs(demixer[row])))
        if demixer[row, pivot] < 0.0:
            demixer[row] *= -1.0
    return demixer


def ordered_tracker(world: StressWorld, *, labels: int = 16) -> Array:
    """Gate-2 style tracker: trust lag-eigenvalue order, preserve signs only."""
    y, z, window = world.y, world.z, world.window
    previous: Array | None = None
    output: List[Array] = []

    for w in range(len(y) // window):
        sl = slice(w * window, (w + 1) * window)
        mean, demixer, _ = fit_amuse(y[sl])

        if previous is None:
            demixer = _initial_orientation(
                y[sl], mean, demixer, z[sl, 0], labels
            )
        else:
            for row in range(len(demixer)):
                if float(demixer[row] @ previous[row]) < 0.0:
                    demixer[row] *= -1.0

        sources = (y[sl] - mean) @ demixer.T
        output.append(np.where(sources[:, 0] >= 0.0, 1.0, -1.0))
        previous = demixer.copy()

    return np.concatenate(output)


def guarded_continuity_tracker(
    world: StressWorld,
    *,
    labels: int = 16,
    gap_ratio: float = 0.45,
    alignment_floor: float = 0.75,
) -> Tuple[Array, Array]:
    """Track identity, but refuse to rewrite it through transient ambiguity.

    Confidence is relative rather than absolute. Persistent crowding is allowed,
    but a sudden collapse of the signature gap or row continuity causes the
    tracker to hold its last trusted demixer for that window.
    """
    y, z, window = world.y, world.z, world.window

    trusted: Array | None = None
    trusted_signature: Array | None = None
    gap_baseline: float | None = None
    output: List[Array] = []
    low_confidence: List[bool] = []

    for w in range(len(y) // window):
        sl = slice(w * window, (w + 1) * window)
        mean, demixer, signature = fit_amuse(y[sl])
        sorted_signature = np.sort(signature)[::-1]
        gap = float(np.min(np.abs(np.diff(sorted_signature))))

        if trusted is None:
            demixer = _initial_orientation(
                y[sl], mean, demixer, z[sl, 0], labels
            )
            use = demixer
            trusted = demixer.copy()
            trusted_signature = signature.copy()
            gap_baseline = gap
            low = False
        else:
            aligned, aligned_signature, similarities = _align_joint(
                demixer,
                signature,
                trusted,
                trusted_signature,
            )

            low = (
                gap < gap_ratio * max(float(gap_baseline), 1e-6)
                or similarities[0] < alignment_floor
            )

            if low:
                # Critical rule: do not let an unidentifiable window rewrite
                # semantic identity. Coast briefly with the last trusted map.
                use = trusted
            else:
                use = aligned
                trusted = aligned.copy()
                trusted_signature = aligned_signature.copy()
                gap_baseline = 0.95 * float(gap_baseline) + 0.05 * gap

        sources = (y[sl] - mean) @ use.T
        output.append(np.where(sources[:, 0] >= 0.0, 1.0, -1.0))
        low_confidence.append(low)

    return np.concatenate(output), np.asarray(low_confidence, dtype=bool)


def recalibrate_each_window(
    world: StressWorld,
    *,
    labels: int = 16,
) -> Array:
    """Strong task-feedback baseline: choose axis and sign anew every window."""
    y, z, window = world.y, world.z, world.window
    output: List[Array] = []

    for w in range(len(y) // window):
        sl = slice(w * window, (w + 1) * window)
        mean, demixer, _ = fit_amuse(y[sl])
        sources = (y[sl] - mean) @ demixer.T

        truth = z[sl, 0][:labels] >= 0.0
        best_accuracy = -1.0
        best_axis = 0
        best_sign = 1.0

        for axis in range(sources.shape[1]):
            for sign in (-1.0, 1.0):
                prediction = sign * sources[:labels, axis] >= 0.0
                accuracy = float(np.mean(prediction == truth))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_axis = axis
                    best_sign = sign

        score = best_sign * sources[:, best_axis]
        output.append(np.where(score >= 0.0, 1.0, -1.0))

    return np.concatenate(output)


def ea_style_recenter(world: StressWorld, *, labels: int = 16) -> Array:
    """Euclidean-alignment-style covariance recentering plus one frozen readout.

    Each window is centered and symmetrically whitened from its zero-lag
    covariance. No temporal information or paired anchors are used.
    """
    y, z, window = world.y, world.z, world.window
    output: List[Array] = []
    coefficient: Array | None = None

    for w in range(len(y) // window):
        sl = slice(w * window, (w + 1) * window)
        block = y[sl]
        mean = block.mean(axis=0)
        xc = block - mean

        cov = (xc.T @ xc) / len(xc)
        cov = 0.5 * (cov + cov.T)
        values, vectors = np.linalg.eigh(cov)
        whitening = (
            vectors
            @ np.diag(1.0 / np.sqrt(values + 1e-6))
            @ vectors.T
        )
        q = xc @ whitening

        if coefficient is None:
            design = np.column_stack([q[:labels], np.ones(labels)])
            target = np.where(z[sl, 0][:labels] >= 0.0, 1.0, -1.0)
            coefficient = np.linalg.solve(
                design.T @ design + 0.1 * np.eye(design.shape[1]),
                design.T @ target,
            )

        score = np.column_stack([q, np.ones(window)]) @ coefficient
        output.append(np.where(score >= 0.0, 1.0, -1.0))

    return np.concatenate(output)


def paired_procrustes(world: StressWorld, *, anchors: int = 16, seed: int = 0) -> Array:
    """Strong attacker when repeated paired anchor stimuli are available.

    A frozen decoder is trained in the first coordinate system. Every later
    window receives 16 matched latent anchor probes in both the reference and
    current coordinate system, then ordinary orthogonal Procrustes aligns it.

    This baseline intentionally has information MovingProblem normally does not:
    explicit cross-window correspondences.
    """
    if world.y.shape[1] != 3:
        raise ValueError("paired_procrustes is used only in the 3-D stress worlds")

    y, z, window = world.y, world.z, world.window
    design = np.column_stack([y[:window], np.ones(window)])
    coefficient = np.linalg.solve(
        design.T @ design + 1e-3 * np.eye(design.shape[1]),
        design.T @ z[:window, 0],
    )

    anchor_latent = np.random.default_rng(seed + 9_999).normal(size=(anchors, 3))
    reference = anchor_latent @ world.mixing[0].T

    output: List[Array] = []
    for w, mixing in enumerate(world.mixing):
        sl = slice(w * window, (w + 1) * window)
        current = anchor_latent @ mixing.T

        u, _, vt = np.linalg.svd(current.T @ reference)
        rotation = u @ vt
        aligned = y[sl] @ rotation

        score = np.column_stack([aligned, np.ones(window)]) @ coefficient
        output.append(np.where(score >= 0.0, 1.0, -1.0))

    return np.concatenate(output)


def _accuracy(prediction: Array, z0: Array, start: int = 0, stop: int | None = None) -> float:
    target = np.where(z0 >= 0.0, 1.0, -1.0)
    return float(np.mean(prediction[start:stop] == target[start:stop]))


def run_gate3_seed(seed: int) -> Dict[str, float]:
    rows: Dict[str, float] = {}

    separated = make_stress_world(seed, "separated")
    guarded, low = guarded_continuity_tracker(separated)
    ea = ea_style_recenter(separated)
    start = separated.window
    rows["separated_guarded"] = _accuracy(guarded, separated.z[:, 0], start)
    rows["separated_ea_recenter"] = _accuracy(ea, separated.z[:, 0], start)

    crowded = make_stress_world(seed, "crowded")
    guarded, low = guarded_continuity_tracker(crowded)
    rows["crowded_guarded"] = _accuracy(guarded, crowded.z[:, 0], crowded.window)
    rows["crowded_low_confidence_fraction"] = float(np.mean(low))

    crossing = make_stress_world(seed, "crossing")
    ordered = ordered_tracker(crossing)
    guarded, low = guarded_continuity_tracker(crossing)
    reacquired = recalibrate_each_window(crossing)
    proc = paired_procrustes(crossing, seed=seed)
    last = len(crossing.y) - 10 * crossing.window
    rows["crossing_ordered"] = _accuracy(ordered, crossing.z[:, 0], crossing.window)
    rows["crossing_guarded"] = _accuracy(guarded, crossing.z[:, 0], crossing.window)
    rows["crossing_guarded_last10"] = _accuracy(guarded, crossing.z[:, 0], last)
    rows["crossing_recalibrate_each"] = _accuracy(
        reacquired, crossing.z[:, 0], crossing.window
    )
    rows["crossing_paired_procrustes"] = _accuracy(
        proc, crossing.z[:, 0], crossing.window
    )
    rows["crossing_low_confidence_fraction"] = float(np.mean(low))

    degenerate = make_stress_world(seed, "degenerate")
    guarded, low = guarded_continuity_tracker(degenerate)
    rows["degenerate_guarded"] = _accuracy(
        guarded, degenerate.z[:, 0], degenerate.window
    )
    rows["degenerate_guarded_last10"] = _accuracy(
        guarded, degenerate.z[:, 0], len(degenerate.y) - 10 * degenerate.window
    )

    dropout = make_stress_world(seed, "dropout")
    guarded, low = guarded_continuity_tracker(dropout)
    rows["dropout_hidden_interval"] = _accuracy(
        guarded,
        dropout.z[:, 0],
        16 * dropout.window,
        24 * dropout.window,
    )
    rows["dropout_after_return"] = _accuracy(
        guarded,
        dropout.z[:, 0],
        28 * dropout.window,
    )

    nonorth = make_stress_world(seed, "nonorthogonal")
    guarded, _ = guarded_continuity_tracker(nonorth)
    rows["nonorthogonal_guarded"] = _accuracy(
        guarded, nonorth.z[:, 0], nonorth.window
    )

    deform = make_stress_world(seed, "deforming_subspace")
    guarded, _ = guarded_continuity_tracker(deform)
    rows["deforming_subspace_guarded"] = _accuracy(
        guarded, deform.z[:, 0], deform.window
    )

    rows["labels_guarded_once"] = 16.0
    rows["labels_recalibrate_each"] = float(
        16 * (len(crossing.y) // crossing.window)
    )
    rows["paired_anchors_per_window"] = 16.0

    return rows


def summarize_gate3(seeds: int = 8) -> Dict[str, Dict[str, float]]:
    runs = [run_gate3_seed(seed) for seed in range(seeds)]
    result: Dict[str, Dict[str, float]] = {}

    for key in runs[0]:
        values = np.asarray([row[key] for row in runs], dtype=float)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    return result
