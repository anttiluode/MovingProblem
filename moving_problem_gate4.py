"""MovingProblem Gate 4 — topology can defeat a gap-only semantic tracker.

A real symmetric two-mode identifying operator has local normal form

    L(a,b) = [[a, b],
              [b,-a]]

with eigenvalue gap 2*sqrt(a^2+b^2).  The degeneracy is at (a,b)=(0,0).

Gate 3 guarded semantic identity by watching the local gap and refusing to
update through low-identifiability windows.  Gate 4 attacks that rule with a
closed loop around the degeneracy at constant, perfectly safe gap.

For a loop that winds once around the origin, a continuously oriented real
eigenvector returns with the opposite sign.  The eigenLINE is unchanged, but
an oriented semantic readout has acquired a Z2 holonomy.  A local gap monitor
cannot see it.

This file also measures the finite-sample confound at an exact crossing:
off-diagonal estimator noise turns the population degeneracy into an apparent
avoided crossing whose gap scales as N^{-1/2}.

No quantum dynamics are simulated.  The same 2x2 normal form appears in the
Landau-Zener problem, but the Landau-Zener transition probability is not a
prediction for this tracker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

Array = np.ndarray


def two_mode_operator(a: float, b: float) -> Array:
    return np.array([[a, b], [b, -a]], dtype=float)


def principal_eigenvector(operator: Array) -> Tuple[float, Array]:
    values, vectors = np.linalg.eigh(operator)
    index = int(np.argmax(values))
    return float(values[index]), vectors[:, index].copy()


def eigengap(operator: Array) -> float:
    values = np.linalg.eigvalsh(operator)
    return float(values[-1] - values[-2])


@dataclass
class LoopResult:
    final_alignment: float
    corrected_final_alignment: float
    min_gap: float
    low_confidence_fraction: float
    winding: float
    sign_holonomy: int


def _unwrap_increment(delta: float) -> float:
    return float((delta + np.pi) % (2.0 * np.pi) - np.pi)


def track_closed_loop(
    *,
    radius: float = 1.0,
    offset_a: float = 0.0,
    offset_b: float = 0.0,
    turns: int = 1,
    steps_per_turn: int = 720,
    gap_threshold: float = 0.25,
    operator_noise: float = 0.0,
    seed: int = 0,
) -> LoopResult:
    """Track one eigenline continuously around a closed operator path.

    The Gate-3-style tracker:
      * estimates the principal eigenvector;
      * flips its sign to maximize continuity with the previous vector;
      * marks low confidence only if the local eigengap is small.

    The winding-aware semantic readout stores one additional global bit:
    parity of winding around the known degeneracy.  It does not alter the
    locally continuous eigenvector; it only corrects the final oriented
    semantic interpretation.

    This exposes the distinction between local frame continuity and global
    semantic holonomy.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if turns < 1:
        raise ValueError("turns must be >= 1")

    rng = np.random.default_rng(seed)
    count = turns * steps_per_turn + 1
    phase = np.linspace(0.0, 2.0 * np.pi * turns, count)

    previous: Array | None = None
    initial: Array | None = None
    gaps = []
    low = []

    cumulative_angle = 0.0
    previous_param_angle: float | None = None

    for phi in phase:
        a = offset_a + radius * np.cos(phi)
        b = offset_b + radius * np.sin(phi)
        operator = two_mode_operator(a, b)

        if operator_noise > 0.0:
            noise = rng.normal(scale=operator_noise, size=(2, 2))
            noise = 0.5 * (noise + noise.T)
            operator = operator + noise

        gap = eigengap(operator)
        _, vector = principal_eigenvector(operator)

        if previous is not None and float(vector @ previous) < 0.0:
            vector *= -1.0

        if initial is None:
            initial = vector.copy()

        gaps.append(gap)
        low.append(gap < gap_threshold)
        previous = vector.copy()

        param_angle = float(np.arctan2(b, a))
        if previous_param_angle is not None:
            cumulative_angle += _unwrap_increment(param_angle - previous_param_angle)
        previous_param_angle = param_angle

    assert initial is not None and previous is not None

    winding = cumulative_angle / (2.0 * np.pi)
    winding_integer = int(np.rint(winding))

    raw_alignment = float(previous @ initial)

    # A loop around a real conical degeneracy contributes a Z2 sign holonomy.
    # Semantic orientation can preserve its original branch only if the system
    # remembers the winding parity (or receives new external grounding).
    holonomy_sign = -1 if abs(winding_integer) % 2 == 1 else 1
    corrected_alignment = float(holonomy_sign * raw_alignment)

    return LoopResult(
        final_alignment=raw_alignment,
        corrected_final_alignment=corrected_alignment,
        min_gap=float(np.min(gaps)),
        low_confidence_fraction=float(np.mean(low)),
        winding=float(winding),
        sign_holonomy=holonomy_sign,
    )


def apparent_gap_at_true_crossing(
    n_samples: int,
    *,
    sigma: float = 1.0,
    repeats: int = 2_000,
    seed: int = 0,
) -> Array:
    """Finite-sample estimate of the gap at a population-level exact crossing.

    The population operator is exactly zero in its traceless 2x2 block.
    A covariance / lag-covariance estimator has O(N^{-1/2}) entry noise.
    We model that directly with a symmetric Gaussian perturbation.

    The observed matrix is almost surely nondegenerate even though the
    population operator is exactly degenerate.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    scale = sigma / np.sqrt(float(n_samples))

    gaps = np.empty(repeats, dtype=float)
    for i in range(repeats):
        n11, n22, n12 = rng.normal(scale=scale, size=3)
        estimate = np.array([[n11, n12], [n12, n22]], dtype=float)
        gaps[i] = eigengap(estimate)
    return gaps


def fit_noise_floor_exponent(
    sample_sizes: Iterable[int] = (128, 256, 512, 1024, 2048, 4096),
    *,
    sigma: float = 1.0,
    repeats: int = 2_000,
    seed: int = 0,
) -> Tuple[float, Dict[int, float]]:
    means: Dict[int, float] = {}
    for index, n in enumerate(sample_sizes):
        gaps = apparent_gap_at_true_crossing(
            int(n),
            sigma=sigma,
            repeats=repeats,
            seed=seed + index * 97,
        )
        means[int(n)] = float(np.mean(gaps))

    x = np.log(np.asarray(list(means.keys()), dtype=float))
    y = np.log(np.asarray(list(means.values()), dtype=float))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope), means


def noisy_avoided_crossing_gap(
    delta: float,
    n_samples: int,
    *,
    sigma: float = 1.0,
    repeats: int = 2_000,
    seed: int = 0,
) -> float:
    """Measured minimum gap at t=0 for L=[[0,delta],[delta,0]] plus estimator noise."""
    rng = np.random.default_rng(seed)
    scale = sigma / np.sqrt(float(n_samples))
    gaps = np.empty(repeats, dtype=float)

    population = two_mode_operator(0.0, float(delta))
    for i in range(repeats):
        n11, n22, n12 = rng.normal(scale=scale, size=3)
        noise = np.array([[n11, n12], [n12, n22]], dtype=float)
        gaps[i] = eigengap(population + noise)
    return float(np.mean(gaps))


def run_gate4(seed: int = 0) -> Dict[str, float]:
    enclosing = track_closed_loop(
        radius=1.0,
        turns=1,
        gap_threshold=0.5,
        seed=seed,
    )
    non_enclosing = track_closed_loop(
        radius=1.0,
        offset_a=1.6,
        turns=1,
        gap_threshold=0.5,
        seed=seed,
    )
    double_loop = track_closed_loop(
        radius=1.0,
        turns=2,
        gap_threshold=0.5,
        seed=seed,
    )

    noisy_enclosing = track_closed_loop(
        radius=1.0,
        turns=1,
        gap_threshold=0.5,
        operator_noise=0.02,
        seed=seed,
    )

    slope, means = fit_noise_floor_exponent(seed=seed)

    n_reference = 1024
    noise_scale = 1.0 / np.sqrt(n_reference)
    tiny_delta = 0.25 * noise_scale
    large_delta = 4.0 * noise_scale

    tiny_gap = noisy_avoided_crossing_gap(
        tiny_delta,
        n_reference,
        seed=seed + 10_000,
    )
    large_gap = noisy_avoided_crossing_gap(
        large_delta,
        n_reference,
        seed=seed + 20_000,
    )

    return {
        "enclosing_guard_final_alignment": enclosing.final_alignment,
        "enclosing_winding_corrected_alignment": enclosing.corrected_final_alignment,
        "enclosing_min_gap": enclosing.min_gap,
        "enclosing_low_confidence_fraction": enclosing.low_confidence_fraction,
        "enclosing_winding": enclosing.winding,
        "non_enclosing_guard_final_alignment": non_enclosing.final_alignment,
        "non_enclosing_winding": non_enclosing.winding,
        "double_loop_guard_final_alignment": double_loop.final_alignment,
        "double_loop_winding": double_loop.winding,
        "noisy_enclosing_guard_final_alignment": noisy_enclosing.final_alignment,
        "noisy_enclosing_winding_corrected_alignment": noisy_enclosing.corrected_final_alignment,
        "noise_floor_loglog_slope": slope,
        "gap_mean_n128": means[128],
        "gap_mean_n4096": means[4096],
        "tiny_delta_over_noise": tiny_delta / noise_scale,
        "tiny_delta_measured_gap_over_noise": tiny_gap / noise_scale,
        "large_delta_over_noise": large_delta / noise_scale,
        "large_delta_measured_gap_over_noise": large_gap / noise_scale,
    }


def summarize_gate4(seeds: int = 16) -> Dict[str, Dict[str, float]]:
    rows = [run_gate4(seed) for seed in range(seeds)]
    summary: Dict[str, Dict[str, float]] = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return summary
