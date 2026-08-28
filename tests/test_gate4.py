import unittest

from moving_problem_gate4 import (
    fit_noise_floor_exponent,
    track_closed_loop,
)


class Gate4Tests(unittest.TestCase):
    def test_safe_gap_loop_silently_flips_oriented_sign(self):
        result = track_closed_loop(radius=1.0, turns=1, gap_threshold=0.5)
        self.assertGreater(result.min_gap, 1.9)
        self.assertEqual(result.low_confidence_fraction, 0.0)
        self.assertLess(result.final_alignment, -0.99)

    def test_winding_memory_restores_closed_loop_semantics(self):
        result = track_closed_loop(radius=1.0, turns=1, gap_threshold=0.5)
        self.assertGreater(result.corrected_final_alignment, 0.99)
        self.assertAlmostEqual(abs(result.winding), 1.0, places=3)

    def test_non_enclosing_loop_has_no_sign_holonomy(self):
        result = track_closed_loop(
            radius=1.0,
            offset_a=1.6,
            turns=1,
            gap_threshold=0.5,
        )
        self.assertGreater(result.final_alignment, 0.99)
        self.assertAlmostEqual(result.winding, 0.0, places=3)

    def test_even_winding_returns_original_sign(self):
        result = track_closed_loop(radius=1.0, turns=2, gap_threshold=0.5)
        self.assertGreater(result.final_alignment, 0.99)
        self.assertAlmostEqual(abs(result.winding), 2.0, places=3)

    def test_estimator_gap_at_true_crossing_scales_n_minus_half(self):
        slope, _ = fit_noise_floor_exponent(repeats=1200)
        self.assertGreater(slope, -0.60)
        self.assertLess(slope, -0.40)

    def test_topological_flip_survives_small_operator_noise(self):
        result = track_closed_loop(
            radius=1.0,
            turns=1,
            gap_threshold=0.5,
            operator_noise=0.02,
            seed=4,
        )
        self.assertEqual(result.low_confidence_fraction, 0.0)
        self.assertLess(result.final_alignment, -0.95)
        self.assertGreater(result.corrected_final_alignment, 0.95)


if __name__ == "__main__":
    unittest.main()
