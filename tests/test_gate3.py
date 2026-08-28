import unittest

from moving_problem_gate3 import (
    guarded_continuity_tracker,
    make_stress_world,
    ordered_tracker,
    paired_procrustes,
    run_gate3_seed,
)


class Gate3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = run_gate3_seed(0)

    def test_crossing_guard_beats_ordered_identity(self):
        self.assertGreater(
            self.row["crossing_guarded"],
            self.row["crossing_ordered"] + 0.12,
        )

    def test_crossing_relocks_late(self):
        self.assertGreater(self.row["crossing_guarded_last10"], 0.90)

    def test_exact_degeneracy_remains_a_boundary(self):
        self.assertLess(self.row["degenerate_guarded"], 0.70)

    def test_disappearance_is_unobservable_but_return_recovers(self):
        self.assertLess(self.row["dropout_hidden_interval"], 0.56)
        self.assertGreater(self.row["dropout_after_return"], 0.88)

    def test_nonorthogonal_and_subspace_deformation_survive(self):
        self.assertGreater(self.row["nonorthogonal_guarded"], 0.88)
        self.assertGreater(self.row["deforming_subspace_guarded"], 0.88)

    def test_paired_procrustes_is_a_strong_attacker(self):
        self.assertGreater(self.row["crossing_paired_procrustes"], 0.97)


if __name__ == "__main__":
    unittest.main()
