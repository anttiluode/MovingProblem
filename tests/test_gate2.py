import unittest

from moving_problem_gate2 import run_gate2_seed


class Gate2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = run_gate2_seed(
            0,
            burn_steps=20_000,
            drift_steps=30_000,
        )

    def test_geometry_survives_coordinate_drift(self):
        self.assertGreater(self.row["coordinate_error_final"], 0.20)
        self.assertLess(self.row["gram_error_final"], 0.08)
        self.assertGreater(self.row["geometry_similarity_task"], 0.97)

    def test_temporal_tracker_beats_zero_lag_axis_identity(self):
        self.assertGreater(
            self.row["oriented_amuse_track_once"],
            self.row["oriented_pca_track_once"] + 0.20,
        )

    def test_temporal_tracker_beats_frozen_late_decoder(self):
        self.assertGreater(
            self.row["oriented_amuse_track_last_quarter"],
            self.row["oriented_frozen_last_quarter"] + 0.05,
        )

    def test_invariants_cannot_name_oriented_freedom(self):
        self.assertLess(self.row["oriented_invariant_features"], 0.56)

    def test_tracking_spends_labels_once(self):
        self.assertEqual(self.row["labels_track_once"], 16.0)
        self.assertGreater(self.row["labels_recal_each"], self.row["labels_track_once"])


if __name__ == "__main__":
    unittest.main()
