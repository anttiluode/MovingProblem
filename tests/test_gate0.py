import unittest

import numpy as np

from moving_problem import fit_amuse, make_moving_stream, run_gate0_seed


class MovingProblemGate0Tests(unittest.TestCase):
    def test_amuse_whitens_and_returns_finite_operator(self):
        stream = make_moving_stream(1, segments=1)
        adapter = fit_amuse(stream.x[:2000])
        s = adapter.transform(stream.x[:2000])
        cov = np.cov(s, rowvar=False)
        self.assertTrue(np.isfinite(adapter.transform_matrix).all())
        self.assertLess(float(np.max(np.abs(cov - np.eye(4)))), 0.15)

    def test_temporal_adapter_beats_zero_lag_on_moving_sessions(self):
        rows = run_gate0_seed(3)
        later = rows[1:]
        temporal = np.mean([r["temporal"] for r in later])
        zero_lag = np.mean([r["zero_lag"] for r in later])
        self.assertGreater(temporal, zero_lag + 0.12)

    def test_frozen_raw_network_breaks_after_basis_change(self):
        rows = run_gate0_seed(7)
        later = rows[1:]
        raw = np.mean([r["raw_frozen"] for r in later])
        temporal = np.mean([r["temporal"] for r in later])
        self.assertGreater(temporal, raw + 0.20)


if __name__ == "__main__":
    unittest.main()
