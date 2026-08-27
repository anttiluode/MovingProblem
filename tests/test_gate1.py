import unittest
from moving_problem_gate1 import directional_pair, run_gate1_seed

class Gate1Tests(unittest.TestCase):
    def test_exact_reverse(self):
        up, down = directional_pair(5)
        self.assertLess(abs(up[::-1] - down).max(), 1e-12)

    def test_local_beats_random(self):
        row = run_gate1_seed(0)
        self.assertGreater(row["learned_local"], row["random_hidden"] + 0.15)

    def test_time_shuffle_hurts(self):
        row = run_gate1_seed(1)
        self.assertGreater(
            row["learned_local"],
            row["shuffled_hidden_time"] + 0.10,
        )

    def test_no_eligibility_is_chance(self):
        row = run_gate1_seed(2)
        self.assertAlmostEqual(row["no_eligibility"], 0.5, places=12)

    def test_power_is_blind(self):
        row = run_gate1_seed(3)
        self.assertAlmostEqual(row["power_only"], 0.5, places=12)
        self.assertLess(row["power_pair_max_difference"], 1e-10)

if __name__ == "__main__":
    unittest.main()
