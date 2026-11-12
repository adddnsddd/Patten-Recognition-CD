"""Checks the reporting denominator and grid re-entry rule."""

import unittest

from experiments.run_suite import _repeat_metrics


class RepeatRateTests(unittest.TestCase):
    def test_reentry_counts_full_repeat_distance_and_ignores_return(self):
        rows = [
            {"x": .65, "y": .5, "phase": "cleaning", "step_distance": .15},
            {"x": .5, "y": .5, "phase": "cleaning", "step_distance": .15},
            {"x": .65, "y": .5, "phase": "return", "step_distance": .15},
        ]
        distance, repeated, rate = _repeat_metrics(rows)
        self.assertAlmostEqual(distance, .3)
        self.assertAlmostEqual(repeated, .1)
        self.assertAlmostEqual(rate, 100 / 3)


if __name__ == "__main__":
    unittest.main()
