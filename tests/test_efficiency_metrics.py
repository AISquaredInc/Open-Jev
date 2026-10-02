"""Decision/threshold regressions that probability error alone cannot catch."""
import copy
import math
import unittest

from scripts.benchmark_efficiency import compare_responses, percentile, summarize


def choice(a=.6, b=.4):
    return {"answers": {"route": {"type": "choice", "choice": "a" if a >= b else "b",
                                  "probabilities": {"a": a, "b": b}}},
            "usage": {"input_tokens": 100, "output_tokens": 0}}


class EfficiencyMetricsTest(unittest.TestCase):
    def test_small_error_without_crossing_passes(self):
        check = compare_responses(choice(), choice(.60001, .39999))
        self.assertTrue(check["passed"])
        self.assertAlmostEqual(check["max_probability_error"], .00001)
        self.assertAlmostEqual(check["boundary_margins"][0]["top_two_margin"], .2)

    def test_tiny_probability_error_still_fails_decision_boundary(self):
        check = compare_responses(choice(.500001, .499999), choice(.499999, .500001))
        self.assertFalse(check["passed"])
        self.assertLess(check["max_probability_error"], 1e-4)
        self.assertEqual(check["changed_decisions"], ["route"])

    def test_tiny_coverage_boundary_flip_is_reported(self):
        check = compare_responses(choice(.899999, .100001), choice(.900001, .099999))
        self.assertFalse(check["passed"])
        self.assertEqual(check["changed_decisions"], [])
        self.assertEqual(check["threshold_flips"]["0.9"]["confidence_coverage"], 1)

    def test_noul_equal_boundary_uses_true_at_half(self):
        a = {"answers": {"gate": {"type": "noul", "noul": .5}}}
        b = {"answers": {"gate": {"type": "noul", "noul": .499999}}}
        check = compare_responses(a, b)
        self.assertEqual(check["changed_decisions"], ["gate"])
        self.assertEqual(check["threshold_flips"]["0.5"]["noul_true"], 1)

    def test_missing_reordered_and_nonfinite_probabilities_fail(self):
        b = copy.deepcopy(choice())
        b["answers"]["route"]["probabilities"] = {"b": .4, "a": .6}
        with self.assertRaisesRegex(ValueError, "candidate coverage"):
            compare_responses(choice(), b)
        b = copy.deepcopy(choice())
        b["answers"]["route"]["probabilities"]["a"] = math.nan
        with self.assertRaises(ValueError):
            compare_responses(choice(), b)
        with self.assertRaisesRegex(ValueError, "question coverage"):
            compare_responses(choice(), {"answers": {}})

    def test_usage_mismatch_fails_parity(self):
        b = choice()
        b["usage"]["input_tokens"] += 1
        self.assertFalse(compare_responses(choice(), b)["passed"])

    def test_interpolated_percentiles_and_throughput(self):
        self.assertEqual(percentile([1, 2, 3, 4], .5), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], .95), 3.85)
        metrics = summarize([100, 200], candidates=8, questions=2)
        self.assertAlmostEqual(metrics["requests_per_second"], 2 / .3)
        self.assertAlmostEqual(metrics["candidate_sequences_per_second"], 16 / .3)
