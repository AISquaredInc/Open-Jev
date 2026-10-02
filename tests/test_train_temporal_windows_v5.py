import unittest

from scripts.train_temporal_windows_v5 import four_combinations, noul_status, paired_decisions


def row(key, logits, target):
    return {"id": key, "kind": "noul", "family": "timeline", "target": target,
            "logits": logits, "wall_seconds": 0.0}


class TemporalPilotTest(unittest.TestCase):
    def test_temperature_ablation_keeps_argmax_but_changes_noul_threshold_behavior(self):
        before = [row("a", [0, 1], [0.0, 1.0]), row("b", [1, 0], [1.0, 0.0])]
        after = [row("a", [0, 2], [0.0, 1.0]), row("b", [2, 0], [1.0, 0.0])]
        combinations = four_combinations(before, after, 1.0, 0.5)
        self.assertEqual(len(combinations), 4)
        self.assertTrue(all(value["accuracy"] == 1.0 for value in combinations.values()))
        baseline = combinations["baseline_logits_at_released_temperature"]
        calibrated = combinations["baseline_logits_at_trained_temperature"]
        self.assertEqual(baseline["noul_thresholds_0.2_0.8"]["accepted"], 0)
        self.assertEqual(calibrated["noul_thresholds_0.2_0.8"]["accepted"], 2)
        self.assertLess(calibrated["nll"], baseline["nll"])
        self.assertEqual(noul_status(before[0], 1.0), "abstained")

    def test_paired_regression_counts_and_mismatched_evidence_rejection(self):
        before = [row("a", [1, 0], [0.0, 1.0]), row("b", [2, 0], [1.0, 0.0])]
        after = [row("a", [0, 2], [0.0, 1.0]), row("b", [0, 1], [1.0, 0.0])]
        result = paired_decisions(before, after, 1.0, 1.0)
        self.assertEqual(result["incorrect_to_correct"], 1)
        self.assertEqual(result["correct_to_incorrect"], 1)
        self.assertEqual(result["argmax_changed"], 2)
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            paired_decisions(before, list(reversed(after)), 1.0, 1.0)
        after[0]["target"] = [1.0, 0.0]
        with self.assertRaisesRegex(ValueError, "targets or types differ"):
            paired_decisions(before, after, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
