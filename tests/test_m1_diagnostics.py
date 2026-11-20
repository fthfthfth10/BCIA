import unittest

import numpy as np
import torch

from m1_diagnostics import PredictionStats, count_pairs, metrics, scene_bootstrap


class M1DiagnosticsTest(unittest.TestCase):
    def test_joint_counts_keep_true_class_and_corrections(self):
        label = torch.tensor([0, 1, 1, 2])
        first = torch.tensor([1, 1, 0, 2])
        second = torch.tensor([0, 2, 2, 2])
        state = 2 * (first == label).long() + (second == label).long()
        self.assertEqual(count_pairs(label, state, 4).tolist()[:3],
                         [[0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])
        self.assertEqual(int(count_pairs(label, second, 6)[0, 0]), 1)

    def test_scene_bootstrap_is_paired_and_reproducible(self):
        first = np.stack([np.eye(6, dtype=np.int64) * 10 for _ in range(14)])
        second = first.copy()
        first[0, 0, 1] = 5
        result = scene_bootstrap(first, second)
        self.assertEqual(result, scene_bootstrap(first, second))
        self.assertEqual(result["draws"], 5000)
        self.assertEqual(result["scenes"], 14)
        self.assertEqual(result["lower"], 0.0)
        self.assertGreater(result["upper"], 0)
        self.assertEqual(metrics(second.sum(axis=0))["mIoU5"], 100)

    def test_prediction_stats_with_fractional_coverage(self):
        stats = PredictionStats("cpu")
        target = torch.tensor([0.0, 2.0, 500.0]).view(1, 1, 1, 3).expand(1, 64, 1, 3)
        q = torch.tensor([1.0, 0.5, 0.0]).view(1, 1, 1, 3)
        pred = target * 0.5
        stats.update(target, pred, target - pred, torch.full_like(q, .25), q)
        result = stats.compute()
        self.assertAlmostEqual(result["prediction_mse"], 1 / 3)
        self.assertAlmostEqual(result["target_variance"], 8 / 9)
        self.assertAlmostEqual(result["zero_prediction_mse"], 4 / 3)
        self.assertAlmostEqual(result["gate_mean"], .25)
        self.assertAlmostEqual(result["residual_l2_mean"], 8 / 3)
