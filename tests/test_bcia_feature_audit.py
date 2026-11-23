import unittest

import numpy as np
import torch

from bcia_feature_audit import (
    RegressionScores, add_intercept, fit_linear, majority_labels, patch_metrics, patch_samples,
    regression_input, route_groups, weighted_cross_products,
)


CPU = torch.device("cpu")


class FeatureAuditTest(unittest.TestCase):
    def test_patch_label_ignores_erosion_and_rejects_ties_or_pluralities(self):
        labels = torch.full((32, 32), 255, dtype=torch.long)
        labels[0, :5] = torch.tensor([2, 2, 2, 1, 3])
        labels[0, 16:20] = torch.tensor([1, 1, 2, 2])
        labels[16, :5] = torch.tensor([0, 0, 1, 1, 2])
        prediction = torch.full_like(labels, 2)
        majority, valid, matches = majority_labels(labels, prediction)
        self.assertEqual(majority.tolist(), [[2, 255], [255, 255]])
        self.assertEqual(valid.tolist(), [[5, 4], [5, 0]])
        self.assertEqual(matches.tolist(), [[3, 2], [1, 0]])

    def test_weighted_linear_fit_and_unpenalized_intercept(self):
        x = torch.tensor([[0.0], [1.0], [2.0], [50.0]], dtype=torch.float64)
        y = 5 - 2 * x
        y[-1] = 999
        weight = torch.tensor([1.0, .5, 1.0, 0.0], dtype=torch.float64)
        g, h = weighted_cross_products(add_intercept(x), y, weight)
        fitted = fit_linear(g, h, weight.sum(), 1e-9)
        torch.testing.assert_close(fitted[:, 0], torch.tensor([-2., 5.], dtype=torch.float64))

    def test_r2_uses_train_mean_baseline_and_q_zero_excludes_patch(self):
        weight = torch.tensor([1., .5, 0.], dtype=torch.float64)
        target = torch.tensor([0., 2., 999.], dtype=torch.float64)[:, None].expand(-1, 64)
        baseline = RegressionScores(CPU)
        baseline.add(target, torch.ones_like(target), weight)
        prediction = RegressionScores(CPU)
        prediction.add(target, target, weight)
        report = baseline.scores(baseline)
        self.assertEqual(report["mse"], 1.)
        self.assertEqual(report["supported_patches"], 2)
        self.assertEqual(report["r2_vs_train_mean"], 0.)
        self.assertEqual(report["prediction_variance"], 0.)
        self.assertEqual(prediction.scores(baseline)["r2_vs_train_mean"], 1.)

    def test_calibration_preserves_invalid_dsm_context(self):
        q = torch.tensor([[[0., .5]]])
        batch = dict(z_rgb=torch.zeros(1, 64, 1, 2), z_dsm=torch.zeros(1, 64, 1, 2), coverage=q)
        stats = {("raw", key): (torch.full((64,), 2.), torch.full((64,), 4.))
                 for key in ("z_rgb", "z_dsm")}
        dsm = regression_input(batch, "train_channel_std", "z_dsm", stats)
        rgb = regression_input(batch, "train_channel_std", "z_rgb", stats)
        self.assertEqual(dsm[0, 0, 0].tolist(), [0., -.5])
        self.assertEqual(rgb[0, 0, 0].tolist(), [-.5, -.5])

    def test_deterministic_prediction_sampling_and_route_partitions(self):
        sample = patch_samples(4)
        np.testing.assert_array_equal(sample, patch_samples(4))
        self.assertTrue(all(len(set(row)) == 128 for row in sample))
        q = torch.tensor([0., .25, .5, .75, .9, 1.])
        label = torch.tensor([0, 0, 0, 0, 1, 255])
        valid = torch.tensor([0, 5, 2, 6, 8, 10])
        correct = torch.tensor([0, 4, 1, 3, 8, 1])
        cases = list(route_groups(q, label, valid, correct, q, [.25, .5, .75]))
        for group in ("coverage", "truth", "g2_pixel_correctness", "innovation_norm"):
            masks = [mask.int() for name, _, mask in cases if name == group]
            torch.testing.assert_close(sum(masks), (q > 0).int())

    def test_patch_iou_uses_first_five_and_leaves_missing_class_undefined(self):
        mat = torch.eye(6, dtype=torch.int64)
        mat[-1, -1] = 0
        scores = patch_metrics(mat)
        self.assertEqual(scores["mIoU5"], 100.)
        self.assertEqual(scores["OA"], 100.)
        self.assertIsNone(scores["mIoU6"])
