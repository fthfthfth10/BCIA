import unittest

import torch

from bcia_directional_context_audit import (
    BASE, PREDICTION, RESIDUAL, probe_features, raw_affine, transform_raw_weights,
)


class DirectionalContextTest(unittest.TestCase):
    def test_zero_padding_and_unsupported_dsm(self):
        rgb = torch.ones(1, 64, 3, 3)
        q = torch.ones(1, 3, 3)
        q[0, 1, 1] = 0
        support = q[:, None] > 0
        dsm = torch.where(support, rgb, 0.0)
        pred_rgb = dsm * 2
        pred_dsm = rgb * 3
        features = probe_features(dict(z_rgb=rgb, z_dsm=dsm, pred_rgb=pred_rgb, pred_dsm=pred_dsm,
                                       u_rgb=rgb-pred_rgb, u_dsm=dsm-torch.where(support, pred_dsm, 0.0),
                                       coverage=q))
        self.assertAlmostEqual(features["pool_rgb"][0, 0, 0, 0].item(), 4/9)
        self.assertAlmostEqual(features["pool_dsm"][0, 0, 0, 0].item(), 3/9)
        for field in ("z_dsm", "pred_rgb", "pred_dsm", "pool_dsm", "u_dsm"):
            self.assertEqual(features[field][0, 0, 1, 1].item(), 0.)
        self.assertEqual(features["u_rgb"][0, 0, 1, 1].item(), 1.)

    def test_transfers_standardized_readout_in_both_directions(self):
        generator = torch.Generator().manual_seed(7)
        rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
        z_rgb, z_dsm, pred_rgb, pred_dsm = [rand(24, 64) for _ in range(4)]
        pred = torch.cat((z_rgb, z_dsm, pred_rgb, pred_dsm), dim=1)
        residual = torch.cat((z_rgb, z_dsm, z_rgb-pred_rgb, z_dsm-pred_dsm), dim=1)
        for fields, old, new in ((PREDICTION, pred, residual), (RESIDUAL, residual, pred)):
            with self.subTest(fields=fields):
                names = (*BASE, *fields)
                moments = {field: (rand(64), rand(64).abs() + .3) for field in names}
                coefficient = rand(257, 6)
                weight, bias = raw_affine(coefficient, names, moments)
                mean, scale = [torch.cat(values) for values in zip(*moments.values())]
                expected = ((old-mean)/scale) @ coefficient[:-1] + coefficient[-1]
                transferred = transform_raw_weights(weight)
                torch.testing.assert_close(expected, new @ transferred + bias, rtol=0, atol=1e-10)
                torch.testing.assert_close(weight, transform_raw_weights(transferred), rtol=0, atol=1e-12)
