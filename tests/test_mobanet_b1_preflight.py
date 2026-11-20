import unittest

import torch

from tests.mobanet_b1_preflight import pad_for_dinov2


class MoBaNetInputTest(unittest.TestCase):
    def test_padding_preserves_window_and_zeroes_invalid_dsm(self):
        rgb = torch.ones(1, 3, 512, 512)
        dsm = torch.full((1, 1, 512, 512), 2.0)
        valid = torch.ones_like(dsm, dtype=torch.bool)
        valid[..., 0, 0] = False
        dsm[..., 0, 1] = float("nan")
        padded_rgb, padded_dsm = pad_for_dinov2(rgb, dsm, valid)
        self.assertEqual(padded_rgb.shape, (1, 3, 518, 518))
        self.assertEqual(padded_dsm.shape, (1, 518, 518))
        self.assertTrue(torch.equal(padded_rgb[..., :512, :512], rgb))
        self.assertTrue(torch.isfinite(padded_dsm).all())
        self.assertEqual(padded_dsm[0, 0, :3].tolist(), [0, 0, 2])
        self.assertEqual(padded_rgb[..., 512:, :].count_nonzero().item(), 0)
        self.assertEqual(padded_dsm[..., :, 512:].count_nonzero().item(), 0)
