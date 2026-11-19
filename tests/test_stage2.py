import unittest

import torch
from torch import nn

from bcia_peft.adapter import BCIA, LightweightDSMEncoder
from bcia_peft.convert_checkpoint import split_qkv


class StageTwoTest(unittest.TestCase):
    def test_qkv_mask_applies_before_key_bias_compatibility_check(self):
        weight = torch.arange(12.0).reshape(6, 2)
        bias = torch.arange(1.0, 7.0)
        mask = torch.tensor([1, 0, 0, 0, 0, 1])
        weights, (query, value) = split_qkv(weight, bias, mask)
        self.assertTrue(torch.equal(torch.cat(weights), weight))
        self.assertEqual(query.tolist(), [1, 0])
        self.assertEqual(value.tolist(), [0, 6])
        mask[2] = 1
        with self.assertRaisesRegex(ValueError, "effective K bias"):
            split_qkv(weight, bias, mask)
        with self.assertRaisesRegex(ValueError, "non-binary"):
            split_qkv(weight, bias, mask.float() / 2)

    def test_nonzero_residual_isolates_invalid_patches_and_retains_valid_gradients(self):
        torch.manual_seed(31)
        encoder, adapter = LightweightDSMEncoder(), BCIA()
        nn.init.normal_(adapter.output_projection.weight, std=0.02)
        spatial = torch.randn(1, 1024, 1024)
        height = torch.randn(1, 1, 512, 512)
        valid = torch.ones_like(height, dtype=torch.bool)
        valid[..., :16, :16] = False
        height[..., :16, :16] = float("nan")
        changed = height.clone()
        changed[..., :16, :16] = 1e10
        changed[..., :16, 16:32] += 5
        changed[..., 16:32, :16] *= -5

        def run(dsm):
            return adapter(spatial, *encoder(dsm, valid))

        original = run(height)
        invalid_only = height.clone()
        invalid_only[..., :16, :16] = -float("inf")
        with torch.no_grad():
            also_valid = run(changed)
            payload_only = run(invalid_only)
        self.assertTrue(all(tensor.isfinite().all() for tensor in original))
        self.assertTrue(torch.equal(original.residual, payload_only.residual))
        self.assertTrue(torch.equal(original.residual[:, 0], also_valid.residual[:, 0]))
        self.assertTrue(torch.equal(original.w_rgb[..., 0, 0], also_valid.w_rgb[..., 0, 0]))
        self.assertEqual(original.pred_rgb[..., 0, 0].count_nonzero().item(), 0)
        self.assertTrue(torch.equal(original.u_rgb[..., 0, 0], original.z_rgb[..., 0, 0]))
        self.assertEqual(original.dsm_contribution[..., 0, 0].count_nonzero().item(), 0)
        self.assertGreater(original.residual[:, 0].count_nonzero().item(), 0)
        self.assertGreater((original.residual[:, 1] - also_valid.residual[:, 1]).abs().max().item(), 1e-5)
        original.residual.square().mean().backward()
        for module in (encoder, adapter.rgb_projection, adapter.g_rgb_to_dsm, adapter.g_dsm_to_rgb,
                       adapter.router, adapter.mix, adapter.output_projection):
            gradients = [param.grad for param in module.parameters()]
            self.assertTrue(all(grad is not None and grad.isfinite().all() for grad in gradients))
            self.assertTrue(any(grad.count_nonzero() for grad in gradients))
