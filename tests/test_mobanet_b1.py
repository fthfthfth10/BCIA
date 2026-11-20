import random
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from mobanet_b1 import MoBaNetB1, choose_masks, supervised_losses


class FixtureNet(nn.Module):
    def __init__(self, upstream):
        super().__init__()
        self.upstream = upstream
        self.decoder = nn.Module()
        self.decoder.ppm = nn.Module()
        self.decoder.ppm.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Sequential(
                nn.Conv2d(3, 6, 1), nn.BatchNorm2d(6), nn.Identity())),
            nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.BatchNorm2d(3)),
        ])

    def forward(self, rgb, dsm, mode):
        assert rgb.shape[-2:] == (518, 518)
        if mode == "Train":
            rgb, dsm = self.upstream.apply_mcrc_mask(rgb, dsm, .25, 1, (.1, .3), (.5, 2))
        output = self.decoder.ppm.stages[0](rgb).expand(-1, -1, 518, 518)
        return (output, output, output) if mode == "Train" else output


class MoBaNetB1Test(unittest.TestCase):
    def test_ppm_batch_one_train_and_crop(self):
        original = object()
        upstream = SimpleNamespace(apply_mcrc_mask=original)
        model = MoBaNetB1(upstream, FixtureNet(upstream), lambda *args: None)
        model.eval()
        model.train()
        self.assertFalse(model.global_bn.training)
        self.assertTrue(model.net.decoder.ppm.stages[1][1].training)
        stats = model.global_bn.running_mean.clone()
        inputs = (torch.ones(1, 3, 512, 512), torch.zeros(1, 1, 512, 512),
                  torch.ones(1, 1, 512, 512, dtype=torch.bool))
        outputs = model(*inputs)
        self.assertTrue(all(output.shape == (1, 6, 512, 512) for output in outputs))
        outputs[0].sum().backward()
        self.assertTrue(torch.equal(model.global_bn.running_mean, stats))
        for parameter in model.global_bn.parameters():
            self.assertTrue(parameter.requires_grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertIs(upstream.apply_mcrc_mask, original)
        model.eval()
        self.assertEqual(model(*inputs).shape, (1, 6, 512, 512))
        self.assertIsNone(model.last_mask)

    def test_mask_probability_and_rng_replay(self):
        torch.manual_seed(42)
        first = choose_masks(20000, .25, "cpu")
        self.assertFalse((first[0] & first[1]).any())
        self.assertAlmostEqual((first[0] | first[1]).float().mean().item(), .25, delta=.015)
        self.assertTrue(all(abs(mask.float().mean().item() - .125) < .012 for mask in first))
        torch.manual_seed(42)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, choose_masks(20000, .25, "cpu"))))
        torch.manual_seed(42)
        random.seed(42)
        exercised = []

        def local_mask(tensor, indices, blocks, scale, aspect):
            if indices.numel():
                tensor[indices, :, :3, :3] = 0

        upstream = SimpleNamespace(apply_mcrc_mask=None)
        model = MoBaNetB1(upstream, FixtureNet(upstream), local_mask)
        for _ in range(100):
            rgb, dsm = model._mcrc(torch.ones(1, 3, 5, 5), torch.ones(1, 5, 5),
                                   .25, 1, (.1, .3), (.5, 2))
            self.assertEqual(bool(model.last_mask[0]), bool((rgb == 0).any()))
            self.assertEqual(bool(model.last_mask[1]), bool((dsm == 0).any()))
            exercised.append((bool(model.last_mask[0]), bool(model.last_mask[1])))
        self.assertEqual(set(exercised), {(False, False), (True, False), (False, True)})

    def test_auxiliary_targets_ignore_correct_and_undefined_pixels(self):
        fused = torch.tensor([[[[2., 2., 2.]], [[0., 0., 0.]]]], requires_grad=True)
        labels = torch.tensor([[[0, 1, 255]]])
        auxiliary = torch.zeros_like(fused, requires_grad=True)
        main, rgb, dsm = supervised_losses((fused, auxiliary, auxiliary), labels)
        torch.testing.assert_close(main, F.cross_entropy(fused, labels, ignore_index=255))
        torch.testing.assert_close(rgb, F.cross_entropy(auxiliary[..., 1:2], torch.ones(1, 1, 1, dtype=torch.long)))
        (main + .01 * (rgb + dsm)).backward()
        self.assertTrue((fused.grad[..., 2] == 0).all())
        self.assertTrue((auxiliary.grad[..., [0, 2]] == 0).all())
        self.assertTrue(all(loss.item() == 0 for loss in supervised_losses(
            (fused, auxiliary, auxiliary), torch.full_like(labels, 255))))
