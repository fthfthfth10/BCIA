import random
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from bcia_peft.adapter import prediction_losses
from bcia_peft.data import SynchronizedGeometry
from bcia_peft.model import BCIAPeft
from bcia_peft.train import (StepBatchSampler, best_record, build_optimizer, check_config,
                             check_pilot_gradients, load_state, prune_previous_best, save_state)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.register_tokens = nn.Parameter(torch.zeros(1, 4, 1024))
        self.embeddings.patch_embeddings = nn.Conv2d(3, 1024, 16, stride=16)
        self.embeddings.forward = self.embed
        self.rope_embeddings = nn.Identity()
        self.layer = nn.ModuleList(TinyBlock() for _ in range(24))
        self.norm = nn.LayerNorm(1024)
        self.requires_grad_(False).eval()

    def embed(self, rgb):
        tokens = self.embeddings.patch_embeddings(rgb).flatten(2).transpose(1, 2)
        registers = self.embeddings.register_tokens.expand(rgb.shape[0], -1, -1)
        return torch.cat((torch.zeros_like(registers[:, :1]), registers, tokens), dim=1)

    def forward(self, rgb):
        tokens = self.norm(self.embed(rgb))[:, 5:]
        return (tokens.transpose(1, 2).reshape(-1, 1024, 32, 32).contiguous(),)


class TinyBlock(nn.Module):
    def forward(self, tokens, **kwargs):
        return tokens


class TinyHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(1024, 6, 1)

    def init_weights(self):
        pass

    def forward(self, features):
        return self.conv(features[0])


class MechanismsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_initial_logits_frozen_parameters_and_shared_weights(self):
        torch.manual_seed(81)
        rgb = torch.randn(1, 3, 512, 512)
        dsm = torch.randn(1, 1, 512, 512)
        valid = torch.ones_like(dsm, dtype=torch.bool)
        with patch("bcia_peft.model.make_backbone", TinyBackbone), \
                patch("bcia_peft.model.SegmenterMaskTransformerHead", TinyHead):
            models = []
            for index in range(5):
                torch.manual_seed(19)
                models.append(BCIAPeft(random_init=True, group=f"G{index}").train())
        with torch.no_grad():
            expected = models[0](rgb, dsm, valid)
            for model in models:
                torch.testing.assert_close(model(rgb, dsm, valid), expected, rtol=0, atol=0)
                self.assertFalse(model.backbone.training)
                self.assertTrue(all(not param.requires_grad for param in model.backbone.parameters()))
        self.assertFalse(any(param.requires_grad for param in models[0].bcia.parameters()))
        self.assertFalse(any(param.requires_grad for param in models[0].dsm_encoder.parameters()))
        self.assertEqual(models[1].parameter_counts()["bcia"]["trainable"], 214272)
        for model in models[3:]:
            for key, value in models[2].state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)

    def test_auxiliary_values_and_gradients_masks(self):
        from bcia_peft.adapter import BCIA

        torch.manual_seed(38)
        adapter = BCIA()
        spatial = torch.randn(1, 1024, 1024)
        dsm = torch.randn(1, 64, 32, 32, requires_grad=True)
        coverage = torch.ones(1, 1, 32, 32)
        coverage[..., 0, 0] = 0
        coverage[..., 0, 1] = 0.5
        output = adapter(spatial, dsm, coverage)
        direct = prediction_losses(output)
        detached = prediction_losses(output, predictors_only=adapter)
        parameters = (adapter.g_rgb_to_dsm.weight, adapter.g_dsm_to_rgb.weight,
                      adapter.rgb_projection.weight, dsm, adapter.router.weight)
        for key in direct:
            self.assertTrue(torch.equal(direct[key], detached[key]))
        grads = torch.autograd.grad(sum(direct.values()), parameters, allow_unused=True)
        isolated = torch.autograd.grad(sum(detached.values()), parameters, allow_unused=True)
        for index in (0, 1):
            self.assertGreater(grads[index].abs().sum().item(), 0)
            self.assertTrue(torch.equal(grads[index], isolated[index]))
        for index in (2, 3):
            self.assertGreater(grads[index].abs().sum().item(), 0)
            self.assertIsNone(isolated[index])
        self.assertIsNone(grads[4])
        self.assertIsNone(isolated[4])
        self.assertEqual(output.dsm_contribution[..., 0, 0].count_nonzero(), 0)
        suppressed = adapter(spatial, dsm, coverage, suppress=("dsm",))
        self.assertEqual(suppressed.dsm_contribution.count_nonzero(), 0)
        self.assertTrue(torch.equal(suppressed.pred_dsm, output.pred_dsm))

    def test_shared_geometry_resize_crop_and_flip(self):
        np.random.seed(7)
        label = np.zeros((512, 512), dtype=np.uint8)
        label[:, 256:] = 1
        rgb = np.repeat((label * 255)[..., None], 3, axis=2)
        dsm = label.astype(np.float32)
        valid = np.ones_like(label, dtype=bool)
        valid[0, 0] = False
        flipped = SynchronizedGeometry(ratio_range=(1, 1), flip_prob=1)
        out_rgb, out_dsm, out_valid, out_label = flipped(rgb, dsm, valid, label)
        np.testing.assert_array_equal(out_label, label[:, ::-1])
        np.testing.assert_array_equal(out_valid, valid[:, ::-1])
        np.testing.assert_array_equal(out_rgb[..., 0], out_label * 255)
        np.testing.assert_array_equal(out_dsm[out_valid], out_label[out_valid])
        cropped = SynchronizedGeometry(ratio_range=(1.25, 1.25), flip_prob=1)
        out_rgb, out_dsm, out_valid, out_label = cropped(rgb, dsm, valid, label)
        self.assertEqual(out_label.shape, (512, 512))
        np.testing.assert_allclose(out_rgb[..., 0][out_valid] / 255, out_dsm[out_valid], atol=0.005)
        np.testing.assert_array_equal(out_dsm[out_valid] >= 0.5, out_label[out_valid])

    def test_step_sampling_and_resume(self):
        config = dict(lr=0.01, weight_decay=0.05, steps=5, warmup_steps=2,
                      warmup_start_ratio=0.25, min_lr_ratio=0.1, power=1)
        batches = list(StepBatchSampler(7, 2, 23, 0, 5))
        self.assertEqual(len({index for batch in batches[:3] for index, _ in batch}), 6)
        self.assertEqual(batches[2:], list(StepBatchSampler(7, 2, 23, 2, 5)))
        model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(), nn.Linear(2, 1))
        optimizer, scheduler = build_optimizer(model, config)
        device = torch.device("cpu")

        def update(batch):
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.tensor(batch, dtype=torch.float32) / 2**32
            inputs += torch.rand_like(inputs) + random.random() + np.random.random()
            loss = model(inputs).square().mean()
            loss.backward()
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            return loss.item(), lr

        identity = {"group": "synthetic"}
        update(batches[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pth"
            save_state(path, model, optimizer, scheduler, identity, 1, None, device)
            results = [update(batch) for batch in batches[1:]]
            expected = {name: value.clone() for name, value in model.state_dict().items()}
            with self.assertRaises(ValueError):
                load_state(path, model, optimizer, scheduler, {"group": "different"})
            self.assertEqual(load_state(path, model, optimizer, scheduler, identity), (1, None))
            resumed = [update(batch) for batch in StepBatchSampler(7, 2, 23, 1, 5)]
        self.assertEqual(results, resumed)
        self.assertTrue(all(torch.equal(value, model.state_dict()[name])
                            for name, value in expected.items()))

    def test_budget_and_auxiliary_weights_are_explicit(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mechanisms.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "steps"):
            check_config(config)

    def test_pilot_config_and_formal_best(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "g2_pilot.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        check_config(config, pilot=True)
        with self.assertRaisesRegex(ValueError, "validate_every"):
            check_config(config)
        first = best_record(dict(mIoU5=70, mIoU6=60), None, 1)
        self.assertIs(best_record(dict(mIoU5=69, mIoU6=61), first, 2), first)
        second = best_record(dict(mIoU5=71, mIoU6=59), first, 3)
        self.assertEqual((second["metric"], second["step"], second["score"]), ("mIoU5", 3, 71))
        self.assertIs(best_record(dict(mIoU5=71, mIoU6=80), second, 4), second)
        self.assertIs(best_record(dict(mIoU5=None, mIoU6=90), second, 5), second)

    def test_pilot_cap_preserves_long_schedule(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "g2_pilot.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        model = nn.Sequential(nn.Linear(1, 1))
        optimizer, scheduler = build_optimizer(model, config)
        for _ in StepBatchSampler(3456, 1, 42, 0, 64):
            optimizer.step()
            scheduler.step()
        self.assertEqual(scheduler.last_epoch, 64)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4 * (0.1 + 0.9 * 64 / 1500))

    def test_best_cleanup_after_checkpoint_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            previous = best_record({"mIoU5": 1}, None, 4)
            current = best_record({"mIoU5": 2}, previous, 8)
            old, new, last = (work / filename for filename in
                              (previous["file"], current["file"], "last.pth"))
            old.touch()
            prune_previous_best(work, previous, current)
            self.assertTrue(old.exists())
            new.touch()
            prune_previous_best(work, previous, current)
            self.assertTrue(old.exists())
            last.touch()
            prune_previous_best(work, previous, current)
            self.assertFalse(old.exists())
            self.assertTrue(new.exists() and last.exists())

    def test_pilot_gradient_guards(self):
        model = nn.Module()
        model.backbone = nn.Linear(1, 1).requires_grad_(False)
        model.decode_head = nn.Linear(1, 1)
        model.dsm_encoder = nn.Linear(1, 1)
        model.bcia = nn.ModuleDict(dict(mix=nn.Linear(1, 1), output_projection=nn.Linear(1, 1)))
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.grad = torch.full_like(param, float("mix" not in name))
        self.assertEqual(check_pilot_gradients(model, 1)["bcia.mix"], 0)
        with self.assertRaisesRegex(RuntimeError, "upstream"):
            check_pilot_gradients(model, 4)
        model.bcia["mix"].weight.grad.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "bcia.mix"):
            check_pilot_gradients(model, 2)
        model.backbone.weight.grad = torch.ones_like(model.backbone.weight)
        with self.assertRaisesRegex(RuntimeError, "Frozen DINO"):
            check_pilot_gradients(model, 2)

    def test_budget_auxiliary_coefficients(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mechanisms.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(steps=3, validate_every=2, save_every=2)
        check_config(config)
        config["group"] = "G3"
        with self.assertRaisesRegex(ValueError, "coefficients"):
            check_config(config)
