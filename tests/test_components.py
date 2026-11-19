import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bcia_peft.adapter import BCIA, LightweightDSMEncoder, prediction_losses
from bcia_peft.checkpoints import inspect_state, load_backbone_checkpoint
from bcia_peft.data import PotsdamValidation
from bcia_peft.model import BCIAPeft, insert_residual
from bcia_peft.validation import PotsdamMetrics, validate


class ComponentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_dsm_invalid_patch_and_partial_coverage(self):
        encoder = LightweightDSMEncoder()
        dsm = torch.randn(1, 1, 512, 512)
        valid = torch.ones_like(dsm, dtype=torch.bool)
        dsm[..., :16, :16] = float("nan")
        valid[..., :8, 16:32] = False
        dsm[..., :8, 16:32] = float("inf")
        dsm[..., 16, 0] = -float("inf")
        features, q = encoder(dsm, valid)
        self.assertEqual(features.shape, (1, 64, 32, 32))
        self.assertEqual(q.shape, (1, 1, 32, 32))
        self.assertEqual(q[0, 0, 0, :3].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(q[0, 0, 1, 0].item(), 255 / 256)
        self.assertTrue(features.isfinite().all())
        self.assertEqual(features[..., 0, 0].count_nonzero().item(), 0)
        clean_valid = valid & torch.isfinite(dsm)
        clean = torch.where(clean_valid, dsm, 999.0)
        clean_features, clean_q = encoder(clean, clean_valid)
        self.assertTrue(torch.equal(features, clean_features))
        self.assertTrue(torch.equal(q, clean_q))

    def test_router_predictors_and_isolated_dsm_gate(self):
        torch.manual_seed(8)
        adapter = BCIA()
        inputs_seen = {}

        def observe(name):
            def hook(_, inputs):
                inputs_seen[name] = inputs[0].detach()
            return hook

        for name in ("g_rgb_to_dsm", "g_dsm_to_rgb", "router"):
            getattr(adapter, name).register_forward_pre_hook(observe(name))
        spatial = torch.randn(1, 1024, 1024)
        height = torch.randn(1, 64, 32, 32)
        coverage = torch.ones(1, 1, 32, 32)
        coverage[..., 0, 0] = 0
        coverage[..., 1, 1] = 0.5
        height[..., 0, 0] = float("nan")
        output = adapter(spatial, height, coverage)
        self.assertTrue(all(tensor.isfinite().all() for tensor in output))
        self.assertTrue(all(tensor.count_nonzero() for tensor in inputs_seen.values()))
        self.assertEqual(inputs_seen["router"].shape, (1, 257, 32, 32))
        self.assertEqual(output.residual.count_nonzero().item(), 0)
        for field in ("u_dsm", "w_dsm", "dsm_contribution"):
            self.assertEqual(getattr(output, field)[..., 0, 0].count_nonzero().item(), 0)
        self.assertTrue(torch.equal(output.u_rgb[..., 0, 0], output.z_rgb[..., 0, 0]))
        self.assertGreater(output.w_rgb[..., 0, 0].item(), 0)
        self.assertFalse(torch.allclose((output.w_rgb + output.w_dsm)[..., 2:, 2:],
                                        torch.ones_like(coverage[..., 2:, 2:])))
        alt = adapter(spatial, torch.randn_like(height), coverage)
        self.assertTrue(torch.equal(output.mixed[..., 0, 0], alt.mixed[..., 0, 0]))
        self.assertTrue(all(loss.item() > 0 for loss in prediction_losses(output).values()))
        torch.stack(tuple(prediction_losses(output).values())).sum().backward()
        for predictor in (adapter.g_rgb_to_dsm, adapter.g_dsm_to_rgb):
            self.assertGreater(predictor.weight.grad.count_nonzero().item(), 0)
        with torch.no_grad():
            no_dsm = adapter(spatial, torch.full_like(height, float("inf")),
                             torch.zeros_like(coverage))
        self.assertTrue(all(tensor.isfinite().all() for tensor in no_dsm))
        self.assertEqual(no_dsm.dsm_contribution.count_nonzero().item(), 0)
        self.assertTrue(all(loss.item() == 0 for loss in prediction_losses(no_dsm).values()))
        self.assertGreater(no_dsm.mixed.count_nonzero().item(), 0)

    def test_spatial_only_insertion_with_nonzero_residual(self):
        tokens = torch.randn(1, 1029, 1024)
        residual = torch.full((1, 1024, 1024), 0.125)
        inserted = insert_residual(tokens, residual)
        self.assertTrue(torch.equal(inserted[:, :5], tokens[:, :5]))
        self.assertTrue(torch.equal(inserted[:, 5:], tokens[:, 5:] + residual))
        self.assertFalse(torch.equal(inserted[:, 5:], tokens[:, 5:]))

    def test_checkpoint_gate(self):
        norm = torch.nn.LayerNorm(4)
        before = norm.weight.detach().clone()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "state.pth"
            torch.save({"weight": torch.full((4,), 2.0)}, path)
            with self.assertRaisesRegex(ValueError, r'"missing": \["bias"\]'):
                load_backbone_checkpoint(norm, path)
            self.assertTrue(torch.equal(norm.weight, before))
            torch.save({"weight": torch.full((4,), 2.0), "bias": torch.zeros(4)}, path)
            load_backbone_checkpoint(norm, path)
        self.assertTrue(torch.equal(norm.weight, torch.full((4,), 2.0)))
        audit = inspect_state({"norm.weight": torch.zeros(4, dtype=torch.float64),
                               "norm.bias": torch.tensor([float("inf")])},
                              {"norm.weight": ((4,), torch.float32),
                               "norm.bias": ((4,), torch.float32)})
        self.assertEqual(audit["zero_layernorm_scales"], ["norm.weight"])
        self.assertEqual(audit["shape_mismatch"], ["norm.bias"])
        self.assertEqual(audit["dtype_mismatch"], ["norm.weight"])
        self.assertEqual(audit["nonfinite"], ["norm.bias"])
        self.assertFalse(audit["compatible"])

    def test_validation_tile_loader_and_metrics(self):
        class PerfectTile(torch.nn.Module):
            def __init__(self, target):
                super().__init__()
                self.target = target

            def forward(self, rgb, dsm, valid):
                return torch.nn.functional.one_hot(self.target.clamp_max(5), 6).permute(2, 0, 1)[None].float()

            def logits_for_tile(self, logits):
                return logits

        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "manifests").mkdir()
            (root / "metadata.json").write_text('{"status":"verified"}')
            (root / "normalization.json").write_text('{"split":"train","mean":2,"std":2}')
            row = dict(id="tile", rgb_path="rgb.png", dsm_path="dsm.npy",
                       valid_path="valid.png", label_path="label.png")
            (root / "manifests" / "val.jsonl").write_text(json.dumps(row) + "\n")
            rgb = np.full((512, 512, 3), [100, 110, 120], dtype=np.uint8)
            Image.fromarray(rgb).save(root / row["rgb_path"])
            dsm = np.full((512, 512), 4.0, dtype=np.float32)
            dsm[0, 0] = np.inf
            np.save(root / row["dsm_path"], dsm)
            valid = np.ones((512, 512), dtype=np.uint8)
            valid[0, 1] = 0
            Image.fromarray(valid).save(root / row["valid_path"])
            label = np.tile(np.arange(512) % 7, (512, 1)).astype(np.uint8)
            Image.fromarray(label).save(root / row["label_path"])
            dataset = PotsdamValidation(root)
            sample = dataset[0]
            self.assertEqual(sample["label"][0, :7].tolist(), [255, 0, 1, 2, 3, 4, 5])
            self.assertEqual(sample["dsm"][0, 0, :3].tolist(), [0, 0, 1])
            self.assertEqual(sample["valid"][0, 0, :3].tolist(), [False, False, True])
            torch.testing.assert_close(sample["rgb"][:, 0, 0], torch.tensor(
                [(100 - 123.675) / 58.395, (110 - 116.28) / 57.12, (120 - 103.53) / 57.375]))
            result = validate(PerfectTile(sample["label"]), dataset, "cpu")
            self.assertEqual((result["tiles"], result["mIoU5"], result["mIoU6"], result["OA"]),
                             (1, 100.0, 100.0, 100.0))
        metrics = PotsdamMetrics()
        metrics.update(torch.tensor([0, 1, 2, 3, 4, 4, 5]),
                       torch.tensor([0, 1, 2, 3, 4, 5, 255]))
        result = metrics.compute()
        self.assertAlmostEqual(result["mIoU5"], 90.0)
        self.assertAlmostEqual(result["mIoU6"], 75.0)
        self.assertAlmostEqual(result["OA"], 500.0 / 6)
        self.assertIsNone(PotsdamMetrics().compute()["OA"])

    def test_model_requires_explicit_init(self):
        with self.assertRaises(ValueError):
            BCIAPeft()
        with self.assertRaises(ValueError):
            BCIAPeft(checkpoint="unused.pth", random_init=True)


if __name__ == "__main__":
    unittest.main()
