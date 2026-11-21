import unittest
from types import SimpleNamespace

import torch
from torch import nn

from lora_b2 import LoRALinear, inject_lora, lora_modules


class B2LoRATest(unittest.TestCase):
    def test_initial_identity_and_first_step_gradients(self):
        torch.manual_seed(42)
        base = nn.Linear(16, 12).requires_grad_(False)
        module = LoRALinear(base)
        inputs = torch.randn(3, 16)
        expected = base(inputs)
        self.assertTrue(torch.equal(module(inputs), expected))
        self.assertEqual(module.scaling, 1)
        module(inputs).square().sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertIsNone(base.bias.grad)
        self.assertEqual(module.lora_a.weight.grad.count_nonzero().item(), 0)
        self.assertGreater(module.lora_b.weight.grad.count_nonzero().item(), 0)

    def test_all_qv_projections_only(self):
        def block():
            projections = {name: nn.Linear(1024, 1024).requires_grad_(False)
                           for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
            return SimpleNamespace(attention=SimpleNamespace(**projections))

        with torch.device("meta"):
            model = SimpleNamespace(group="G1", backbone=SimpleNamespace(layer=[block() for _ in range(24)]))
            inject_lora(model)
        self.assertEqual(len(lora_modules(model)), 48)
        self.assertEqual(sum(param.numel() for module in lora_modules(model)
                             for param in module.parameters() if param.requires_grad), 393216)
        for layer in model.backbone.layer:
            self.assertIsInstance(layer.attention.k_proj, nn.Linear)
            self.assertIsInstance(layer.attention.o_proj, nn.Linear)
        with self.assertRaisesRegex(ValueError, "frozen"):
            LoRALinear(nn.Linear(16, 12))
