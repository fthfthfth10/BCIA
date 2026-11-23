import unittest
from types import SimpleNamespace

import torch
from torch import nn

from lora_b2 import LoRALinear, lora_modules
from lora_b3 import G2_CONFIG, LORA_PARAMETERS, experiment_config, inject_lora, read_json


def backbone_model(group, count=24):
    with torch.device("meta"):
        layers = []
        for _ in range(count):
            attention = SimpleNamespace(**{
                name: nn.Linear(1024, 1024).requires_grad_(False)
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            })
            layers.append(SimpleNamespace(attention=attention))
    return SimpleNamespace(group=group, backbone=SimpleNamespace(layer=layers))


class B3LoRATest(unittest.TestCase):
    def test_injects_only_qv_in_all_g2_blocks(self):
        model = backbone_model("G2")
        originals = [(block.attention.q_proj, block.attention.v_proj,
                      block.attention.k_proj, block.attention.o_proj)
                     for block in model.backbone.layer]
        with torch.device("meta"):
            self.assertIs(inject_lora(model), model)
        self.assertEqual(len(lora_modules(model)), 48)
        self.assertEqual(sum(param.numel() for module in lora_modules(model)
                             for param in module.parameters() if param.requires_grad),
                         LORA_PARAMETERS)
        for block, (query, value, key, output) in zip(model.backbone.layer, originals):
            self.assertIs(type(block.attention.q_proj), LoRALinear)
            self.assertIs(type(block.attention.v_proj), LoRALinear)
            self.assertIs(block.attention.q_proj.base, query)
            self.assertIs(block.attention.v_proj.base, value)
            self.assertIs(block.attention.k_proj, key)
            self.assertIs(block.attention.o_proj, output)

    def test_rejects_non_g2_and_invalid_projections_before_mutation(self):
        for model in (backbone_model("G1"), backbone_model("G2", count=23)):
            with self.assertRaisesRegex(ValueError, "24-block G2"):
                inject_lora(model)
        model = backbone_model("G2")
        model.backbone.layer[-1].attention.v_proj.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            inject_lora(model)
        self.assertTrue(all(isinstance(module, nn.Linear) for module in lora_modules(model)))

    def test_uses_unchanged_formal_g2_config(self):
        config = experiment_config()
        self.assertEqual(config, read_json(G2_CONFIG))
        self.assertEqual(config["mse"], {"rgb_to_dsm": 0, "dsm_to_rgb": 0})
