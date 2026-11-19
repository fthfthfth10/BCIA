import json
import unittest
from pathlib import Path

from bcia_peft.train import check_config


class FormalProtocolTest(unittest.TestCase):
    def test_frozen_configs(self):
        expected = dict(ce_reduction="valid_pixels", seed=42, batch_size=1, workers=0,
                        steps=40000, validate_every=2000, save_every=2000, lr=1e-4,
                        weight_decay=0.05, warmup_steps=1500, warmup_start_ratio=0.1,
                        power=1.0, min_lr_ratio=0.0, mse=dict(rgb_to_dsm=0, dsm_to_rgb=0))
        root = Path(__file__).resolve().parents[1] / "configs"
        for group in ("G0", "G1", "G2"):
            config = json.loads((root / f"formal_{group.lower()}.json").read_text(encoding="utf-8"))
            check_config(config)
            self.assertEqual(config, dict(expected, group=group))
