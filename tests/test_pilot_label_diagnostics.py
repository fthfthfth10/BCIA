import unittest
from types import SimpleNamespace

import torch
from mmseg.models.losses import CrossEntropyLoss

from tests.pilot_label_diagnostics import compare_ce, label_stats


class PilotLabelDiagnosticsTest(unittest.TestCase):
    def test_sparse_labels_and_padding(self):
        labels = torch.tensor([[[1, 255], [255, 255]]])
        stats = label_stats(labels, (1, 2))
        self.assertEqual(stats["class_counts"], [0, 1, 0, 0, 0, 0])
        self.assertEqual(stats["valid_fraction"], 0.25)
        self.assertEqual(stats["pad_fraction"], 0.5)
        self.assertEqual(stats["content_ignore_fraction"], 0.5)

    def test_mmseg_ce_denominator(self):
        loss = CrossEntropyLoss()
        head = SimpleNamespace(align_corners=False, loss_by_feat=lambda logits, samples: {
            "loss_ce": loss(logits, torch.stack([sample.gt_sem_seg.data[0] for sample in samples]),
                            ignore_index=255)})
        model = SimpleNamespace(decode_head=head)
        logits = torch.zeros(1, 6, 1, 1)
        labels = torch.tensor([[[1, 255], [255, 255]]])
        current, valid = compare_ce(model, logits, labels)
        self.assertAlmostEqual(current * 4, valid)
        self.assertEqual(compare_ce(model, logits, torch.full_like(labels, 255)), (0.0, None))
