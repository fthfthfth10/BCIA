import unittest
from types import SimpleNamespace

import torch
from mmseg.models.losses import CrossEntropyLoss

from tests.pilot_label_diagnostics import compare_ce, label_stats
from bcia_peft.train import segmentation_loss


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
        head = SimpleNamespace(align_corners=False, ignore_index=255, loss_by_feat=lambda logits, samples: {
            "loss_ce": loss(logits, torch.stack([sample.gt_sem_seg.data[0] for sample in samples]),
                            ignore_index=255)})
        model = SimpleNamespace(decode_head=head)
        logits = torch.zeros(1, 6, 1, 1)
        labels = torch.tensor([[[1, 255], [255, 255]]])
        current, valid = compare_ce(model, logits, labels)
        self.assertAlmostEqual(current * 4, valid)
        self.assertEqual(compare_ce(model, logits, torch.full_like(labels, 255)), (0.0, None))

    def test_training_ce_reductions_and_gradients(self):
        loss = CrossEntropyLoss()
        head = SimpleNamespace(align_corners=False, ignore_index=255,
                               loss_by_feat=lambda logits, samples: {"loss_ce": loss(
                                   logits, torch.stack([item.gt_sem_seg.data[0] for item in samples]),
                                   ignore_index=255)})
        model = SimpleNamespace(decode_head=head)
        logits = torch.arange(24, dtype=torch.float32).reshape(1, 6, 2, 2).div(12).requires_grad_()
        labels = torch.tensor([[[1, 255], [4, 255]]])
        for reduction, denominator in (("full_pixels", 4), ("valid_pixels", 2)):
            actual = segmentation_loss(model, logits, labels, "cpu", reduction=reduction)
            expected = torch.nn.functional.cross_entropy(
                logits, labels, ignore_index=255, reduction="sum") / denominator
            torch.testing.assert_close(actual, expected)
            gradient, = torch.autograd.grad(actual, logits)
            reference, = torch.autograd.grad(expected, logits)
            torch.testing.assert_close(gradient, reference)
            self.assertEqual(gradient[..., 1].count_nonzero().item(), 0)
        empty = segmentation_loss(model, logits, torch.full_like(labels, 255), "cpu",
                                  reduction="valid_pixels")
        self.assertEqual(empty.item(), 0)
        self.assertEqual(torch.autograd.grad(empty, logits)[0].count_nonzero().item(), 0)
