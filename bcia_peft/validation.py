import argparse
import json
from pathlib import Path

import torch
from mmseg.evaluation.metrics import IoUMetric
from mmseg.datasets import PotsdamDataset

from .data import PotsdamValidation
from .model import BCIAPeft, IGNORE_INDEX, NUM_CLASSES


class PotsdamMetrics:
    def __init__(self):
        self.counts = torch.zeros(4, NUM_CLASSES, dtype=torch.float64)

    def update(self, prediction, label):
        areas = IoUMetric.intersect_and_union(prediction, label, NUM_CLASSES, IGNORE_INDEX)
        self.counts += torch.stack(areas)

    def compute(self):
        intersection, union, _, label = self.counts
        iou = 100 * intersection / union

        def ratio(value):
            return value.item() if value.isfinite() else None

        return dict(mIoU5=ratio(iou[:5].mean()), mIoU6=ratio(iou.mean()),
                    OA=ratio(100 * intersection.sum() / label.sum()),
                    IoU=dict(zip(PotsdamDataset.METAINFO["classes"], map(ratio, iou))))


@torch.inference_mode()
def validate(model, dataset, device, limit=None):
    model.eval()
    metrics = PotsdamMetrics()
    count = len(dataset) if limit is None else min(limit, len(dataset))
    if count < 1:
        raise ValueError("Validation requires at least one tile")
    for index in range(count):
        sample = dataset[index]
        inputs = [sample[key].unsqueeze(0).to(device) for key in ("rgb", "dsm", "valid")]
        logits = model.logits_for_tile(model(*inputs))
        metrics.update(logits.argmax(1).squeeze(0).cpu(), sample["label"])
    return dict(tiles=count, **metrics.compute())


def main():
    parser = argparse.ArgumentParser(description="Potsdam validation without a training Runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Override config backbone checkpoint")
    parser.add_argument("--data-root", type=Path, help="Override config prepared dataset root")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int, help="Optional validation tile limit")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or (args.config.parent / config["backbone_checkpoint"]
                                     if config["backbone_checkpoint"] else None)
    root = args.data_root or (args.config.parent / config["data_root"] if config["data_root"] else None)
    if checkpoint is None or root is None:
        parser.error("Provide both backbone_checkpoint and data_root via config or CLI")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = PotsdamValidation(root)
    model = BCIAPeft(checkpoint=checkpoint).to(args.device)
    print(json.dumps(dict(head="untrained; interface smoke only",
                          **validate(model, dataset, args.device, args.limit)), indent=2))


if __name__ == "__main__":
    main()
