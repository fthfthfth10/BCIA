"""Replay G2 pilot samples with a fixed checkpoint and no optimizer."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from bcia_peft.adapter import prediction_losses
from bcia_peft.data import PotsdamSlices, SynchronizedGeometry
from bcia_peft.model import BCIAPeft, IGNORE_INDEX, NUM_CLASSES, TILE_SIZE
from bcia_peft.train import StepBatchSampler, file_sha256, initialize_rng, segmentation_loss


class ObservedGeometry(SynchronizedGeometry):
    def __call__(self, *inputs):
        state = np.random.get_state()
        extent = int(TILE_SIZE * np.random.uniform(*self.ratio_range))
        np.random.set_state(state)
        self.resized_hw = (extent, extent)
        result = super().__call__(*inputs)
        self.content_hw = result[-1].shape
        return result


def label_stats(labels, content_hw):
    count = labels.numel()
    selected = labels[labels != IGNORE_INDEX]
    if ((selected < 0) | (selected >= NUM_CLASSES)).any():
        raise ValueError("Unexpected mapped class")
    content_count = math.prod(content_hw)
    return dict(valid_labels=selected.numel(), valid_fraction=selected.numel() / count,
                class_counts=torch.bincount(selected, minlength=NUM_CLASSES).tolist(),
                pad_fraction=1 - content_count / count,
                content_ignore_fraction=1 - selected.numel() / content_count)


def compare_ce(model, logits, labels):
    logits = logits.cpu()
    dense = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                          align_corners=model.decode_head.align_corners)
    summed = F.cross_entropy(dense, labels, ignore_index=IGNORE_INDEX, reduction="sum").item()
    count = (labels != IGNORE_INDEX).sum().item()
    current = segmentation_loss(model, logits, labels, "cpu").item()
    valid_mean = summed / count if count else None
    expected = valid_mean * count / labels.numel() if count else 0.0
    if not math.isclose(current, expected, rel_tol=1e-6, abs_tol=1e-8):
        raise AssertionError(f"CE denominator mismatch: {current} vs {expected}")
    return current, valid_mean


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    work = args.pilot_dir
    state = torch.load(str(work / "last.pth"), map_location="cpu", weights_only=True, mmap=True)
    identity = state["identity"]
    config = identity["config"]
    if identity["mode"] != "pilot" or config["group"] != "G2" or config["batch_size"] != 1:
        raise ValueError("Expected a single-sample G2 pilot checkpoint")
    initialize_rng(config["seed"], device)
    geometry = ObservedGeometry()
    dataset = PotsdamSlices(os.environ["BCIA_DATA_ROOT"], "train", geometry=geometry)
    for filename in ("manifests/train.jsonl", "normalization.json"):
        if file_sha256(dataset.root / filename) != identity["data_protocol"]["sha256"][filename]:
            raise ValueError(f"Pilot data fingerprint mismatch: {filename}")
    model = BCIAPeft(checkpoint=os.environ["BCIA_BACKBONE_CHECKPOINT"], group="G2")
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    sampler = StepBatchSampler(len(dataset), 1, config["seed"], 0, state["step"])
    for step, ((index, seed),) in enumerate(sampler, 1):
        record = json.loads((work / "steps" / f"{step:08d}.json").read_text(encoding="utf-8"))
        sample = dataset[index, seed]
        if record["samples"] != [sample["id"]]:
            raise ValueError(f"Sample order mismatch at step {step}")
        row = dataset.rows[index]
        source_path = dataset.root / row["label_path"]
        if file_sha256(source_path) != row["label_sha256"]:
            raise ValueError(f"Source label fingerprint mismatch at step {step}")
        with Image.open(source_path) as image:
            source = np.array(image)
        source_counts = np.bincount(source.ravel(), minlength=NUM_CLASSES + 1).tolist()
        labels = sample["label"].unsqueeze(0)
        logits, output = model(*(sample[key].unsqueeze(0).to(device)
                                 for key in ("rgb", "dsm", "valid")), return_adapter=True)
        current, valid_mean = compare_ce(model, logits, labels)
        mse = {name: value.item() for name, value in prediction_losses(output).items()}
        print(json.dumps(dict(step=step, sample=sample["id"], augmentation_seed=seed,
                              source_label_counts=source_counts, resized_hw=geometry.resized_hw,
                              content_hw=geometry.content_hw, dsm_valid_fraction=output.coverage.mean().item(),
                              historical_ce=record["segmentation"], current_ce=current,
                              valid_mean_ce=valid_mean, auxiliary_mse=mse,
                              **label_stats(labels, geometry.content_hw)), allow_nan=False))


if __name__ == "__main__":
    main()
