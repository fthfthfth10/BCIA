"""Check a completed formal run and export its small numerical summary."""

import argparse
import json
import math
from pathlib import Path
import statistics

import torch

from bcia_peft.checkpoints import _read, file_sha256
from bcia_peft.model import BCIAPeft
from bcia_peft.train import build_optimizer, load_state


def review(work, backbone, group, config):
    identity = json.loads((work / "run.json").read_text(encoding="utf-8"))
    if identity["config"] != config or identity["mode"] != "formal" or identity["best_metric"] != "mIoU5":
        raise ValueError("Unexpected formal run identity")
    if file_sha256(backbone) != identity["backbone_sha256"]:
        raise ValueError("Backbone fingerprint differs from run identity")
    steps = config["steps"]
    records = sorted((work / "steps").glob("*.json"))
    if len(records) != steps:
        raise ValueError(f"Expected {steps} completed steps, found {len(records)}")
    validation = []
    losses, rates = [], []
    for step, path in enumerate(records, 1):
        row = json.loads(path.read_text(encoding="utf-8"))
        lr_values = set(row["lr"].values())
        if (row["step"] != step or row["mse"] or len(lr_values) != 1 or
                not math.isfinite(row["segmentation"]) or len(row["samples"]) != 1):
            raise ValueError(f"Unexpected step log: {path.name}")
        offset, warmup = step - 1, config["warmup_steps"]
        ratio = 0.1 + 0.9 * offset / warmup if offset < warmup else 1 - (offset - warmup) / (steps - warmup)
        if not math.isclose(next(iter(lr_values)), config["lr"] * ratio, rel_tol=1e-10):
            raise ValueError(f"Unexpected learning rate at step {step}")
        losses.append(row["segmentation"])
        rates.append(next(iter(lr_values)))
        if step % config["validate_every"] == 0:
            metrics = row["validation"]
            if (metrics["tiles"] != 2016 or len(metrics["IoU"]) != 6 or
                    not all(isinstance(value, (float, int)) and math.isfinite(value)
                            for value in [metrics["mIoU5"], metrics["mIoU6"], metrics["OA"],
                                          *metrics["IoU"].values()])):
                raise ValueError(f"Invalid validation at step {step}")
            validation.append(dict(step=step, **metrics, mean_train_ce=statistics.mean(losses),
                                   min_train_ce=min(losses), max_train_ce=max(losses),
                                   first_lr=rates[0], last_lr=rates[-1]))
            losses.clear()
            rates.clear()
        elif "validation" in row:
            raise ValueError(f"Unexpected validation at step {step}")
    best = max(validation, key=lambda row: row["mIoU5"])
    expected = dict(metric="mIoU5", score=best["mIoU5"], step=best["step"],
                    file=f"best-{best['step']:08d}.pth")
    if {path.name for path in work.glob("best-*.pth")} != {expected["file"]}:
        raise ValueError("Unexpected Best filenames")
    model = BCIAPeft(backbone, group=group)
    reference = _read(backbone)
    for path, step in ((work / "last.pth", steps), (work / expected["file"], best["step"])):
        state = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        if (state["identity"] != identity or state["step"] != step or state["best"] != expected or
                state["scheduler"]["last_epoch"] != step or not state["optimizer"]["state"]):
            raise ValueError(f"Invalid checkpoint record: {path.name}")
        model.load_state_dict(state["model"], strict=True)
        if not all(torch.equal(state["model"][f"backbone.{key}"], tensor)
                   for key, tensor in reference.items()):
            raise ValueError(f"Frozen backbone changed in {path.name}")
        del state
    optimizer, scheduler = build_optimizer(model, config)
    restored_step, restored_best = load_state(work / "last.pth", model, optimizer, scheduler, identity)
    if restored_step != steps or restored_best != expected or scheduler.last_epoch != steps:
        raise ValueError("Last checkpoint recovery differs from completed run")
    return dict(group=group, best=expected, last_step=steps, validation=validation,
                identity=identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(review(args.work_dir, args.backbone, config["group"], config),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
