import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from torch.utils.data import DataLoader

from .adapter import prediction_losses
from .checkpoints import file_sha256
from .data import PotsdamSlices, PotsdamValidation
from .model import BCIAPeft, GROUPS
from .validation import validate


DIRECTIONS = ("rgb_to_dsm", "dsm_to_rgb")


class StepBatchSampler:
    def __init__(self, count, batch_size, seed, start, stop):
        self.steps_per_epoch = count // batch_size
        if self.steps_per_epoch < 1:
            raise ValueError("Training split must contain at least one complete batch")
        self.count, self.batch_size, self.seed = count, batch_size, seed
        self.start, self.stop = start, stop

    def __iter__(self):
        current_epoch = None
        for step in range(self.start, self.stop):
            epoch, offset = divmod(step, self.steps_per_epoch)
            if current_epoch != epoch:
                generator = torch.Generator().manual_seed((self.seed + epoch) % 2**32)
                order = torch.randperm(self.count, generator=generator).tolist()
                augmentation = np.random.RandomState([self.seed, epoch % 2**32])
                seeds = augmentation.randint(0, 2**32, size=self.count, dtype=np.uint32)
                current_epoch = epoch
            begin = offset * self.batch_size
            yield [(order[pos], int(seeds[pos])) for pos in range(begin, begin + self.batch_size)]

    def __len__(self):
        return self.stop - self.start


def training_loader(dataset, config, start):
    sampler = StepBatchSampler(len(dataset), config["batch_size"], config["seed"],
                               start, config["steps"])
    return DataLoader(dataset, batch_sampler=sampler, num_workers=config["workers"],
                      generator=torch.Generator().manual_seed(config["seed"]))


def data_protocol(train, val):
    train_scenes = {row["scene"] for row in train.rows}
    val_scenes = {row["scene"] for row in val.rows}
    stats = json.loads((train.root / "normalization.json").read_text(encoding="utf-8"))
    if (len(train_scenes), len(val_scenes), len(train), len(val)) != (24, 14, 3456, 2016):
        raise ValueError("Expected the audited 24/14 Potsdam split with 3456/2016 slices")
    if train_scenes & val_scenes or train_scenes != set(stats["scene_ids"]):
        raise ValueError("Unexpected scene overlap or normalization split")
    for dataset in (train, val):
        if len({row["id"] for row in dataset.rows}) != len(dataset):
            raise ValueError("Duplicate sample identifiers")
    files = ("metadata.json", "normalization.json", "manifests/train.jsonl", "manifests/val.jsonl")
    return dict(label="Eroded/noBoundary", train_tiles=len(train), val_tiles=len(val),
                train_scenes=sorted(train_scenes), val_scenes=sorted(val_scenes),
                dsm_mean=stats["mean"], dsm_std=stats["std"],
                sha256={name: file_sha256(train.root / name) for name in files})


def check_config(config):
    if config["group"] not in GROUPS:
        raise ValueError("Expected G0 through G4")
    for key in ("steps", "batch_size", "validate_every", "save_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Set {key} explicitly to a positive integer")
    for key in ("seed", "workers", "warmup_steps"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["seed"] >= 2**32 or config["warmup_steps"] >= config["steps"]:
        raise ValueError("Seed must fit uint32 and warmup must end before the budget")
    for key, lower, upper in (("lr", 0, math.inf), ("weight_decay", 0, math.inf),
                               ("power", 0, math.inf), ("min_lr_ratio", 0, 1),
                               ("warmup_start_ratio", 0, 1)):
        value = config[key]
        if type(value) not in (float, int) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"Invalid {key}")
    if min(config["lr"], config["power"], config["warmup_start_ratio"]) <= 0:
        raise ValueError("lr, power and warmup_start_ratio must be positive")
    weights = config["mse"]
    if set(weights) != set(DIRECTIONS) or any(
            type(value) not in (float, int) or not math.isfinite(value) or value < 0
            for value in weights.values()):
        raise ValueError("Set both MSE coefficients to finite nonnegative values")
    if config["group"] in ("G3", "G4") and not all(weights.values()):
        raise ValueError("G3/G4 require both MSE coefficients to be explicitly positive")
    if config["group"] in ("G0", "G1", "G2") and any(weights.values()):
        raise ValueError("G0/G1/G2 use segmentation supervision only")


def build_optimizer(model, config):
    groups = []
    for name, module in model.named_children():
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if parameters:
            groups.append(dict(name=name, params=parameters))
    optimizer = torch.optim.AdamW(groups, lr=config["lr"], weight_decay=config["weight_decay"])

    def lr_ratio(step):
        warmup = config["warmup_steps"]
        if step < warmup:
            initial = config["warmup_start_ratio"]
            return initial + (1.0 - initial) * step / warmup
        progress = min(1.0, (step - warmup) / (config["steps"] - warmup))
        floor = config["min_lr_ratio"]
        return floor + (1.0 - floor) * (1.0 - progress) ** config["power"]

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_ratio)


def initialize_rng(seed, device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    name, keys, position, gaussian, cached = np.random.get_state()
    return dict(python=random.getstate(), numpy=(name, keys.tolist(), position, gaussian, cached),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_write(path, write):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_log(path, values):
    atomic_write(path, lambda stream: stream.write(
        (json.dumps(values, indent=2, allow_nan=False) + "\n").encode("utf-8")))


def save_state(path, model, optimizer, scheduler, identity, step, best, device):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), identity=identity,
                 step=step, best=best, rng=rng_state(device))
    atomic_write(path, lambda stream: torch.save(state, stream))


def load_state(path, model, optimizer, scheduler, identity):
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if state["identity"] != identity:
        raise ValueError("Resume requires the same configuration, initial weights and data protocol")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    return state["step"], state["best"]


def initial_hashes(model):
    hashes = {}
    for name, module in model.named_children():
        if name == "backbone":
            continue
        digest = hashlib.sha256()
        for key, tensor in module.state_dict().items():
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        hashes[name] = digest.hexdigest()
    return hashes


def train_step(model, batch, optimizer, scheduler, weights, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs = [batch[key].to(device) for key in ("rgb", "dsm", "valid")]
    logits, output = model(*inputs, return_adapter=True)
    labels = batch["label"].cpu()
    logits = logits.cpu()
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                               align_corners=model.decode_head.align_corners)
    samples = [SegDataSample(gt_sem_seg=PixelData(data=label.unsqueeze(0)))
               for label in labels]
    segmentation = model.decode_head.loss_by_feat(logits, samples)["loss_ce"].to(device)
    auxiliary = {}
    total = segmentation
    if model.group in ("G3", "G4"):
        auxiliary = prediction_losses(output, predictors_only=model.bcia if model.group == "G3" else None)
        total = segmentation + sum(weights[key] * loss for key, loss in auxiliary.items())
    if not torch.isfinite(total).item():
        raise FloatingPointError("Nonfinite training loss")
    total.backward()
    learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    optimizer.step()
    scheduler.step()
    return dict(loss=total.item(), segmentation=segmentation.item(),
                mse={key: loss.item() for key, loss in auxiliary.items()},
                lr=learning_rates, samples=list(batch["id"]))


def main():
    parser = argparse.ArgumentParser(description="Run an explicitly configured BCIA mechanism experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resume", action="store_true", help="Resume last.pth from work-dir")
    parser.add_argument("--describe", action="store_true", help="Print run identity without training or writing")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    check_config(config)
    device = torch.device(args.device)
    initialize_rng(config["seed"], device)
    root = Path(os.environ["BCIA_DATA_ROOT"])
    weights = Path(os.environ["BCIA_BACKBONE_CHECKPOINT"])
    train, val = PotsdamSlices(root, "train"), PotsdamValidation(root)
    protocol = data_protocol(train, val)
    backbone_hash = file_sha256(weights)
    model = BCIAPeft(checkpoint=weights, group=config["group"])
    hashes = initial_hashes(model)
    model.to(device)
    optimizer, scheduler = build_optimizer(model, config)
    identity = dict(config=config, device=args.device, torch=str(torch.__version__),
                    device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                    data_protocol=protocol, backbone_sha256=backbone_hash,
                    initial_sha256=hashes, parameters=model.parameter_counts(),
                    parameter_groups={group["name"]: sum(param.numel() for param in group["params"])
                                      for group in optimizer.param_groups})
    if args.describe:
        print(json.dumps(identity, indent=2))
        return
    work = args.work_dir
    start, best = 0, None
    last = work / "last.pth"
    if args.resume:
        start, best = load_state(last, model, optimizer, scheduler, identity)
    else:
        initialize_rng(config["seed"], device)
        work.mkdir(parents=True, exist_ok=False)
        write_log(work / "run.json", identity)
    log_dir = work / "steps"
    log_dir.mkdir(exist_ok=True)
    for step, batch in enumerate(training_loader(train, config, start), start + 1):
        values = train_step(model, batch, optimizer, scheduler, config["mse"], device)
        validation_due = step % config["validate_every"] == 0 or step == config["steps"]
        improved = False
        if validation_due:
            values["validation"] = validate(model, val, device)
            score = values["validation"]["mIoU5"]
            if score is not None and (best is None or score > best["score"]):
                best = dict(step=step, score=score, file=f"best-{step:08d}.pth")
                improved = True
        values["step"] = step
        write_log(log_dir / f"{step:08d}.json", values)
        if improved:
            save_state(work / best["file"], model, optimizer, scheduler, identity, step, best, device)
        if validation_due or step % config["save_every"] == 0:
            save_state(last, model, optimizer, scheduler, identity, step, best, device)
        print(json.dumps(values, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
