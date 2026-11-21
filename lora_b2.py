"""Run the frozen Potsdam v3 B2 LoRA baseline using the existing G1 model."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
from torch import nn

from bcia_peft.checkpoints import _read, file_sha256
from bcia_peft.data import PotsdamSlices, PotsdamValidation
from bcia_peft.model import BCIAPeft
from bcia_peft.train import (BEST_METRIC, best_record, build_optimizer, check_config, data_protocol,
                             initial_hashes, initialize_rng, load_state, prune_previous_best,
                             save_state, segmentation_loss, training_loader, write_log)
from bcia_peft.validation import validate


TARGETS = ("q_proj", "v_proj")
RANK = 4
ALPHA = 4
DROPOUT = 0.0
BACKBONE_SHA256 = "f7224cd45820d310d8ba9777b5c4379589d6d3e44d525596fff5805223bcba9e"
V3_METADATA_SHA256 = "c5b1b911d5de65ff2a58581add340f5bb4c89d2832fdabdd900caabe542236c8"


class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        if not isinstance(base, nn.Linear) or any(param.requires_grad for param in base.parameters()):
            raise ValueError("LoRA requires a frozen Linear")
        self.base = base
        self.lora_a = nn.Linear(base.in_features, RANK, bias=False)
        self.lora_b = nn.Linear(RANK, base.out_features, bias=False)
        nn.init.zeros_(self.lora_b.weight)
        self.scaling = ALPHA / RANK

    def forward(self, inputs):
        return self.base(inputs) + self.scaling * self.lora_b(self.lora_a(inputs))


def inject_lora(model):
    if model.group != "G1" or len(model.backbone.layer) != 24:
        raise ValueError("B2 requires the 24-block G1 backbone")
    for block in model.backbone.layer:
        for name in TARGETS:
            linear = getattr(block.attention, name)
            if (linear.in_features, linear.out_features) != (1024, 1024):
                raise ValueError(f"Unexpected attention projection: {name}")
            setattr(block.attention, name, LoRALinear(linear))
    return model


def lora_modules(model):
    return [getattr(block.attention, name) for block in model.backbone.layer for name in TARGETS]


def check_backbone(model, source):
    tensors = model.backbone.state_dict()
    mapped = {}
    for name, tensor in tensors.items():
        if ".lora_a." in name or ".lora_b." in name:
            continue
        key = name.replace(".base.", ".")
        mapped[key] = tensor
    if set(mapped) != set(source) or not all(torch.equal(tensor.cpu(), source[key])
                                             for key, tensor in mapped.items()):
        raise ValueError("Frozen backbone differs from the pretrained checkpoint")
    for name, parameter in model.backbone.named_parameters():
        is_lora = ".lora_a." in name or ".lora_b." in name
        if parameter.requires_grad != is_lora:
            raise ValueError(f"Unexpected trainability: {name}")


def step_update(model, batch, optimizer, scheduler, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(*(batch[key].to(device) for key in ("rgb", "dsm", "valid")))
    loss = segmentation_loss(model, logits, batch["label"], "cpu", reduction="valid_pixels")
    if not torch.isfinite(logits).all().item() or not loss.isfinite().item():
        raise FloatingPointError("Nonfinite B2 logits or loss")
    loss.backward()
    for name, parameter in model.backbone.named_parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError(f"Gradient in frozen projection/backbone: {name}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(param.grad is None for param in trainable) or not torch.stack(
            [torch.isfinite(param.grad).all() for param in trainable]).all().item():
        raise FloatingPointError("Missing or nonfinite trainable gradients")
    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    optimizer.step()
    if not torch.stack([torch.isfinite(param).all() for param in trainable]).all().item():
        raise FloatingPointError("Nonfinite trainable parameters after AdamW")
    scheduler.step()
    return dict(ce=loss.item(), lr=rates, sample=batch["id"][0])


def preflight(model, batch, identity, work, backbone, device):
    inputs = tuple(batch[key].to(device) for key in ("rgb", "dsm", "valid"))
    model.eval()
    with torch.inference_mode():
        baseline = model(*inputs)
    base_hashes = initial_hashes(model)
    inject_lora(model).to(device)
    with torch.inference_mode():
        adapted = model(*inputs)
    if not torch.equal(baseline, adapted) or base_hashes != initial_hashes(model):
        raise ValueError("B2 initial logits or non-LoRA modules differ from G1")
    del baseline, adapted, inputs
    source = _read(backbone)
    check_backbone(model, source)
    optimizer, scheduler = build_optimizer(model, identity["config"])
    initialize_rng(identity["config"]["seed"], device)
    torch.cuda.reset_peak_memory_stats(device)
    record = step_update(model, batch, optimizer, scheduler, device)
    torch.cuda.synchronize(device)
    record.update(gpu_peak_allocated=torch.cuda.max_memory_allocated(device),
                  gpu_peak_reserved=torch.cuda.max_memory_reserved(device),
                  a_nonzero_grad=sum(layer.lora_a.weight.grad.count_nonzero().item() > 0
                                     for layer in lora_modules(model)),
                  b_nonzero_grad=sum(layer.lora_b.weight.grad.count_nonzero().item() > 0
                                     for layer in lora_modules(model)),
                  b_updated=sum(layer.lora_b.weight.count_nonzero().item() > 0
                                for layer in lora_modules(model)))
    if (record["a_nonzero_grad"] != 0 or record["b_nonzero_grad"] != 48 or
            record["b_updated"] != 48 or not optimizer.state):
        raise ValueError(f"Unexpected first-step LoRA gradients or updates: {record}")
    check_backbone(model, source)
    checkpoint = work / "last.pth"
    save_state(checkpoint, model, optimizer, scheduler, identity, 1, None, device)
    if (load_state(checkpoint, model, optimizer, scheduler, identity) != (1, None) or
            scheduler.last_epoch != 1 or
            any(state["step"].item() != 1 for state in optimizer.state.values())):
        raise ValueError("B2 preflight restore mismatch")
    write_log(work / "preflight.json", record)
    print(json.dumps(record), flush=True)


def run(model, train, val, optimizer, scheduler, identity, work, backbone, device, *, resume):
    config = identity["config"]
    last = work / "last.pth"
    start, best = load_state(last, model, optimizer, scheduler, identity) if resume else (0, None)
    if not 0 <= start <= config["steps"] or scheduler.last_epoch != start:
        raise ValueError("Checkpoint step or scheduler mismatch")
    log_dir = work / "steps"
    log_dir.mkdir(exist_ok=resume)
    began = time.perf_counter()
    for step, batch in enumerate(training_loader(train, config, start), start + 1):
        step_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        row = step_update(model, batch, optimizer, scheduler, device)
        torch.cuda.synchronize(device)
        row.update(step=step, seconds=time.perf_counter() - step_start,
                   gpu_peak_allocated=torch.cuda.max_memory_allocated(device),
                   gpu_peak_reserved=torch.cuda.max_memory_reserved(device))
        previous = best
        if step % config["validate_every"] == 0:
            row["validation"] = validate(model, val, device)
            if any(value is None for value in (row["validation"]["mIoU5"],
                                               row["validation"]["mIoU6"],
                                               row["validation"]["OA"],
                                               *row["validation"]["IoU"].values())):
                raise ValueError("Undefined full validation metric")
            best = best_record(row["validation"], previous, step)
        write_log(log_dir / f"{step:08d}.json", row)
        if best is not previous:
            save_state(work / best["file"], model, optimizer, scheduler, identity, step, best, device)
        if step % config["save_every"] == 0:
            save_state(last, model, optimizer, scheduler, identity, step, best, device)
            if best is not previous:
                prune_previous_best(work, previous, best)
        print(json.dumps(row), flush=True)
    if start < config["steps"]:
        write_log(work / "duration.json", dict(seconds=time.perf_counter() - began, resumed_from=start))
    finalize(model, train, val, identity, work, backbone, device)


@torch.inference_mode()
def finalize(model, train, val, identity, work, backbone, device):
    config = identity["config"]
    steps = config["steps"]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((work / "steps").glob("*.json"))]
    if [row["step"] for row in rows] != list(range(1, steps + 1)):
        raise ValueError("Incomplete formal B2 step logs")
    history, losses = [], []
    for row in rows:
        losses.append(row["ce"])
        if not math.isfinite(row["ce"]):
            raise ValueError("Nonfinite logged loss")
        if "validation" in row:
            history.append(dict(step=row["step"], metrics=row["validation"],
                                mean_ce=statistics.mean(losses), last_ce=row["ce"],
                                lr=row["lr"]["backbone"]))
            losses.clear()
    if [row["step"] for row in history] != list(range(2000, steps + 1, 2000)):
        raise ValueError("Incomplete formal B2 validation logs")
    best_row = max(history, key=lambda row: row["metrics"][BEST_METRIC])
    best = best_record(best_row["metrics"], None, best_row["step"])
    source = _read(backbone)
    hashes = {}
    for name, step in (("last.pth", steps), (best["file"], best["step"])):
        path = work / name
        state = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        if (state["step"] != step or state["identity"] != identity or state["best"] != best or
                state["scheduler"]["last_epoch"] != step or not state["optimizer"]["state"]):
            raise ValueError(f"Unexpected checkpoint: {name}")
        model.load_state_dict(state["model"], strict=True)
        check_backbone(model, source)
        hashes[name] = file_sha256(path)
        del state
    replay = validate(model, val, device)
    if replay != best_row["metrics"] or data_protocol(train, val) != identity["data_protocol"]:
        raise ValueError("Best replay or Potsdam fingerprint mismatch")
    write_log(work / "summary.json", dict(identity=identity, best=best, replay=replay, validation=history,
                                        duration=json.loads((work / "duration.json").read_text(encoding="utf-8")),
                                        checkpoint_sha256=hashes,
                                        gpu_peak_allocated=max(row["gpu_peak_allocated"] for row in rows),
                                        gpu_peak_reserved=max(row["gpu_peak_reserved"] for row in rows)))


def current_commit(*args):
    root = Path.cwd().as_posix()
    return subprocess.check_output(["git", "-c", f"safe.directory={root}", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--commit")
    args = parser.parse_args()
    if (args.preflight and args.resume) or args.work_dir.exists() != args.resume:
        parser.error("Use a new directory or resume an existing formal run")
    if not args.preflight and (not args.commit or current_commit("rev-parse", "HEAD") != args.commit or
                               current_commit("status", "--porcelain")):
        parser.error("Formal training requires a pinned clean commit")
    config = json.loads(Path("configs/lora_b2.json").read_text(encoding="utf-8"))
    reference = json.loads(Path("configs/formal_g1.json").read_text(encoding="utf-8"))
    if config.pop("lora") != dict(rank=RANK, alpha=ALPHA, dropout=DROPOUT, targets=list(TARGETS)) or config != reference:
        raise ValueError("B2 must use the fixed LoRA and G1 protocols")
    check_config(config)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    initialize_rng(config["seed"], device)
    root = Path(os.environ["BCIA_DATA_ROOT"]).resolve(strict=True)
    backbone = Path(os.environ["BCIA_BACKBONE_CHECKPOINT"]).resolve(strict=True)
    if root.name != "potsdam_rgb_dsm_v3" or file_sha256(root / "metadata.json") != V3_METADATA_SHA256:
        raise ValueError("Expected frozen Potsdam RGB+DSM v3 metadata")
    if file_sha256(backbone) != BACKBONE_SHA256:
        raise ValueError("Unexpected ViT-L/16 checkpoint")
    train, val = PotsdamSlices(root, "train"), PotsdamValidation(root)
    protocol = data_protocol(train, val)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected = dict(normalization=metadata["normalization_sha256"], **metadata["manifest_sha256"])
    if (protocol["sha256"]["normalization.json"] != expected["normalization"] or
            any(protocol["sha256"][f"manifests/{split}.jsonl"] != expected[split] for split in ("train", "val"))):
        raise ValueError("Potsdam v3 manifest or normalization mismatch")
    model = BCIAPeft(backbone, group="G1")
    g1_hashes = initial_hashes(model)
    identity = dict(config=config, commit=args.commit, mode="preflight" if args.preflight else "formal",
                    baseline="B2", data_protocol=protocol, backbone_sha256=BACKBONE_SHA256,
                    lora=dict(rank=RANK, alpha=ALPHA, dropout=DROPOUT, targets=list(TARGETS)),
                    g1_initial_sha256=g1_hashes, gpu=torch.cuda.get_device_name(device),
                    torch=str(torch.__version__), cuda=torch.version.cuda, precision="FP32")
    if not args.resume:
        args.work_dir.mkdir(parents=True)
    if args.preflight:
        batch = next(iter(training_loader(train, config, 0, 1)))
        write_log(args.work_dir / "run.json", identity)
        preflight(model.to(device), batch, identity, args.work_dir, backbone, device)
    else:
        inject_lora(model).to(device)
        identity["parameters"] = model.parameter_counts()
        optimizer, scheduler = build_optimizer(model, config)
        if not args.resume:
            write_log(args.work_dir / "run.json", identity)
            initialize_rng(config["seed"], device)
        run(model, train, val, optimizer, scheduler, identity, args.work_dir, backbone, device,
            resume=args.resume)


if __name__ == "__main__":
    main()
