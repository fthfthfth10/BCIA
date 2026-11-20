"""BCIA adapter and formal runner for the locally pinned MoBaNet baseline."""

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from bcia_peft.checkpoints import file_sha256
from bcia_peft.data import PotsdamSlices, PotsdamValidation
from bcia_peft.model import IGNORE_INDEX, NUM_CLASSES
from bcia_peft.train import (BEST_METRIC, best_record, build_optimizer, data_protocol,
                             initial_hashes,
                             initialize_rng, load_state, prune_previous_best, save_state,
                             training_loader, write_log)
from bcia_peft.validation import validate
from tests.mobanet_b1_preflight import UPSTREAM_COMMIT, WEIGHT_SHA256, pad_for_dinov2


METADATA_SHA256 = "c5b1b911d5de65ff2a58581add340f5bb4c89d2832fdabdd900caabe542236c8"


def initialize_b1(seed, device):
    initialize_rng(seed, device)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_masks(batch, ratio, device):
    selected = torch.rand(batch, device=device) < ratio
    rgb = torch.rand(batch, device=device) < 0.5
    return selected & rgb, selected & ~rgb


class MoBaNetB1(nn.Module):
    def __init__(self, upstream, net, local_mask):
        super().__init__()
        self.net = net
        self.upstream = upstream
        self.local_mask = local_mask
        pool, layers = net.decoder.ppm.stages[0]
        if pool.output_size != 1 or not isinstance(layers[1], nn.BatchNorm2d):
            raise ValueError("Unexpected upstream PPM global branch")
        self.global_bn.eval()
        if not all(parameter.requires_grad for parameter in self.global_bn.parameters()):
            raise ValueError("PPM affine parameters must remain trainable")
        self.last_mask = None

    @property
    def global_bn(self):
        return self.net.decoder.ppm.stages[0][1][1]

    def train(self, mode=True):
        super().train(mode)
        self.global_bn.eval()
        return self

    def _mcrc(self, rgb, dsm, ratio, num_blocks, block_scale, aspect_ratio):
        rgb_mask, dsm_mask = choose_masks(rgb.size(0), ratio, rgb.device)
        self.last_mask = (rgb_mask, dsm_mask)
        self.local_mask(rgb, rgb_mask.nonzero(as_tuple=True)[0], num_blocks, block_scale, aspect_ratio)
        self.local_mask(dsm.unsqueeze(1), dsm_mask.nonzero(as_tuple=True)[0],
                        num_blocks, block_scale, aspect_ratio)
        return rgb, dsm

    def forward(self, rgb, dsm, valid):
        self.last_mask = None
        height, width = rgb.shape[-2:]
        rgb, dsm = pad_for_dinov2(rgb, dsm, valid)
        if self.training:
            with patch.object(self.upstream, "apply_mcrc_mask", self._mcrc):
                result = self.net(rgb, dsm, mode="Train")
            if not isinstance(result, tuple) or len(result) != 3 or self.last_mask is None:
                raise RuntimeError("MoBaNet MCRC training ABI changed")
            return tuple(item[..., :height, :width] for item in result)
        return self.net(rgb, dsm, mode="Test")[..., :height, :width]

    def logits_for_tile(self, logits):
        return logits


def valid_ce(logits, labels):
    valid = (labels != IGNORE_INDEX).sum().clamp_min(1)
    return F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="sum") / valid


def supervised_losses(outputs, labels):
    fused, rgb, dsm = (item.cpu() for item in outputs)
    labels = labels.cpu()
    hard = labels.masked_fill(fused.detach().argmax(dim=1) == labels, IGNORE_INDEX)
    return valid_ce(fused, labels), valid_ce(rgb, hard), valid_ce(dsm, hard)


def get_model(config, device):
    source = Path(os.environ["BCIA_MOBANET_SOURCE"]).resolve(strict=True)
    weights = Path(os.environ["BCIA_MOBANET_WEIGHTS"]).resolve(strict=True)
    if git(source, "rev-parse", "HEAD") != UPSTREAM_COMMIT or git(source, "status", "--porcelain"):
        raise ValueError("MoBaNet checkout must match the pinned, clean upstream tree")
    if file_sha256(weights) != WEIGHT_SHA256:
        raise ValueError("Unexpected DINOv2 weight hash")
    sys.path.insert(0, str(source))
    upstream = importlib.import_module("Net")
    if Path(upstream.__file__).resolve() != source / "Net.py":
        raise ValueError("Wrong MoBaNet module")
    modules = importlib.import_module("Model.models.ImageEncoder.vit.peft_modules")
    args = SimpleNamespace(**config["upstream"], dinov2_hub_dir=str(source / "dinov2_hub"),
                           dinov2_ckpt=str(weights), dinov2_strict=True)
    net = upstream.Net(num_classes=NUM_CLASSES, runtime_args=args)
    backbone = net.image_encoder.backbone
    pretrained = torch.load(str(weights), map_location="cpu", weights_only=True)
    if (set(pretrained) != set(backbone.state_dict()) or
            not all(torch.equal(value, pretrained[name]) for name, value in backbone.state_dict().items()) or
            any(param.requires_grad for param in backbone.parameters())):
        raise ValueError("Unexpected DINOv2 weights or freezing")
    return MoBaNetB1(upstream, net, modules._apply_local_mask).to(device)


def git(root, *args):
    return subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
                                    "-C", str(root), *args], text=True).strip()


def update(model, batch, optimizer, scheduler, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(*(batch[key].to(device) for key in ("rgb", "dsm", "valid")))
    main, rgb, dsm = supervised_losses(outputs, batch["label"])
    loss = main + model.net.mcrc_aux_weight * (rgb + dsm)
    if not loss.isfinite().item():
        raise FloatingPointError("Nonfinite MoBaNet loss")
    loss.backward()
    if any(param.grad is not None for param in model.net.image_encoder.backbone.parameters()):
        raise RuntimeError("Frozen DINOv2 received gradients")
    gradients = [param.grad for param in model.parameters() if param.requires_grad]
    if any(grad is None for grad in gradients) or not torch.stack(
            [torch.isfinite(grad).all() for grad in gradients]).all().item():
        raise FloatingPointError("Missing or nonfinite MoBaNet gradients")
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    if not torch.stack([torch.isfinite(param).all() for param in model.parameters()
                        if param.requires_grad]).all().item():
        raise FloatingPointError("Nonfinite MoBaNet parameters")
    scheduler.step()
    rgb_mask, dsm_mask = model.last_mask
    return dict(ce=main.item(), aux_rgb=rgb.item(), aux_dsm=dsm.item(), loss=loss.item(), lr=lr,
                rgb_masked=int(rgb_mask.sum()), dsm_masked=int(dsm_mask.sum()),
                sample=batch["id"][0])


def preflight(model, train, val, config, identity, work, device, optimizer, scheduler):
    batch = next(iter(training_loader(train, config, 0, 1)))
    before = {key: param.detach().clone() for key, param in model.named_parameters() if param.requires_grad}
    bn_stats = model.global_bn.running_mean.clone(), model.global_bn.running_var.clone()
    torch.cuda.reset_peak_memory_stats(device)
    report = update(model, batch, optimizer, scheduler, device)
    torch.cuda.synchronize(device)
    if (not all(torch.equal(value, current) for value, current in zip(
            bn_stats, (model.global_bn.running_mean, model.global_bn.running_var))) or
            not all(param.grad.count_nonzero().item() > 0 for param in model.global_bn.parameters()) or
            not any(not torch.equal(before[key], param) for key, param in model.named_parameters() if key in before)
            or not optimizer.state):
        raise ValueError("Preflight did not preserve BN statistics or update trainable parameters")
    report.update(gpu_peak_allocated=torch.cuda.max_memory_allocated(device),
                  gpu_peak_reserved=torch.cuda.max_memory_reserved(device))
    last = work / "last.pth"
    save_state(last, model, optimizer, scheduler, identity, 1, None, device)
    restored = load_state(last, model, optimizer, scheduler, identity)
    if restored != (1, None) or scheduler.last_epoch != 1 or not all(
            int(state["step"].item()) == 1 for state in optimizer.state.values()):
        raise ValueError("Preflight checkpoint recovery mismatch")
    report["validation_smoke"] = validate(model, val, device, limit=1)
    write_log(work / "preflight.json", report)
    print(json.dumps(report), flush=True)


def train(model, dataset, val, config, identity, work, device, optimizer, scheduler, *, resume):
    last = work / "last.pth"
    start, best = load_state(last, model, optimizer, scheduler, identity) if resume else (0, None)
    if resume and scheduler.last_epoch != start:
        raise ValueError("Checkpoint scheduler does not match iteration")
    if not 0 <= start <= config["steps"]:
        raise ValueError("Checkpoint iteration is outside the formal budget")
    log_dir = work / "steps"
    log_dir.mkdir(exist_ok=resume)
    began = time.perf_counter()
    for step, batch in enumerate(training_loader(dataset, config, start), start + 1):
        start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        record = update(model, batch, optimizer, scheduler, device)
        previous = best
        if step % config["validate_every"] == 0:
            record["validation"] = validate(model, val, device)
            if any(metric is None for metric in (record["validation"]["mIoU5"],
                                                 record["validation"]["mIoU6"],
                                                 record["validation"]["OA"],
                                                 *record["validation"]["IoU"].values())):
                raise ValueError("Undefined full validation metric")
            best = best_record(record["validation"], previous, step)
        torch.cuda.synchronize(device)
        record.update(step=step, seconds=time.perf_counter() - start_time,
                      gpu_peak_allocated=torch.cuda.max_memory_allocated(device),
                      gpu_peak_reserved=torch.cuda.max_memory_reserved(device))
        write_log(log_dir / f"{step:08d}.json", record)
        if best is not previous:
            save_state(work / best["file"], model, optimizer, scheduler, identity, step, best, device)
        if step % config["save_every"] == 0:
            save_state(last, model, optimizer, scheduler, identity, step, best, device)
            if best is not previous:
                prune_previous_best(work, previous, best)
        print(json.dumps(record), flush=True)
    seconds = time.perf_counter() - began
    if start < config["steps"]:
        write_log(work / "duration.json", dict(seconds=seconds, resumed_from=start))
    finalize(model, val, config, identity, work, device)


def finalize(model, val, config, identity, work, device):
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((work / "steps").glob("*.json"))]
    if [row["step"] for row in rows] != list(range(1, config["steps"] + 1)):
        raise ValueError("Formal step log incomplete")
    history = [row for row in rows if "validation" in row]
    if [row["step"] for row in history] != list(range(2000, config["steps"] + 1, 2000)):
        raise ValueError("Formal validation log incomplete")
    best_row = max(history, key=lambda row: row["validation"][BEST_METRIC])
    best = best_record(best_row["validation"], None, best_row["step"])
    for fname, step in (("last.pth", config["steps"]), (best["file"], best["step"])):
        checkpoint = torch.load(str(work / fname), map_location="cpu", weights_only=True, mmap=True)
        if (checkpoint["identity"] != identity or checkpoint["best"] != best or
                checkpoint["step"] != step or checkpoint["scheduler"]["last_epoch"] != step):
            raise ValueError(f"Unexpected {fname}")
        model.load_state_dict(checkpoint["model"], strict=True)
        source = torch.load(os.environ["BCIA_MOBANET_WEIGHTS"], map_location="cpu", weights_only=True)
        if not all(torch.equal(tensor, source[key]) for key, tensor in
                   model.net.image_encoder.backbone.cpu().state_dict().items()):
            raise ValueError("Frozen DINOv2 changed")
        model.to(device)
        del checkpoint, source
    observed = validate(model, val, device)
    expected = best_row["validation"]
    if observed != expected:
        raise ValueError(f"Best replay differs: expected {expected}, observed {observed}")
    protocol = data_protocol(PotsdamSlices(val.root, "train"), val)
    if protocol != identity["data_protocol"]:
        raise ValueError("Potsdam v3 fingerprint changed by the end of evaluation")
    summary = dict(identity=identity, best=best, replay=observed, validation=history,
                   duration=json.loads((work / "duration.json").read_text(encoding="utf-8")),
                   rgb_masked=sum(row["rgb_masked"] for row in rows),
                   dsm_masked=sum(row["dsm_masked"] for row in rows),
                   gpu_peak_allocated=max(row["gpu_peak_allocated"] for row in rows),
                   gpu_peak_reserved=max(row["gpu_peak_reserved"] for row in rows),
                   checkpoint_sha256={name: file_sha256(work / name) for name in ("last.pth", best["file"])})
    write_log(work / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.preflight and args.resume:
        parser.error("Cannot resume preflight")
    root = Path.cwd()
    if not args.preflight and (not args.commit or git(root, "rev-parse", "HEAD") != args.commit or
                               git(root, "status", "--porcelain")):
        parser.error("Formal training requires a pinned, clean committed tree")
    if args.work_dir.exists() != args.resume:
        parser.error("Use a new work directory or explicitly resume an existing run")
    config = json.loads(Path("configs/mobanet_b1.json").read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    device = torch.device("cuda")
    initialize_b1(config["seed"], device)
    data = Path(os.environ["BCIA_DATA_ROOT"]).resolve()
    if data.name != "potsdam_rgb_dsm_v3" or file_sha256(data / "metadata.json") != METADATA_SHA256:
        raise ValueError("Expected frozen Potsdam v3")
    dataset, val = PotsdamSlices(data, "train"), PotsdamValidation(data)
    protocol = data_protocol(dataset, val)
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    if (any(protocol["sha256"][f"manifests/{split}.jsonl"] != expected
            for split, expected in metadata["manifest_sha256"].items()) or
            protocol["sha256"]["normalization.json"] != metadata["normalization_sha256"]):
        raise ValueError("Potsdam v3 manifest or normalization mismatch")
    model = get_model(config, device)
    optimizer, scheduler = build_optimizer(model, config)
    identity = dict(config=config, mode="preflight" if args.preflight else "formal",
                    commit=args.commit, upstream_commit=UPSTREAM_COMMIT, weights_sha256=WEIGHT_SHA256,
                    data_protocol=protocol, gpu=torch.cuda.get_device_name(device),
                    torch=str(torch.__version__), cuda=torch.version.cuda, python=sys.version.split()[0],
                    determinism="warn_only_with_cudnn_deterministic", precision="FP32",
                    initial_sha256=initial_hashes(model),
                    parameters=sum(param.numel() for param in model.parameters()),
                    trainable=sum(param.numel() for param in model.parameters() if param.requires_grad))
    if not args.resume:
        args.work_dir.mkdir(parents=True)
        write_log(args.work_dir / "run.json", identity)
        initialize_b1(config["seed"], device)
    if args.preflight:
        preflight(model, dataset, val, config, identity, args.work_dir, device, optimizer, scheduler)
    else:
        train(model, dataset, val, config, identity, args.work_dir, device, optimizer, scheduler,
              resume=args.resume)


if __name__ == "__main__":
    main()
