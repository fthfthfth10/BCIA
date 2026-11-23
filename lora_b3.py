"""Run the frozen Potsdam v3 B3 LoRA comparison on the original G2 model."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn

from bcia_peft.checkpoints import _read, file_sha256
from bcia_peft.data import PotsdamSlices, PotsdamValidation
from bcia_peft.model import BCIAPeft
from bcia_peft.train import (BEST_METRIC, build_optimizer, check_config,
                             data_protocol, initial_hashes, initialize_rng, load_state, save_state,
                             training_loader, write_log)
from lora_b2 import (ALPHA, BACKBONE_SHA256, DROPOUT, RANK, TARGETS, V3_METADATA_SHA256,
                     LoRALinear, check_backbone, current_commit, lora_modules, run, step_update)


CONFIG = Path("configs/lora_b3.json")
G2_CONFIG = Path("configs/formal_g2.json")
G2_RUN = Path("work_dirs/formal/v3_seed42/G2/run.json")
PREFLIGHT_DIR = Path("work_dirs/baselines/b3_lora_bcia_v3_preflight")
FORMAL_DIR = Path("work_dirs/baselines/b3_lora_bcia_v3_seed42")
LORA = dict(rank=RANK, alpha=ALPHA, dropout=DROPOUT, targets=list(TARGETS))
LORA_PARAMETERS = 393_216


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_config():
    config = read_json(CONFIG)
    if (config.pop("lora") != LORA or LORA != read_json(Path("configs/lora_b2.json"))["lora"] or
            config != read_json(G2_CONFIG) or config["group"] != "G2"):
        raise ValueError("B3 requires the frozen G2 protocol and B2 LoRA settings")
    check_config(config)
    return config


def inject_lora(model):
    if model.group != "G2" or len(model.backbone.layer) != 24:
        raise ValueError("B3 requires the original 24-block G2 backbone")
    projections = [(block.attention, name, getattr(block.attention, name))
                   for block in model.backbone.layer for name in TARGETS]
    if any(not isinstance(layer, nn.Linear) or
           (layer.in_features, layer.out_features) != (1024, 1024) or
           any(param.requires_grad for param in layer.parameters())
           for _, _, layer in projections):
        raise ValueError("B3 requires 48 unwrapped, frozen 1024-dimensional Q/V projections")
    for attention, name, layer in projections:
        setattr(attention, name, LoRALinear(layer))
    return model


def lora_initial_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.backbone.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def reference_data(config, root, backbone):
    reference = read_json(G2_RUN)
    weight_hash = file_sha256(backbone)
    if (reference["mode"] != "formal" or reference["config"] != config or
            reference["best_metric"] != BEST_METRIC or
            weight_hash != BACKBONE_SHA256 or weight_hash != reference["backbone_sha256"]):
        raise ValueError("G2 formal protocol or pretrained checkpoint mismatch")
    if root.name != "potsdam_rgb_dsm_v3":
        raise ValueError("B3 requires Potsdam RGB+DSM v3")
    train, val = PotsdamSlices(root, "train"), PotsdamValidation(root)
    protocol = data_protocol(train, val)
    if (protocol["sha256"]["metadata.json"] != V3_METADATA_SHA256 or
            protocol != reference["data_protocol"]):
        raise ValueError("Potsdam v3 data fingerprint differs from the formal G2 run")
    return reference, train, val, protocol


def preflight_logits(model, batch, device):
    inputs = tuple(batch[key].to(device) for key in ("rgb", "dsm", "valid"))
    with torch.inference_mode():
        logits = model.eval()(*inputs)
    inject_lora(model).to(device)
    with torch.inference_mode():
        adapted = model(*inputs)
    if not torch.equal(logits, adapted) or not logits.isfinite().all().item():
        raise ValueError("B3 initial logits must exactly match finite G2 logits")


def adapter_gradient_norms(model):
    modules = {"dsm_encoder": model.dsm_encoder}
    modules.update({f"bcia.{name}": module for name, module in model.bcia.named_children()})
    return {name: torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(parameter.grad) for parameter in module.parameters()
            ])).item()
            for name, module in modules.items()}


def preflight(model, batches, identity, work, backbone, device):
    config = identity["config"]
    source = _read(backbone)
    check_backbone(model, source)
    optimizer, scheduler = build_optimizer(model, config)
    initialize_rng(config["seed"], device)
    records = []
    layers = lora_modules(model)
    for step, batch in enumerate(batches, 1):
        torch.cuda.reset_peak_memory_stats(device)
        record = step_update(model, batch, optimizer, scheduler, device)
        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        a_grad = sum(layer.lora_a.weight.grad.count_nonzero().item() > 0 for layer in layers)
        b_grad = sum(layer.lora_b.weight.grad.count_nonzero().item() > 0 for layer in layers)
        b_updated = sum(layer.lora_b.weight.count_nonzero().item() > 0 for layer in layers)
        norms = adapter_gradient_norms(model)
        upstream = [norm for name, norm in norms.items() if name != "bcia.output_projection"]
        if ((a_grad, b_grad, b_updated) != (0 if step == 1 else 48, 48, 48) or
                norms["bcia.output_projection"] == 0 or
                (step == 2 and any(norm == 0 for norm in upstream))):
            raise RuntimeError(f"Unexpected B3 gradients at step {step}: "
                               f"LoRA A={a_grad}, B={b_grad}, B updated={b_updated}, norms={norms}")
        record.update(step=step, a_nonzero_grad=a_grad, b_nonzero_grad=b_grad, b_updated=b_updated,
                      gradient_norms=norms, gpu_peak_allocated=peak_allocated,
                      gpu_peak_reserved=peak_reserved)
        records.append(record)
        print(json.dumps(record), flush=True)
    if len(records) != 2:
        raise ValueError("B3 preflight requires two real training batches")
    check_backbone(model, source)
    del source
    checkpoint = work / "last.pth"
    save_state(checkpoint, model, optimizer, scheduler, identity, 2, None, device)
    del optimizer, scheduler
    lora_b = layers[0].lora_b.weight
    expected = lora_b.detach().cpu().clone()
    with torch.no_grad():
        lora_b.zero_()
    optimizer, scheduler = build_optimizer(model, config)
    if (load_state(checkpoint, model, optimizer, scheduler, identity) != (2, None) or
            scheduler.last_epoch != 2 or not torch.equal(lora_b.detach().cpu(), expected) or
            len(optimizer.state) != sum(len(group["params"]) for group in optimizer.param_groups) or
            any(state["step"].item() != 2 for state in optimizer.state.values())):
        raise ValueError("B3 preflight did not restore model and AdamW state at step 2")
    write_log(work / "preflight.json", dict(initial_logits_equal=True, restore_step=2, steps=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--commit")
    args = parser.parse_args()
    work = PREFLIGHT_DIR if args.preflight else FORMAL_DIR
    if (args.preflight and (args.resume or args.commit)) or work.exists() != args.resume:
        parser.error("Use a new preflight/formal directory or resume the existing formal run")
    if not args.preflight and (not args.commit or current_commit("rev-parse", "HEAD") != args.commit or
                               current_commit("rev-parse", "origin/main") != args.commit or
                               current_commit("status", "--porcelain")):
        parser.error("Formal B3 requires a clean commit synchronized with origin/main")

    config = experiment_config()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    initialize_rng(config["seed"], device)
    root = Path(os.environ["BCIA_DATA_ROOT"]).resolve(strict=True)
    backbone = Path(os.environ["BCIA_BACKBONE_CHECKPOINT"]).resolve(strict=True)
    reference, train, val, protocol = reference_data(config, root, backbone)
    model = BCIAPeft(backbone, group="G2")
    g2_hashes = initial_hashes(model)
    if g2_hashes != reference["initial_sha256"] or model.parameter_counts() != reference["parameters"]:
        raise ValueError("G2 initialization fingerprints or parameter counts differ from the formal run")

    if args.preflight:
        batches = iter(training_loader(train, config, 0, 2))
        first = next(batches)
        preflight_logits(model.to(device), first, device)
    else:
        inject_lora(model).to(device)

    parameters = model.parameter_counts()
    expected_backbone = dict(total=reference["parameters"]["backbone"]["total"] + LORA_PARAMETERS,
                             trainable=LORA_PARAMETERS)
    expected_counts = dict(reference["parameters"], backbone=expected_backbone)
    if (parameters != expected_counts or len(lora_modules(model)) != 48 or
            g2_hashes != initial_hashes(model)):
        raise ValueError("Unexpected B3 parameter counts or changes to the initialized G2 modules")
    identity = dict(config=config, baseline="B3", data_protocol=protocol, backbone_sha256=BACKBONE_SHA256,
                    g2_initial_sha256=g2_hashes, lora=LORA, lora_initial_sha256=lora_initial_hash(model),
                    b3_source_sha256=file_sha256(Path(__file__)), gpu=torch.cuda.get_device_name(device),
                    torch=str(torch.__version__), cuda=torch.version.cuda, precision="FP32",
                    parameters=parameters)
    preflight_identity = dict(identity, mode="preflight", commit=None)
    if args.preflight:
        work.mkdir(parents=True)
        write_log(work / "run.json", preflight_identity)
        preflight(model, (first, next(batches)), preflight_identity, work, backbone, device)
        return

    if (not (PREFLIGHT_DIR / "preflight.json").is_file() or
            read_json(PREFLIGHT_DIR / "run.json") != preflight_identity):
        raise ValueError("Formal B3 requires a matching, completed two-step GPU preflight")
    identity.update(mode="formal", commit=args.commit)
    if args.resume:
        if read_json(work / "run.json") != identity:
            raise ValueError("Resume requires the same B3 run identity")
    else:
        work.mkdir(parents=True)
        write_log(work / "run.json", identity)
    optimizer, scheduler = build_optimizer(model, config)
    initialize_rng(config["seed"], device)
    run(model, train, val, optimizer, scheduler, identity, work, backbone, device, resume=args.resume)


if __name__ == "__main__":
    main()
