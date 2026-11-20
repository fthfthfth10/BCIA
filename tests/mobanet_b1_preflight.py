"""Check the pinned, unmodified MoBaNet ViT-B/14 with FP32 batch 1."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch
from torch.nn import functional as F


UPSTREAM_COMMIT = "77b59b9ff2763ddc8d07065bb7547f23d292d391"
WEIGHT_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pad_for_dinov2(rgb, dsm, valid):
    height, width = rgb.shape[-2:]
    padding = (0, (-width) % 14, 0, (-height) % 14)
    height_input = torch.where(valid & torch.isfinite(dsm), dsm, 0)
    return F.pad(rgb, padding), F.pad(height_input.squeeze(1), padding)


@torch.no_grad()
def probe(source, weights):
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(source), *arguments], text=True).strip()

    if git("rev-parse", "HEAD") != UPSTREAM_COMMIT or git("status", "--porcelain"):
        raise ValueError("Expected an unmodified official MoBaNet checkout at the pinned commit")
    if fingerprint(weights) != WEIGHT_SHA256:
        raise ValueError("Unexpected Meta ViT-B/14 weight fingerprint")
    sys.path.insert(0, str(source))
    from Net import Net
    from Model.models.ImageEncoder.vit.peft_modules import apply_mcrc_mask

    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    runtime = SimpleNamespace(encoder="dinov2_vitb14", dinov2_hub_dir=str(source / "dinov2_hub"),
                              dinov2_ckpt=str(weights), dinov2_strict=True,
                              use_cpia=True, use_dgfm=True, mcrc=True, mcrc_ratio=.25)
    model = Net(runtime_args=runtime)
    pretrained = torch.load(str(weights), map_location="cpu", weights_only=True)
    backbone = model.image_encoder.backbone
    assert set(pretrained) == set(backbone.state_dict())
    assert all(torch.equal(value, pretrained[name]) for name, value in backbone.state_dict().items())
    assert all(not param.requires_grad for param in backbone.parameters())
    assert all(param.requires_grad for param in model.image_encoder.extra_patch_embed.parameters())
    assert all(torch.equal(value, backbone.patch_embed.state_dict()[name]) for name, value in
               model.image_encoder.extra_patch_embed.state_dict().items())
    rgb = torch.randn(1, 3, 512, 512, device="cuda")
    dsm = torch.randn(1, 1, 512, 512, device="cuda")
    valid = torch.ones_like(dsm, dtype=torch.bool)
    valid[..., :16, :16] = False
    padded_rgb, padded_dsm = pad_for_dinov2(rgb, dsm, valid)
    masked = apply_mcrc_mask(padded_rgb.clone(), padded_dsm.clone(), ratio=.25)
    assert torch.equal(masked[0], padded_rgb) and torch.equal(masked[1], padded_dsm)
    model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    logits = model(padded_rgb, padded_dsm, mode="Test")[..., :512, :512]
    assert logits.shape == (1, 6, 512, 512) and torch.isfinite(logits).all()
    torch.cuda.synchronize()
    memory = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
    model.train()
    try:
        model(padded_rgb, padded_dsm, mode="Train")
    except ValueError as error:
        if "Expected more than 1 value per channel" not in str(error):
            raise
        blocker = str(error)
    else:
        raise AssertionError("Expected the pinned PPM batch-1 failure")
    return dict(status="blocked", upstream=UPSTREAM_COMMIT, weight_sha256=WEIGHT_SHA256,
                strict_pretrain_tensors=len(pretrained), tap_indices=model.image_encoder.tap_indices,
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total=sum(p.numel() for p in model.parameters()),
                eval_peak_bytes=memory, eval_logits=list(logits.shape),
                train_forward_error=blocker, mcrc_batch1_corruption=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(probe(args.source.resolve(), args.weights.resolve()), indent=2))


if __name__ == "__main__":
    main()
