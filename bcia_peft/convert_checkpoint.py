import argparse
import json
from pathlib import Path

import torch

from .checkpoints import _meta_spec, _read, _spec, file_sha256, inspect_state
from .model import make_backbone


META_SHA256 = "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"

EMBEDDINGS = {
    "cls_token": "embeddings.cls_token",
    "storage_tokens": "embeddings.register_tokens",
    "mask_token": "embeddings.mask_token",
    "patch_embed.proj.weight": "embeddings.patch_embeddings.weight",
    "patch_embed.proj.bias": "embeddings.patch_embeddings.bias",
    "norm.weight": "norm.weight",
    "norm.bias": "norm.bias",
}

BLOCK = {
    "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
    "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
    "attn.proj.weight": "attention.o_proj.weight",
    "attn.proj.bias": "attention.o_proj.bias",
    "ls1.gamma": "layer_scale1.lambda1", "ls2.gamma": "layer_scale2.lambda1",
    "mlp.fc1.weight": "mlp.up_proj.weight", "mlp.fc1.bias": "mlp.up_proj.bias",
    "mlp.fc2.weight": "mlp.down_proj.weight", "mlp.fc2.bias": "mlp.down_proj.bias",
}


def split_qkv(weight, bias, mask):
    if not ((mask == 0) | (mask == 1)).all().item():
        raise ValueError("Unsupported non-binary QKV bias mask")
    query, key, value = (bias * mask).chunk(3)
    if key.count_nonzero().item():
        raise ValueError("Nonzero effective K bias cannot load into this backbone")
    return weight.chunk(3), (query, value)


def map_native(state):
    audit = inspect_state(state, _meta_spec())
    if not audit["compatible"]:
        raise ValueError("Invalid Meta checkpoint: " + json.dumps(audit))

    mapped = {target: state[source] for source, target in EMBEDDINGS.items()}
    mapped["embeddings.mask_token"] = state["mask_token"].unsqueeze(1)
    mapped["rope_embeddings.periods"] = state["rope_embed.periods"].float()
    legacy = {key: tensor for key, tensor in mapped.items()
              if key not in ("norm.weight", "norm.bias", "rope_embeddings.periods")}
    for index in range(24):
        source, target = f"blocks.{index}.", f"layer.{index}."
        block = {target + dst: state[source + src] for src, dst in BLOCK.items()}
        weights, (query_bias, value_bias) = split_qkv(
            state[source + "attn.qkv.weight"], state[source + "attn.qkv.bias"],
            state[source + "attn.qkv.bias_mask"],
        )
        for projection, tensor in zip(("q", "k", "v"), weights):
            block[target + f"attention.{projection}_proj.weight"] = tensor
        for projection, old_bias, effective in zip(
                ("q", "v"), state[source + "attn.qkv.bias"].chunk(3)[::2],
                (query_bias, value_bias)):
            key = target + f"attention.{projection}_proj.bias"
            legacy[key] = old_bias
            block[key] = effective
        mapped.update(block)
        legacy.update({key: tensor for key, tensor in block.items() if key not in legacy})
    return mapped, legacy


def convert(native_path, legacy_path, output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if not output_path.parent.is_dir():
        raise ValueError("Create the output directory before conversion")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    original_hash = file_sha256(native_path)
    if original_hash != META_SHA256:
        raise ValueError("Input is not the audited official ViT-L/16 LVD1689M checkpoint")
    native = _read(native_path)
    mapped, legacy_expected = map_native(native)
    legacy = _read(legacy_path)
    legacy_audit = inspect_state(legacy, _spec(legacy_expected))
    if not legacy_audit["compatible"]:
        raise ValueError("Legacy checkpoint structure differs: " + json.dumps(legacy_audit))
    mismatches = [key for key in legacy_expected if not torch.equal(legacy[key], legacy_expected[key])]
    if mismatches:
        raise ValueError("Legacy checkpoint differs from the native weight mapping: " + str(mismatches))
    with torch.device("meta"):
        backbone = make_backbone()
    audit = inspect_state(mapped, _spec(backbone.state_dict()))
    if not audit["compatible"]:
        raise ValueError("Mapped checkpoint is not strictly compatible: " + json.dumps(audit))
    backbone.load_state_dict(mapped, strict=True, assign=True)
    for key, tensor in backbone.state_dict().items():
        if not torch.equal(tensor, mapped[key]):
            raise RuntimeError(f"Strict round trip changed {key}")
    with output_path.open("xb") as stream:
        torch.save(mapped, stream)
    return {
        "meta_sha256": original_hash,
        "legacy_sha256": file_sha256(legacy_path),
        "new_sha256": file_sha256(output_path),
        "legacy_exact_tensor_matches": len(legacy_expected),
        "raw_qkv_mask_nonzeros": sum(native[f"blocks.{index}.attn.qkv.bias_mask"]
                                     .count_nonzero().item() for index in range(24)),
        "new_state": audit,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert the audited Meta LVD1689M checkpoint")
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--msre", required=True, type=Path, help="Read-only legacy comparison file")
    parser.add_argument("--output", required=True, type=Path, help="New file; must not exist")
    args = parser.parse_args()
    print(json.dumps(convert(args.meta, args.msre, args.output), indent=2))


if __name__ == "__main__":
    main()
