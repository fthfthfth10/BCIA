import argparse
import hashlib
import json

import torch


def _read(path):
    state = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor)
                                              for value in state.values()):
        raise ValueError("Expected a flat tensor state dict")
    return state


def _spec(state):
    return {key: (tensor.shape, tensor.dtype) for key, tensor in state.items()}


def _meta_spec():
    fp32 = torch.float32
    spec = {
        "cls_token": ((1, 1, 1024), fp32),
        "storage_tokens": ((1, 4, 1024), fp32),
        "mask_token": ((1, 1024), fp32),
        "patch_embed.proj.weight": ((1024, 3, 16, 16), fp32),
        "patch_embed.proj.bias": ((1024,), fp32),
        "rope_embed.periods": ((16,), torch.bfloat16),
        "norm.weight": ((1024,), fp32),
        "norm.bias": ((1024,), fp32),
    }
    block = {
        "norm1.weight": (1024,), "norm1.bias": (1024,),
        "norm2.weight": (1024,), "norm2.bias": (1024,),
        "attn.qkv.weight": (3072, 1024), "attn.qkv.bias": (3072,),
        "attn.qkv.bias_mask": (3072,), "attn.proj.weight": (1024, 1024),
        "attn.proj.bias": (1024,), "ls1.gamma": (1024,), "ls2.gamma": (1024,),
        "mlp.fc1.weight": (4096, 1024), "mlp.fc1.bias": (4096,),
        "mlp.fc2.weight": (1024, 4096), "mlp.fc2.bias": (1024,),
    }
    spec.update({f"blocks.{index}.{key}": (shape, fp32)
                 for index in range(24) for key, shape in block.items()})
    return spec


def inspect_state(state, expected):
    common = sorted(state.keys() & expected.keys())
    result = dict(
        tensors=len(state),
        expected_tensors=len(expected),
        missing=sorted(expected.keys() - state.keys()),
        unexpected=sorted(state.keys() - expected.keys()),
        shape_mismatch=[key for key in common if state[key].shape != expected[key][0]],
        dtype_mismatch=[key for key in common if state[key].dtype != expected[key][1]],
        nonfinite=[key for key, value in sorted(state.items())
                   if not torch.isfinite(value).all().item()],
    )
    norm_scales = [key for key in common if key.endswith(
        ("norm.weight", "norm1.weight", "norm2.weight"))]
    result["zero_layernorm_scales"] = [key for key in norm_scales
                                        if not torch.count_nonzero(state[key]).item()]
    result["invalid_rope_periods"] = [
        key for key in ("rope_embed.periods", "rope_embeddings.periods")
        if key in state and not (state[key] > 0).all().item()
    ]
    result["compatible"] = not any(value for value in result.values()
                                   if isinstance(value, list))
    result["checked_layernorm_scales"] = len(norm_scales)
    result["final_layernorm"] = {
        key: dict(shape=list(state[key].shape), dtype=str(state[key].dtype),
                  max_abs=state[key].abs().max().item())
        for key in ("norm.weight", "norm.bias") if key in state and state[key].numel()
    }
    return result


def load_backbone_checkpoint(backbone, path):
    state = _read(path)
    audit = inspect_state(state, _spec(backbone.state_dict()))
    if not audit["compatible"]:
        raise ValueError("Incompatible MsRE DINOv3 checkpoint; no weights loaded: "
                         + json.dumps(audit, sort_keys=True))
    backbone.load_state_dict(state, strict=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_file(path, expected):
    return dict(sha256=file_sha256(path), **inspect_state(_read(path), expected))


def main():
    from .model import make_backbone

    parser = argparse.ArgumentParser(description="Read-only DINOv3 checkpoint audit")
    parser.add_argument("--meta", required=True, help="Unmodified Meta ViT-L/16 LVD1689M file")
    parser.add_argument("--msre", required=True, help="Existing MsRE-converted file")
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    with torch.device("meta"):
        expected = _spec(make_backbone().state_dict())
    meta_result = audit_file(args.meta, _meta_spec())
    msre_result = audit_file(args.msre, expected)
    report = {"meta_native_layout": meta_result, "msre_layout": msre_result}
    print(json.dumps(report, indent=2))
    return 0 if meta_result["compatible"] and msre_result["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
