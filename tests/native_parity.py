import argparse
import json
import math

import torch
from torch.nn import functional as F

from bcia_peft.checkpoints import _read
from bcia_peft.model import make_backbone


def native_rope(periods, size):
    axis = 2 * (torch.arange(0.5, size, dtype=torch.float32) / size) - 1
    coords = torch.cartesian_prod(axis, axis)
    angles = (2 * math.pi * coords[..., None] / periods).flatten(1).repeat(1, 2)
    return angles.cos(), angles.sin()


def native_block(tokens, state, rope):
    norm = F.layer_norm(tokens, (1024,), state["norm1.weight"], state["norm1.bias"], 1e-5)
    qkv = F.linear(norm, state["attn.qkv.weight"], state["attn.qkv.bias"] * state["attn.qkv.bias_mask"])
    query, key, value = qkv.reshape(1, -1, 3, 16, 64).transpose(1, 3).unbind(2)

    def rotate(tensor):
        prefix, spatial = tensor[:, :, :5], tensor[:, :, 5:]
        left, right = spatial.chunk(2, dim=-1)
        return torch.cat((prefix, spatial * rope[0] + torch.cat((-right, left), -1) * rope[1]), 2)

    attention = F.scaled_dot_product_attention(rotate(query), rotate(key), value)
    projected = F.linear(attention.transpose(1, 2).reshape_as(tokens),
                         state["attn.proj.weight"], state["attn.proj.bias"])
    tokens = tokens + projected * state["ls1.gamma"]
    norm = F.layer_norm(tokens, (1024,), state["norm2.weight"], state["norm2.bias"], 1e-5)
    mlp = F.gelu(F.linear(norm, state["mlp.fc1.weight"], state["mlp.fc1.bias"]))
    return tokens + F.linear(mlp, state["mlp.fc2.weight"], state["mlp.fc2.bias"]) * state["ls2.gamma"]


@torch.inference_mode()
def run(meta, converted):
    torch.set_num_threads(4)
    torch.manual_seed(23)
    native = _read(meta)
    with torch.device("meta"):
        target = make_backbone()
    target.load_state_dict(_read(converted), strict=True, assign=True)
    for size in (2, 32):
        rope = native_rope(native["rope_embed.periods"], size)
        actual = target.rope_embeddings(torch.empty(1, 3, 16 * size, 16 * size))
        for reference, value in zip(rope, actual):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
    rgb = torch.randn(1, 3, 32, 32)
    patches = F.conv2d(rgb, native["patch_embed.proj.weight"],
                       native["patch_embed.proj.bias"], stride=16)
    tokens = torch.cat((native["cls_token"], native["storage_tokens"], patches.flatten(2).transpose(1, 2)), 1)
    torch.testing.assert_close(target.embeddings(rgb), tokens, atol=0, rtol=0)
    rope = native_rope(native["rope_embed.periods"], 2)
    local_error = []
    for index, layer in enumerate(target.layer):
        prefix = f"blocks.{index}."
        state = {key[len(prefix):]: val for key, val in native.items() if key.startswith(prefix)}
        expected = native_block(tokens, state, rope)
        actual = layer(tokens, position_embeddings=rope)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=2e-5)
        local_error.append((actual - expected).abs().max().item())
        tokens = expected
    expected = F.layer_norm(tokens, (1024,), native["norm.weight"], native["norm.bias"], 1e-5)
    actual = target(rgb)[0].flatten(2).transpose(1, 2)
    torch.testing.assert_close(actual, expected[:, 5:], atol=3e-4, rtol=5e-5)
    print(json.dumps(dict(status="PASS", rope_grids_exact=[2, 32], blocks=24,
                          max_single_block_error=max(local_error),
                          max_final_patch_error=(actual - expected[:, 5:]).abs().max().item()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU fused-native numerical reference for ViT-L/16")
    parser.add_argument("--meta", required=True)
    parser.add_argument("--converted", required=True)
    args = parser.parse_args()
    run(args.meta, args.converted)
