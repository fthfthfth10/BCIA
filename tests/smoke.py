import argparse
import json
from collections import Counter

import torch
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

from bcia_peft.model import BCIAPeft, INSERT_AFTER


def checked(tensor, shape, device, dtype=torch.float32):
    assert tensor.shape == shape, tensor.shape
    assert tensor.dtype == dtype and tensor.device == device, (tensor.dtype, tensor.device)
    assert torch.isfinite(tensor).all().item()


def nonzero_gradient(module):
    gradients = [param.grad for param in module.parameters() if param.grad is not None]
    assert gradients and all(grad.isfinite().all().item() for grad in gradients)
    return any(grad.count_nonzero().item() for grad in gradients)


def run(device, checkpoint=None):
    device = torch.device(device)
    cuda = device.type == "cuda"
    torch.manual_seed(16)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(device)

    model = BCIAPeft(checkpoint=checkpoint, random_init=checkpoint is None).to(device).train()
    assert len(model.backbone.layer) == 24 and INSERT_AFTER == 17
    assert all(not module.training for module in model.backbone.modules())
    assert all(not param.requires_grad for param in model.backbone.parameters())
    for module in (model.dsm_encoder, model.bcia, model.decode_head):
        assert module.training
        assert all(param.requires_grad for param in module.parameters())
    for param in model.parameters():
        assert param.dtype == torch.float32 and param.device == device

    rgb = torch.randn(1, 3, 512, 512, device=device)
    dsm = torch.randn(1, 1, 512, 512, device=device)
    valid = torch.ones_like(dsm, dtype=torch.bool)
    dsm[..., :16, :16] = float("nan")
    dsm[..., 16, 0] = -float("inf")
    dsm[..., :8, 16:32] = float("inf")
    valid[..., :8, 16:32] = False
    checked(rgb, (1, 3, 512, 512), device)
    checked(valid, (1, 1, 512, 512), device, torch.bool)
    assert dsm.shape == valid.shape and dsm.dtype == rgb.dtype and dsm.device == device
    input_tensors = rgb, dsm, valid

    with torch.no_grad():
        baseline, bypassed = model.extract_features(*input_tensors, bypass=True)
    assert bypassed is None

    seen = {}
    calls = Counter()

    def observe_input(name, index=0):
        def record(_, args):
            calls[name] += 1
            seen[name] = args[0][index].detach() if name == "head" else args[index].detach()
        return record

    def after18(_, args, output):
        seen["after18"] = output.detach()

    def record_block(index):
        def record(_, args):
            calls[index] += 1
        return record

    hooks = [block.register_forward_pre_hook(record_block(index))
             for index, block in enumerate(model.backbone.layer)]
    hooks += [
        model.backbone.layer[17].register_forward_hook(after18),
        model.backbone.layer[18].register_forward_pre_hook(observe_input("before19")),
        model.bcia.register_forward_pre_hook(observe_input("adapter")),
        model.decode_head.register_forward_pre_hook(observe_input("head")),
    ]
    for name in ("g_rgb_to_dsm", "g_dsm_to_rgb", "router"):
        hooks.append(getattr(model.bcia, name).register_forward_pre_hook(observe_input(name)))

    if cuda:
        torch.cuda.synchronize(device)
        resident = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        logits, output = model(*input_tensors, return_adapter=True)
    finally:
        for hook in hooks:
            hook.remove()
    checked(logits, (1, 6, 32, 32), device)
    checked(model.logits_for_tile(logits.detach()), (1, 6, 512, 512), device)
    checked(seen["after18"], (1, 1029, 1024), device)
    checked(seen["before19"], (1, 1029, 1024), device)
    checked(seen["adapter"], (1, 1024, 1024), device)
    checked(seen["head"], (1, 1024, 32, 32), device)
    checked(output.residual, (1, 1024, 1024), device)
    for name in ("coverage", "w_rgb", "w_dsm"):
        checked(getattr(output, name), (1, 1, 32, 32), device)
    for name in ("z_rgb", "z_dsm", "pred_rgb", "pred_dsm", "u_rgb", "u_dsm",
                 "dsm_contribution", "mixed"):
        checked(getattr(output, name), (1, 64, 32, 32), device)
    for name, channels in (("g_rgb_to_dsm", 64), ("g_dsm_to_rgb", 64), ("router", 257)):
        checked(seen[name], (1, channels, 32, 32), device)
        assert calls[name] == 1 and seen[name].count_nonzero().item()
    assert calls["adapter"] == 1 and all(calls[index] == 1 for index in range(24))
    assert torch.equal(seen["adapter"], seen["after18"][:, 5:])
    assert torch.equal(seen["after18"][:, :5], seen["before19"][:, :5])
    assert torch.equal(seen["after18"], seen["before19"])
    assert output.residual.count_nonzero().item() == 0
    assert torch.equal(seen["head"], baseline[0])
    assert output.coverage[0, 0, 0, :3].tolist() == [0.0, 0.5, 1.0]
    assert output.coverage[0, 0, 1, 0].item() == 255 / 256
    assert output.dsm_contribution[..., 0, 0].count_nonzero().item() == 0
    assert output.w_dsm[..., 0, 0].count_nonzero().item() == 0
    assert output.w_rgb[..., 0, 0].item() > 0

    labels = torch.randint(0, 6, (1, 512, 512), device=device)
    labels[..., :16, :16] = 255
    sample = SegDataSample(gt_sem_seg=PixelData(data=labels))
    loss = model.decode_head.loss_by_feat(logits, [sample])["loss_ce"]
    assert loss.isfinite().item() and loss.item() > 0
    loss.backward()
    assert all(param.grad is None for param in model.backbone.parameters())
    assert nonzero_gradient(model.decode_head)
    assert nonzero_gradient(model.bcia.output_projection)
    for module in (model.dsm_encoder, model.bcia.rgb_projection, model.bcia.g_rgb_to_dsm,
                   model.bcia.g_dsm_to_rgb, model.bcia.router, model.bcia.mix):
        assert all(param.grad is not None for param in module.parameters())
        assert not nonzero_gradient(module)

    counts = model.parameter_counts()
    counts["all"] = {key: sum(row[key] for row in counts.values())
                     for key in ("total", "trainable")}
    weight_mode = "strict Meta-converted backbone, untrained head" if checkpoint else "random synthetic only"
    report = dict(status="PASS", weights=weight_mode, device=str(device), batch=1,
                  dtype="FP32 (TF32 disabled; no autocast)", zero_residual=True,
                  exact_backbone_bypass=True, prefix_unchanged=True, strict_dsm_gate=True,
                  predictors_and_router_nonzero_inputs=True, segmentation_loss=loss.item(),
                  frozen_grads_absent=True, output_projection_and_head_nonzero_grads=True,
                  first_step_upstream_grads_zero=True, optimizer_updates=0, params=counts)
    if cuda:
        torch.cuda.synchronize(device)
        report["gpu"] = dict(
            name=torch.cuda.get_device_name(device),
            total_MiB=torch.cuda.get_device_properties(device).total_memory / 1024**2,
            pre_forward_allocated_MiB=resident / 1024**2,
            peak_allocated_MiB=torch.cuda.max_memory_allocated(device) / 1024**2,
            peak_reserved_MiB=torch.cuda.max_memory_reserved(device) / 1024**2,
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One synthetic forward/backward; no optimizer")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint", help="Strictly compatible pretrained BCIA backbone")
    args = parser.parse_args()
    print(json.dumps(run(args.device, args.checkpoint), indent=2))
