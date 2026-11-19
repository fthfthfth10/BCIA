"""Read-only real-batch loss and auxiliary-gradient diagnostics."""

import json
import math
import os
import time

import torch

from bcia_peft.adapter import prediction_losses
from bcia_peft.data import PotsdamSlices
from bcia_peft.model import BCIAPeft
from bcia_peft.train import initialize_rng, segmentation_loss, training_loader


def benchmark(model, logits, labels, device, repeats=12):
    torch.cuda.synchronize()
    started = time.perf_counter()
    results = []
    for _ in range(repeats):
        leaf = logits.detach().requires_grad_(True)
        loss = segmentation_loss(model, leaf, labels, device)
        gradient, = torch.autograd.grad(loss, leaf)
        torch.cuda.synchronize()
        results.append((loss.detach().cpu(), gradient.cpu()))
    duration = time.perf_counter() - started
    first_loss, first_grad = results[0]
    if not all(torch.isfinite(loss) and torch.isfinite(grad).all() for loss, grad in results):
        raise FloatingPointError("Nonfinite loss or logits gradient")
    return first_loss, first_grad, dict(
        batches_per_second=repeats / duration,
        repeated_bitwise_equal=all(torch.equal(loss, first_loss) and torch.equal(grad, first_grad)
                                   for loss, grad in results),
        max_repeat_gradient_delta=max((grad - first_grad).abs().max().item() for _, grad in results))


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda")
    initialize_rng(42, device)
    dataset = PotsdamSlices(os.environ["BCIA_DATA_ROOT"], "train")
    batch = next(iter(training_loader(dataset, dict(batch_size=1, seed=42, steps=1, workers=0), 0)))
    model = BCIAPeft(checkpoint=os.environ["BCIA_BACKBONE_CHECKPOINT"], group="G2").to(device).eval()
    logits, output = model(*(batch[key].to(device) for key in ("rgb", "dsm", "valid")),
                           return_adapter=True)
    labels = batch["label"]
    cpu_loss, cpu_grad, cpu = benchmark(model, logits, labels, "cpu")
    gpu_error = None
    try:
        gpu_loss, gpu_grad, gpu = benchmark(model, logits, labels, device)
    except RuntimeError as error:
        if "upsample_bilinear2d_backward" not in str(error) or "deterministic" not in str(error):
            raise
        gpu_error = str(error)
        torch.use_deterministic_algorithms(False)
        try:
            gpu_loss, gpu_grad, gpu = benchmark(model, logits, labels, device)
        finally:
            torch.use_deterministic_algorithms(True)
    delta = gpu_grad - cpu_grad
    loss_report = dict(cpu_loss=cpu_loss.item(), gpu_loss=gpu_loss.item(),
                       loss_abs_delta=(gpu_loss - cpu_loss).abs().item(),
                       gradient_max_abs_delta=delta.abs().max().item(),
                       gradient_relative_l2=delta.norm().item() / cpu_grad.norm().item(),
                       cpu=cpu, gpu=gpu, strict_gpu_error=gpu_error)
    modules = {"dsm_encoder": model.dsm_encoder, "decode_head": model.decode_head}
    modules.update(dict(model.bcia.named_children()))
    named_parameters = [(name, param) for name, module in modules.items()
                        for param in module.parameters()]
    parameters = [param for _, param in named_parameters]

    def norms(loss):
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        result = {name: None for name in modules}
        for (name, _), gradient in zip(named_parameters, gradients):
            if gradient is not None:
                result[name] = math.hypot(result[name] or 0, gradient.norm().item())
                if not math.isfinite(result[name]):
                    raise FloatingPointError(f"Nonfinite gradient in {name}")
        return result

    auxiliary = {}
    direct = prediction_losses(output)
    detached = prediction_losses(output, predictors_only=model.bcia)
    for direction, predictor, input_name in (
            ("rgb_to_dsm", "g_rgb_to_dsm", "rgb_projection"),
            ("dsm_to_rgb", "g_dsm_to_rgb", "dsm_encoder")):
        isolated, joint = norms(detached[direction]), norms(direct[direction])
        assert torch.equal(direct[direction], detached[direction])
        assert {name for name, value in isolated.items() if value is not None} == {predictor}
        assert {name for name, value in joint.items() if value is not None} == {predictor, input_name}
        assert isolated[predictor] > 0 and joint[input_name] > 0
        auxiliary[direction] = dict(mse=direct[direction].item(), G3=isolated, G4=joint)
    print(json.dumps(dict(sample=batch["id"], coverage_mean=output.coverage.mean().item(),
                          loss=loss_report, auxiliary=auxiliary,
                          segmentation_grad_norms=norms(segmentation_loss(model, logits, labels, "cpu"))),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
