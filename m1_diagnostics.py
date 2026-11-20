"""Read-only evaluation of the frozen G0–G2 Best checkpoints."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from mmseg.datasets import PotsdamDataset

from bcia_peft.checkpoints import _read, file_sha256
from bcia_peft.data import PotsdamSlices, PotsdamValidation
from bcia_peft.model import BCIAPeft, IGNORE_INDEX
from bcia_peft.train import data_protocol, initialize_rng, write_log


CLASSES = PotsdamDataset.METAINFO["classes"]
VARIANTS = ("G0_full", "G1_full", "G2_full", "G2_no_rgb_path", "G2_no_dsm_path",
            "G2_no_paths", "G2_bypass", "G2_no_dsm_modality", "G1_no_dsm_modality")
TRAIN_COMMIT = "2b2a68314bbd9694940dfc3855c171563472665f"


def count_pairs(labels, prediction, columns):
    return torch.bincount(labels * columns + prediction, minlength=6 * columns
                          ).reshape(6, columns).cpu().numpy()


def ious(matrix):
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    union = matrix.sum(axis=-2) + matrix.sum(axis=-1) - diagonal
    return np.divide(100 * diagonal, union, out=np.full(union.shape, np.nan), where=union > 0)


def metrics(matrix):
    scores = ious(matrix)
    return dict(mIoU5=float(scores[:5].mean()), mIoU6=float(scores.mean()),
                OA=float(100 * matrix.trace() / matrix.sum()), IoU=dict(zip(CLASSES, scores)))


def scene_bootstrap(first, second, seed=42, draws=5000):
    indices = np.random.default_rng(seed).integers(len(first), size=(draws, len(first)))
    before, after = (ious(matrices[indices].sum(axis=1))[:, :5] for matrices in (first, second))
    delta = (after - before).mean(axis=1)
    if not np.isfinite(delta).all():
        raise ValueError("Undefined IoU in bootstrap sample")
    return dict(seed=seed, draws=draws, scenes=len(first), confidence=0.95,
                lower=float(np.quantile(delta, .025)), median=float(np.median(delta)),
                upper=float(np.quantile(delta, .975)), positive_fraction=float((delta > 0).mean()))


class PredictionStats:
    def __init__(self, device):
        self.channels = torch.zeros(7, 64, device=device, dtype=torch.float64)
        self.patch = torch.zeros(8, device=device, dtype=torch.float64)
        self.minimum, self.maximum = float("inf"), -float("inf")

    def update(self, target, pred, residual, gate, coverage):
        target, pred, residual, gate, q = (value.double() for value in
                                          (target, pred, residual, gate, coverage))
        fields = (target, target.square(), pred, pred.square(), residual,
                  residual.square(), target * pred)
        self.channels += torch.stack([(field * q).sum((0, 2, 3)) for field in fields])
        norm = residual.norm(dim=1, keepdim=True)
        fields = (torch.ones_like(q), gate, gate.square(), gate < .01, gate > .99,
                  norm, norm.square(), norm * gate)
        self.patch += torch.stack([(field * q).sum() for field in fields])
        supported = gate[q > 0]
        if supported.numel():
            self.minimum = min(self.minimum, supported.min().item())
            self.maximum = max(self.maximum, supported.max().item())

    def compute(self):
        weight, gate, gate_sq, low, high, norm, norm_sq, routed = self.patch.cpu().tolist()
        target, energy, pred, pred_sq, error, mse, cross = self.channels.cpu().numpy() / weight
        variance = np.mean(energy - target**2)
        zero = np.mean(energy)
        return dict(q_sum=weight, prediction_mse=float(mse.mean()), target_variance=float(variance),
                    prediction_variance=float(np.mean(pred_sq - pred**2)),
                    prediction_rms=float(np.sqrt(pred_sq.mean())), zero_prediction_mse=float(zero),
                    normalized_mse=float(mse.mean() / variance), mse_over_zero=float(mse.mean() / zero),
                    residual_mean=float(error.mean()), residual_std=float(np.sqrt(mse.mean() - error.mean()**2)),
                    target_prediction_cosine=float(cross.mean() / np.sqrt(zero * pred_sq.mean())),
                    gate_mean=gate / weight, gate_std=float(np.sqrt(max(0, gate_sq / weight -
                                       (gate / weight)**2))), gate_min=self.minimum, gate_max=self.maximum,
                    gate_below_001=low / weight, gate_above_099=high / weight,
                    residual_l2_mean=norm / weight, residual_l2_rms=float(np.sqrt(norm_sq / weight)),
                    routed_l2_mean=routed / weight)


def load_groups(work, backbone, protocol, device):
    sequence = json.loads((work / "sequence.json").read_text(encoding="utf-8"))
    if sequence["commit"] != TRAIN_COMMIT:
        raise ValueError("Unexpected training source")
    backbone_hash = file_sha256(backbone)
    reference = _read(backbone)
    models, evidence = {}, {}
    for group in ("G0", "G1", "G2"):
        config = json.loads(Path(f"configs/formal_{group.lower()}.json").read_text(encoding="utf-8"))
        report = json.loads((work / f"{group}_summary.json").read_text(encoding="utf-8"))
        identity = report["identity"]
        best_row = max(report["validation"], key=lambda row: row["mIoU5"])
        best = report["best"]
        if (identity != sequence["groups"][group] or identity["data_protocol"] != protocol or
                identity["backbone_sha256"] != backbone_hash or identity["config"] != config or
                identity["mode"] != "formal" or best != dict(metric="mIoU5", step=best_row["step"],
                    score=best_row["mIoU5"], file=f"best-{best_row['step']:08d}.pth")):
            raise ValueError(f"Unexpected formal checkpoint identity: {group}")
        path = work / group / best["file"]
        state = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        if (state["identity"] != identity or state["best"] != best or state["step"] != best["step"] or
                not all(torch.equal(state["model"][f"backbone.{key}"], tensor)
                        for key, tensor in reference.items())):
            raise ValueError(f"Checkpoint content mismatch: {group}")
        model = BCIAPeft(backbone, group=group)
        model.load_state_dict(state["model"], strict=True)
        del state
        if models:
            model.backbone = models["G0"].backbone
        models[group] = model.to(device).eval()
        evidence[group] = dict(best=best, sha256=file_sha256(path), validation=best_row)
    return models, evidence


@torch.inference_mode()
def evaluate(work, backbone, root, output, device):
    train, val = PotsdamSlices(root, "train"), PotsdamValidation(root)
    protocol = data_protocol(train, val)
    models, evidence = load_groups(work, backbone, protocol, device)
    scenes = sorted({row["scene"] for row in val.rows})
    matrices = {scene: {name: np.zeros((6, 6), dtype=np.int64) for name in VARIANTS} for scene in scenes}
    pairs = {scene: np.zeros((6, 4), dtype=np.int64) for scene in scenes}
    counts = dict.fromkeys(scenes, 0)
    rgb_stats, dsm_stats = (PredictionStats(device) for _ in range(2))
    for index, row in enumerate(val.rows):
        for asset in ("rgb", "dsm", "valid", "label"):
            if file_sha256(val.root / row[f"{asset}_path"]) != row[f"{asset}_sha256"]:
                raise ValueError(f"Asset fingerprint mismatch: {row['id']} {asset}")
        sample = val[index]
        inputs = tuple(sample[key].unsqueeze(0).to(device) for key in ("rgb", "dsm", "valid"))
        missing = (inputs[0], torch.zeros_like(inputs[1]), torch.zeros_like(inputs[2]))
        label = sample["label"].to(device)
        keep = label != IGNORE_INDEX
        label = label[keep]
        scene = row["scene"]
        counts[scene] += 1
        interventions = (
            ("G0", inputs, {}), ("G1", inputs, {}), ("G2", inputs, {"return_adapter": True}),
            ("G2", inputs, {"suppress": ("rgb",)}), ("G2", inputs, {"suppress": ("dsm",)}),
            ("G2", inputs, {"suppress": ("rgb", "dsm")}), ("G2", inputs, {"bypass": True}),
            ("G2", missing, {}), ("G1", missing, {}),
        )
        paired = {}
        for name, (group, data, kwargs) in zip(VARIANTS, interventions):
            model = models[group]
            value = model(*data, **kwargs)
            if name == "G2_full":
                value, taps = value
                rgb_stats.update(taps.z_rgb, taps.pred_rgb, taps.u_rgb, taps.w_rgb, taps.coverage)
                dsm_stats.update(taps.z_dsm, taps.pred_dsm, taps.u_dsm, taps.w_dsm, taps.coverage)
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite logits at {row['id']} {name}")
            prediction = model.logits_for_tile(value).argmax(1).squeeze(0)[keep]
            matrices[scene][name] += count_pairs(label, prediction, 6)
            if name in ("G1_full", "G2_full"):
                paired[name] = (prediction == label).long()
        pairs[scene] += count_pairs(label, 2 * paired["G1_full"] + paired["G2_full"], 4)
        if (index + 1) % 144 == 0:
            print(f"Evaluated {index + 1}/{len(val)} windows", flush=True)
    totals = {name: metrics(sum(matrices[scene][name] for scene in scenes)) for name in VARIANTS}
    for group, item in evidence.items():
        replay, original = totals[f"{group}_full"], item["validation"]
        differences = {key: replay[key] - original[key] for key in ("mIoU5", "mIoU6", "OA")}
        differences.update({key: replay["IoU"][key] - original["IoU"][key] for key in CLASSES})
        item["replay_delta_pp"] = differences
    before, after = (np.stack([matrices[scene][name] for scene in scenes])
                     for name in ("G1_full", "G2_full"))
    write_log(output, dict(training_commit=TRAIN_COMMIT, checkpoints=evidence, data_protocol=protocol,
                           tiles=len(val), classes=CLASSES, totals=totals,
                           bootstrap=scene_bootstrap(before, after),
                           pair_columns=["both_wrong", "g1_wrong_g2_right", "g1_right_g2_wrong", "both_right"],
                           paired_totals=sum(pairs.values()).tolist(),
                           scenes={scene: dict(tiles=counts[scene], paired=pairs[scene].tolist(),
                                               confusion={name: matrix.tolist() for name, matrix in matrices[scene].items()})
                                   for scene in scenes},
                           prediction={"rgb_to_dsm": dsm_stats.compute(), "dsm_to_rgb": rgb_stats.compute()}))
    for group, item in evidence.items():
        if max(map(abs, item["replay_delta_pp"].values())) > 1e-4:
            raise ValueError(f"Formal replay differs by more than 0.0001 percentage points: {group}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    initialize_rng(42, device)
    evaluate(args.work_dir, Path(os.environ["BCIA_BACKBONE_CHECKPOINT"]),
             Path(os.environ["BCIA_DATA_ROOT"]), args.output, device)


if __name__ == "__main__":
    main()
