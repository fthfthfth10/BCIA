"""Frozen G2 feature extraction and train-only conditional/semantic probes."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from bcia_peft.checkpoints import _read, file_sha256
from bcia_peft.data import PotsdamSlices, PotsdamValidation
from bcia_peft.model import BCIAPeft, IGNORE_INDEX, NUM_CLASSES
from bcia_peft.train import data_protocol, initialize_rng, write_log
from bcia_peft.validation import PotsdamMetrics
from m1_diagnostics import CLASSES, TRAIN_COMMIT

VECTOR_FIELDS = ("z_rgb", "z_dsm", "pred_rgb", "pred_dsm", "u_rgb", "u_dsm")
MAP_FIELDS = ("w_rgb", "w_dsm", "coverage")
FIELDS = (*VECTOR_FIELDS, *MAP_FIELDS, "patch_label", "label_pixels", "g2_correct_pixels")
FORMAL = Path("work_dirs/formal/v3_seed42")
M1 = Path("M1_SCENE_CONFUSION.json")
ASSETS = ("rgb", "dsm", "valid", "label")
SEED = 42
GRID, PATCH, WIDTH = 32, 16, 64


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def patch_sum(pixels):
    h, w = pixels.shape
    return pixels.reshape(h // PATCH, PATCH, w // PATCH, PATCH).sum((1, 3))


def majority_labels(label, prediction):
    """Exclude ignore pixels and ambiguous or empty patches."""
    keep = label != IGNORE_INDEX
    counts = torch.stack([patch_sum(label == k) for k in range(NUM_CLASSES)])
    most, label_id = counts.max(0)
    valid = patch_sum(keep)
    label_id = torch.where(2 * most > valid, label_id, IGNORE_INDEX)
    correct = patch_sum((prediction == label) & keep)
    return label_id.to(torch.uint8), valid.to(torch.int16), correct.to(torch.int16)


def g2_reference(protocol, backbone):
    run = read_json(FORMAL / "G2/run.json")
    summary = read_json(FORMAL / "G2_summary.json")
    m1 = read_json(M1)
    config = read_json(Path("configs/formal_g2.json"))
    best = summary["best"]
    expected = m1["checkpoints"]["G2"]
    sha256 = file_sha256(backbone)
    if (run != summary["identity"] or run["config"] != config or run["mode"] != "formal" or
            config["group"] != "G2" or any(config["mse"].values()) or
            m1["training_commit"] != TRAIN_COMMIT or m1["data_protocol"] != protocol or
            run["data_protocol"] != protocol or run["backbone_sha256"] != sha256 or
            best != expected["best"] or best["step"] != 40000 or best["metric"] != "mIoU5"):
        raise ValueError("G2 Best, data or DINOv3 identity mismatch")
    path = FORMAL / "G2" / best["file"]
    checkpoint_hash = file_sha256(path)
    if checkpoint_hash != expected["sha256"]:
        raise ValueError("G2 checkpoint SHA-256 differs from M1")
    state = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    if state["identity"] != run or state["best"] != best or state["step"] != best["step"]:
        raise ValueError("G2 checkpoint contents do not match the formal run")
    reference = _read(backbone)
    source_state = {key[9:]: value for key, value in state["model"].items() if key.startswith("backbone.")}
    if set(source_state) != set(reference) or any(
            not torch.equal(value, reference[key]) for key, value in source_state.items()):
        raise ValueError("Frozen DINOv3 differs from the original weights")
    model = BCIAPeft(backbone, group="G2")
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False).eval()
    metrics = {k: expected["validation"][k] for k in ("tiles", "mIoU5", "mIoU6", "OA", "IoU")}
    return model, dict(training_commit=TRAIN_COMMIT, checkpoint_sha256=checkpoint_hash,
                       best=best, backbone_sha256=sha256, official_validation=metrics)


def make_cache(path, count):
    dtype = {field: np.float32 for field in (*VECTOR_FIELDS, *MAP_FIELDS)}
    dtype.update(patch_label=np.uint8, label_pixels=np.int16, g2_correct_pixels=np.int16)
    return {field: np.lib.format.open_memmap(
                path / f"{field}.npy", mode="w+", dtype=dtype[field],
                shape=(count, WIDTH, GRID, GRID) if field in VECTOR_FIELDS else (count, GRID, GRID))
            for field in FIELDS}


@torch.inference_mode()
def extract_split(model, data, work, device):
    work.mkdir()
    cache = make_cache(work, len(data))
    windows = []
    evaluator = PotsdamMetrics() if data.split == "val" else None
    for index, row in enumerate(data.rows):
        for asset in ASSETS:
            if file_sha256(data.root / row[f"{asset}_path"]) != row[f"{asset}_sha256"]:
                raise ValueError(f"Incorrect {asset} SHA-256 at {data.split} / {row['id']}")
        sample = data[index]
        if sample["id"] != row["id"]:
            raise ValueError("Feature order differs from the manifest")
        inputs = [sample[key].unsqueeze(0).to(device) for key in ASSETS[:3]]
        logits, features = model(*inputs, return_adapter=True)
        if any(not tensor.isfinite().all().item() for tensor in (logits, *features)):
            raise FloatingPointError(f"Nonfinite G2 feature or logits: {row['id']}")
        prediction = model.logits_for_tile(logits).argmax(1).squeeze(0).cpu()
        if evaluator is not None:
            evaluator.update(prediction, sample["label"])
        for field in (*VECTOR_FIELDS, *MAP_FIELDS):
            tensor = getattr(features, field)
            if tensor.dtype != torch.float32:
                raise TypeError(f"Expected FP32 {field}")
            cache[field][index] = (tensor[0, 0] if field in MAP_FIELDS else tensor[0]).cpu().numpy()
        derived = majority_labels(sample["label"], prediction)
        for field, value in zip(FIELDS[-3:], derived):
            cache[field][index] = value.numpy()
        windows.append({k: row[k] for k in ("id", "scene", "xyxy", *(f"{asset}_sha256" for asset in ASSETS))})
        if (index + 1) % 144 == 0:
            print(f"Extracted {data.split}: {index + 1}/{len(data)} windows", flush=True)

    for array in cache.values():
        array.flush()
    del array, cache
    write_log(work / "windows.json", windows)
    hashes = {file.name: file_sha256(file) for file in sorted(work.iterdir())}
    return dict(windows=len(data), sha256=hashes,
                replay=dict(tiles=len(data), **evaluator.compute()) if evaluator is not None else None)


def validation_delta(actual, expected):
    if actual["tiles"] != expected["tiles"] or list(actual["IoU"]) != list(expected["IoU"]):
        raise ValueError("Invalid G2 validation replay count or class order")
    keys = ("mIoU5", "mIoU6", "OA")
    return {**{key: actual[key] - expected[key] for key in keys},
            **{key: actual["IoU"][key] - expected["IoU"][key] for key in expected["IoU"]}}


def extract(root, backbone, work, device, *, reuse_validation=False):
    train = PotsdamSlices(root, "train", photometric=False)
    train.geometry = None
    val = PotsdamValidation(root)
    protocol = data_protocol(train, val)
    model, evidence = g2_reference(protocol, backbone)
    if reuse_validation:
        if (work / "train").exists():
            raise ValueError("Refusing to overwrite an existing train cache")
        result = read_json(work / "metadata.json")
        if result["model"] != evidence or result["data_protocol"] != protocol:
            raise ValueError("Validation cache identity differs from this G2 run")
        val_result = result["splits"]["val"]
        cached = CacheSplit(work / "val", val_result, protocol["val_scenes"])
        if [row["id"] for row in cached.windows] != [row["id"] for row in val.rows]:
            raise ValueError("Validation cache order mismatch")
        del cached
    else:
        work.mkdir(parents=True)
        result = dict(data_protocol=protocol, model=evidence, precision="FP32", tta=False,
                      classes=CLASSES, seed=SEED, torch=str(torch.__version__),
                      device=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                      splits={})
        val_result = extract_split(model.to(device), val, work / "val", device)
    delta = validation_delta(val_result["replay"], evidence["official_validation"])
    if max(map(abs, delta.values())) > 1e-4:
        write_log(work / "validation_mismatch.json", val_result)
        raise ValueError("Frozen G2 replay exceeds the M1 tolerance of 0.0001 percentage points")
    val_result["delta_pp"] = delta
    result["splits"]["val"] = val_result
    write_log(work / "metadata.json", result)
    print("Validation cache verified against the formal G2 record", flush=True)
    result["splits"]["train"] = extract_split(model.to(device), train, work / "train", device)
    write_log(work / "metadata.json", result)
    print("Frozen G2 feature extraction complete", flush=True)


def cache_path(value):
    base = (Path("work_dirs") / "diagnostics").resolve()
    path = value.resolve()
    if base not in path.parents:
        raise ValueError("Use a local path under work_dirs/diagnostics for caches")
    return path


BATCH = 12
REGRESSION_POINTS = 128
REGRESSION_RIDGE = 1e-6
SEMANTIC_RIDGE = 1e-3
CALIBRATIONS = ("raw", "patch_layer_norm", "train_channel_std")
DIRECTIONS = {"rgb_to_dsm": ("z_rgb", "z_dsm", "pred_dsm"),
              "dsm_to_rgb": ("z_dsm", "z_rgb", "pred_rgb")}
SEMANTIC_FIELDS = ("z_rgb", "z_dsm", "u_rgb", "u_dsm")
SEMANTIC_GROUPS = {
    "z_rgb": ("z_rgb",), "z_dsm": ("z_dsm",), "u_rgb": ("u_rgb",), "u_dsm": ("u_dsm",),
    "z_rgb+z_dsm": ("z_rgb", "z_dsm"), "z_rgb+z_dsm+u_rgb+u_dsm": SEMANTIC_FIELDS,
}


class CacheSplit:
    def __init__(self, work, proof, scene_ids):
        path = work
        hashes = proof["sha256"]
        expected = {"windows.json", *(f"{name}.npy" for name in FIELDS)}
        if set(hashes) != expected or any(file_sha256(path / name) != hashes[name] for name in hashes):
            raise ValueError(f"Incomplete or altered {path.name} cache")
        self.windows = read_json(path / "windows.json")
        self.arrays = {field: np.load(path / f"{field}.npy", mmap_mode="r", allow_pickle=False)
                       for field in FIELDS}
        self.scenes = {scene: np.array([i for i, row in enumerate(self.windows) if row["scene"] == scene])
                       for scene in scene_ids}
        if (not all(len(ids) for ids in self.scenes.values()) or
                sum(map(len, self.scenes.values())) != len(self.windows) or
                len(self.windows) != proof["windows"] or
                len({row["id"] for row in self.windows}) != len(self.windows)):
            raise ValueError(f"Unexpected or overlapping {path.name} scene assignment")
        for field, array in self.arrays.items():
            dims = (len(self.windows), WIDTH, GRID, GRID) if field in VECTOR_FIELDS else (
                len(self.windows), GRID, GRID)
            dtype = np.uint8 if field == "patch_label" else (
                np.int16 if field in FIELDS[-2:] else np.float32)
            if array.shape != dims or array.dtype != dtype:
                raise ValueError(f"Invalid cache array shape or dtype: {field}")

    def batches(self, fields, device, indices=None):
        ids = np.arange(len(self.windows)) if indices is None else indices
        for offset in range(0, len(ids), BATCH):
            selected = ids[offset:offset + BATCH]
            yield selected, {field: torch.from_numpy(self.arrays[field][selected]).to(device)
                             for field in fields}


def as_rows(tensor):
    return tensor.movedim(1, -1).reshape(-1, tensor.shape[1])


def weighted_moments(accumulator, rows, weight):
    rows = rows.double()
    accumulator[0] += (rows * weight[:, None]).sum(0)
    accumulator[1] += (rows.square() * weight[:, None]).sum(0)


def moment_values(accumulator, weight):
    mean, second = accumulator / weight
    return mean, (second - mean.square()).clamp_min(0).sqrt()


def layer_norm(values):
    means = values.mean(1, keepdim=True)
    return (values - means) * (values.var(1, keepdim=True, correction=0) + 1e-5).rsqrt()


def normalize(values, field, calibration, moments):
    if calibration == "patch_layer_norm":
        return layer_norm(values)
    if calibration == "train_channel_std":
        mean, deviation = moments[("raw", field)]
        return (values - mean.to(values)[None, :, None, None]) / deviation.to(values)[
            None, :, None, None].clamp_min(1e-6)
    return values


def training_statistics(train, device):
    dtype = dict(device=device, dtype=torch.float64)
    moment_keys = [(mode, name) for mode in CALIBRATIONS[:2] for name in VECTOR_FIELDS[:2]]
    moments = {key: torch.zeros(2, WIDTH, **dtype) for key in moment_keys}
    semantic = {key: torch.zeros(2, WIDTH, **dtype) for key in SEMANTIC_FIELDS}
    counts = torch.zeros(NUM_CLASSES, device=device, dtype=torch.int64)
    weight_sum = torch.zeros((), **dtype)
    norm_chunks = {path: [] for path in ("rgb", "dsm")}
    for _, batch in train.batches((*SEMANTIC_FIELDS, "coverage", "patch_label"), device):
        q = batch["coverage"].flatten().double()
        label = batch["patch_label"].flatten().long()
        eligible = label != IGNORE_INDEX
        counts += torch.bincount(label[eligible], minlength=NUM_CLASSES)
        weight_sum += q.sum()
        for mode, name in moment_keys:
            weighted_moments(moments[(mode, name)], as_rows(normalize(batch[name], name, mode, {})), q)
        for name in SEMANTIC_FIELDS:
            weighted_moments(semantic[name], as_rows(batch[name])[eligible], torch.ones_like(q[eligible]))
        for path in norm_chunks:
            norms = as_rows(batch[f"u_{path}"]).double().norm(dim=1)
            norm_chunks[path].append(norms[q > 0].cpu().numpy())
    if weight_sum.item() <= 0 or (counts == 0).any().item():
        raise ValueError("Train statistics require all classes and DSM support")
    moments = {key: moment_values(value, weight_sum) for key, value in moments.items()}
    semantic = {key: moment_values(value, counts.sum()) for key, value in semantic.items()}
    edges = {path: np.quantile(np.concatenate(chunks), [.25, .5, .75]).tolist()
             for path, chunks in norm_chunks.items()}
    return moments, semantic, counts, edges


def convolution_rows(values):
    return torch.nn.functional.unfold(values, kernel_size=3, padding=1).transpose(1, 2)


def center_columns(device):
    return torch.arange(WIDTH, device=device) * 9 + 4


def weighted_cross_products(design, targets, weight):
    weight_column = weight[:, None]
    return design.T @ (design * weight_column), design.T @ (targets * weight_column)


def add_intercept(design):
    return torch.cat((design, torch.ones_like(design[:, :1])), dim=1)


def fit_linear(gram, cross, weight_sum, regularizer):
    scaled = gram / weight_sum
    penalty = regularizer * scaled.diag()[:-1].mean().clamp_min(1e-12)
    scaled.diagonal()[:-1].add_(penalty)
    return torch.linalg.solve(scaled, cross / weight_sum)


def patch_samples(window_count):
    generator = np.random.default_rng(SEED)
    return np.stack([generator.choice(GRID * GRID, REGRESSION_POINTS, replace=False)
                     for _ in range(window_count)])


def regression_input(batch, calibration, name, moments):
    result = normalize(batch[name], name, calibration, moments)
    if name == "z_dsm":
        result = torch.where(batch["coverage"][:, None] > 0, result, 0.0)
    return result


def fit_regression(train, device, moments, calibration, names, samples):
    source, target, _ = names
    shape = WIDTH * 9 + 1
    gram = torch.zeros(shape, shape, dtype=torch.float64, device=device)
    cross = torch.zeros(shape, WIDTH, dtype=torch.float64, device=device)
    q_total = torch.zeros((), dtype=torch.float64, device=device)
    points = 0
    for ids, batch in train.batches((source, target, "coverage"), device):
        choice = torch.from_numpy(samples[ids]).to(device)
        windows = torch.arange(len(ids), device=device)[:, None]
        q = batch["coverage"].flatten(1)[windows, choice].flatten().double()
        keep = q > 0
        q = q[keep]
        x = convolution_rows(regression_input(batch, calibration, source, moments))[windows, choice]
        y = normalize(batch[target], target, calibration, moments).flatten(2).transpose(1, 2)[windows, choice]
        design = add_intercept(x.reshape(-1, shape - 1)[keep].double())
        update_gram, update_cross = weighted_cross_products(design, y.reshape(-1, WIDTH)[keep].double(), q)
        gram += update_gram
        cross += update_cross
        points += q.numel()
        q_total += q.sum()
    if points < shape:
        raise ValueError("Not enough supported train patches for regression")
    columns = torch.cat((center_columns(device), torch.tensor([shape - 1], device=device)))
    return {
        "linear_1x1": fit_linear(gram[columns][:, columns], cross[columns], q_total, REGRESSION_RIDGE),
        "conv_3x3": fit_linear(gram, cross, q_total, REGRESSION_RIDGE),
    }, points


class RegressionScores:
    def __init__(self, device):
        self.sums = torch.zeros(5, dtype=torch.float64, device=device)
        self.means = torch.zeros(2, WIDTH, dtype=torch.float64, device=device)
        self.supported = torch.zeros((), dtype=torch.int64, device=device)

    def add(self, target, prediction, weight):
        w = weight[:, None]
        error = prediction - target
        self.sums += torch.stack((weight.sum(), (w * error.square()).sum(),
                                  (w * target.square()).sum(), (w * prediction.square()).sum(),
                                  (w * target * prediction).sum()))
        self.means[0] += (w * target).sum(0)
        self.means[1] += (w * prediction).sum(0)
        self.supported += (weight > 0).sum()

    def merge(self, other):
        self.sums += other.sums
        self.means += other.means
        self.supported += other.supported

    def scores(self, mean_baseline):
        total, sse, t2, p2, dot = self.sums.tolist()
        if total == 0:
            return dict(q_sum=0, supported_patches=0, mse=None, r2_vs_train_mean=None,
                        cosine=None, target_variance=None, prediction_variance=None)
        baseline_sse = mean_baseline.sums[1].item()
        scale = total * WIDTH
        center = (self.means / total).square().mean(dim=1).tolist()
        return dict(q_sum=total, supported_patches=self.supported.item(), mse=sse / scale,
                    r2_vs_train_mean=1 - sse / baseline_sse if baseline_sse > 0 else None,
                    cosine=dot / (t2 * p2)**.5 if t2 > 0 and p2 > 0 else None,
                    target_variance=max(0.0, t2 / scale - center[0]),
                    prediction_variance=max(0.0, p2 / scale - center[1]))


def eval_regression(val, device, moments, direction, calibration, coefficients, points):
    source, target, frozen = DIRECTIONS[direction]
    mean, std = moments[("raw", target)]
    std = std.clamp_min(1e-6)
    baseline = (moments[(calibration, target)][0] if calibration != "train_channel_std"
                else torch.zeros_like(mean))
    methods = ("train_mean", "g2_frozen", *coefficients)
    spaces = (calibration, "raw_from_train_channel_std") if calibration == "train_channel_std" else (calibration,)
    total = {space: {method: RegressionScores(device) for method in methods} for space in spaces}
    rows = []

    def emit(scope, space, metrics):
        for name, score in metrics.items():
            rows.append(dict(scene=scope, direction=direction, calibration=space, predictor=name,
                             train_fit_patches=points if name in coefficients else 0,
                             **score.scores(metrics["train_mean"])))

    for scene, ids in val.scenes.items():
        scores = {space: {method: RegressionScores(device) for method in methods} for space in spaces}
        for _, batch in val.batches((source, target, frozen, "coverage"), device, indices=ids):
            q = batch["coverage"].flatten().double()
            x = regression_input(batch, calibration, source, moments)
            y = as_rows(normalize(batch[target], target, calibration, moments)).double()
            original = as_rows(normalize(batch[frozen], target, calibration, moments)).double()
            predictions = dict(train_mean=baseline.expand_as(y), g2_frozen=original,
                               linear_1x1=add_intercept(as_rows(x).double()) @ coefficients["linear_1x1"],
                               conv_3x3=add_intercept(
                                   convolution_rows(x).reshape(-1, WIDTH * 9).double()) @ coefficients["conv_3x3"])
            for name, prediction in predictions.items():
                scores[calibration][name].add(y, prediction, q)
                if len(spaces) == 2:
                    scores[spaces[1]][name].add(as_rows(batch[target]).double(), prediction * std + mean, q)
        for space in spaces:
            emit(scene, space, scores[space])
            for name in methods:
                total[space][name].merge(scores[space][name])
    for space in spaces:
        emit("ALL", space, total[space])
    return rows


def semantic_design(batch, moments):
    rows = []
    for name, (mean, std) in moments.items():
        values = as_rows(batch[name]).double()
        rows.append((values - mean) / std.clamp_min(1e-6))
    return add_intercept(torch.cat(rows, dim=1))


def fit_semantic(train, device, moments, counts):
    size = len(SEMANTIC_FIELDS) * WIDTH + 1
    gram = torch.zeros(size, size, dtype=torch.float64, device=device)
    cross = torch.zeros(size, NUM_CLASSES, dtype=torch.float64, device=device)
    class_weight = counts.sum() / (NUM_CLASSES * counts.double())
    for _, batch in train.batches((*SEMANTIC_FIELDS, "patch_label"), device):
        labels = batch["patch_label"].flatten().long()
        keep = labels != IGNORE_INDEX
        labels = labels[keep]
        design = semantic_design(batch, moments)[keep]
        one_hot = torch.nn.functional.one_hot(labels, NUM_CLASSES).double()
        g, h = weighted_cross_products(design, one_hot, class_weight[labels])
        gram += g
        cross += h
    probes = {}
    for probe, inputs in SEMANTIC_GROUPS.items():
        columns = [i for k, field in enumerate(SEMANTIC_FIELDS) if field in inputs
                   for i in range(k * WIDTH, (k + 1) * WIDTH)]
        columns = torch.tensor([*columns, size - 1], device=device)
        probes[probe] = (columns, fit_linear(gram[columns][:, columns], cross[columns],
                                             counts.sum(), SEMANTIC_RIDGE))
    return probes


def confusion_matrix(label, predicted):
    return torch.bincount(label.long() * NUM_CLASSES + predicted,
                          minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def patch_metrics(matrix):
    diagonal = matrix.diagonal().double()
    union = matrix.sum(0) + matrix.sum(1) - diagonal
    iou = [100 * tp.item() / total.item() if total.item() else None for tp, total in zip(diagonal, union)]
    return dict(mIoU5=sum(iou[:5]) / 5 if all(v is not None for v in iou[:5]) else None,
                mIoU6=sum(iou) / 6 if all(v is not None for v in iou) else None,
                OA=100 * diagonal.sum().item() / matrix.sum().item() if matrix.sum().item() else None,
                **dict(zip(CLASSES, iou)))


def eval_semantic(val, device, moments, probes):
    subsets = ("all_labeled", "q_positive")
    matrices = {scope: {name: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
                        for name in probes} for scope in subsets}
    for _, batch in val.batches((*SEMANTIC_FIELDS, "patch_label", "coverage"), device):
        design = semantic_design(batch, moments)
        label = batch["patch_label"].flatten()
        keep = label != IGNORE_INDEX
        support = keep & (batch["coverage"].flatten() > 0)
        for name, (cols, weight) in probes.items():
            prediction = (design[:, cols] @ weight).argmax(1)
            for scope, selected in zip(subsets, (keep, support)):
                matrices[scope][name] += confusion_matrix(label[selected], prediction[selected])
    return [dict(subset=scope, probe=name, dimensions=len(probes[name][0]) - 1,
                 labeled_patches=mat.sum().item(), **patch_metrics(mat))
            for scope, variants in matrices.items() for name, mat in variants.items()]


def route_groups(q, label, valid, correct, norm, edges):
    support = q > 0
    yield "support", "q>0", support
    yield "support", "q=0", ~support
    for lo, hi in zip((0, .25, .5, .75), (.25, .5, .75, 1)):
        yield "coverage", f"({lo},{hi}]" if hi < 1 else "(0.75,1)", (
            (q > lo) & (q <= hi) if hi < 1 else (q > lo) & (q < hi))
    yield "coverage", "q=1", q == 1
    for k, name in enumerate(CLASSES):
        yield "truth", name, support & (label == k)
    yield "truth", "no_strict_majority", support & (label == IGNORE_INDEX)
    yield "g2_pixel_correctness", ">50%", support & (valid > 0) & (2 * correct > valid)
    yield "g2_pixel_correctness", "<=50%", support & (valid > 0) & (2 * correct <= valid)
    yield "g2_pixel_correctness", "unlabeled", support & (valid == 0)
    for i, (lo, hi) in enumerate(zip((-float("inf"), *edges), (*edges, float("inf"))), 1):
        yield "innovation_norm", f"train_quartile_{i}", support & (norm > lo) & (norm <= hi)


def evaluate_routes(val, device, edges):
    stats = {}
    spatial = {path: torch.zeros(3, dtype=torch.float64, device=device) for path in edges}
    fields = ("u_rgb", "u_dsm", "w_rgb", "w_dsm", "coverage", *FIELDS[-3:])
    for _, batch in val.batches(fields, device):
        q = batch["coverage"].flatten().double()
        label, valid, correct = [batch[key].flatten().long() for key in FIELDS[-3:]]
        for path, cutoffs in edges.items():
            gate = batch[f"w_{path}"].flatten().double()
            norm = as_rows(batch[f"u_{path}"]).double().norm(dim=1)
            weight = torch.where(q > 0, q, 1.0)
            values = torch.stack((torch.ones_like(q), weight, gate * weight, gate.square() * weight,
                                  (gate < .01) * weight, (gate > .99) * weight, norm * weight,
                                  (gate * norm) * weight), dim=1)
            for group, name, selected in route_groups(q, label, valid, correct, norm, cutoffs):
                key = (path, group, name)
                if key not in stats:
                    stats[key] = torch.zeros(values.shape[1], dtype=torch.float64, device=device)
                stats[key] += (values * selected[:, None]).sum(0)
            qm, gm = q.reshape(-1, GRID**2), gate.reshape(-1, GRID**2)
            ws = qm.sum(1)
            good = ws > 0
            qm, gm, ws = qm[good], gm[good], ws[good]
            mu = (qm * gm).sum(1) / ws
            sd = ((qm * (gm - mu[:, None]).square()).sum(1) / ws).sqrt()
            spread = torch.where(qm > 0, gm, -float("inf")).max(1).values - torch.where(
                qm > 0, gm, float("inf")).min(1).values
            spatial[path] += torch.stack((sd.sum(), (spread > .1).sum(), good.sum()))
    out = []
    for (path, group, name), values in stats.items():
        count, weight, first, second, near_zero, near_one, residual, routed = values.tolist()
        if count == 0:
            continue
        mean = first / weight
        out.append(dict(path=path, group=group, bin=name, patches=int(count), weight=weight,
                        gate_mean=mean, gate_sd=max(0.0, second / weight - mean**2)**.5,
                        low_fraction=near_zero / weight, high_fraction=near_one / weight,
                        innovation_l2=residual / weight, routed_l2=routed / weight))
    diagnostics = {}
    for path, value in spatial.items():
        sd_sum, varying, windows = value.tolist()
        diagnostics[path] = dict(windows=int(windows), mean_within_window_sd=sd_sum / windows,
                                 window_range_gt_01_fraction=varying / windows)
    return out, diagnostics


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def diagnose(work, device):
    metadata = read_json(work / "metadata.json")
    protocol = metadata["data_protocol"]
    m1 = read_json(M1)
    if (set(metadata["splits"]) != {"train", "val"} or protocol != m1["data_protocol"] or
            metadata["model"]["checkpoint_sha256"] != m1["checkpoints"]["G2"]["sha256"] or
            metadata["model"]["best"] != m1["checkpoints"]["G2"]["best"] or
            tuple(metadata["classes"]) != tuple(CLASSES) or
            metadata["precision"] != "FP32" or metadata["tta"]):
        raise ValueError("Expected completed audited G2 train and validation feature splits")
    if set(protocol["train_scenes"]) & set(protocol["val_scenes"]):
        raise ValueError("Training and validation scenes overlap")
    train = CacheSplit(work / "train", metadata["splits"]["train"], protocol["train_scenes"])
    m, sem, counts, edges = training_statistics(train, device)
    samples = patch_samples(len(train.windows))
    probes = {}
    for direction, names in DIRECTIONS.items():
        for calibration in CALIBRATIONS:
            probes[direction, calibration] = fit_regression(train, device, m, calibration, names, samples)
            print(f"Fitted train-only {direction} {calibration}", flush=True)
    classifiers = fit_semantic(train, device, sem, counts)
    print("Fitted all train-only semantic probes; opening held-out features", flush=True)
    val = CacheSplit(work / "val", metadata["splits"]["val"], protocol["val_scenes"])
    prediction = []
    for (direction, calibration), (coefficients, count) in probes.items():
        prediction.extend(eval_regression(val, device, m, direction, calibration, coefficients, count))
        print(f"Evaluated {direction} {calibration}: {len(val.scenes)} held-out scenes", flush=True)
    semantic = eval_semantic(val, device, sem, classifiers)
    routing, spatial = evaluate_routes(val, device, edges)
    write_log(work / "audit.json", dict(
        prediction=prediction, semantic=semantic, routing=routing, spatial_gate=spatial,
        innovation_edges=edges, semantic_train_counts=dict(zip(CLASSES, counts.tolist())),
        train_label_excluded=len(train.windows) * GRID**2 - int(counts.sum().item()),
        val_label_excluded=len(val.windows) * GRID**2 - semantic[0]["labeled_patches"]))
    write_csv(work / "BCIA_PREDICTION_PROBES.csv", prediction)
    write_csv(work / "BCIA_SEMANTIC_PROBES.csv", semantic)
    print("Probe evaluation and route summaries complete", flush=True)




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("extract", "diagnose"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--reuse-validation", action="store_true")
    args = parser.parse_args()
    if args.reuse_validation and args.phase != "extract":
        parser.error("--reuse-validation applies only to extraction")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    initialize_rng(SEED, device)
    work = cache_path(args.work_dir)
    if args.phase == "extract":
        extract(Path(os.environ["BCIA_DATA_ROOT"]), Path(os.environ["BCIA_BACKBONE_CHECKPOINT"]),
                work, device, reuse_validation=args.reuse_validation)
    else:
        diagnose(work, device)


if __name__ == "__main__":
    main()
