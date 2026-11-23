"""Fit directional and matched-width local-context probes on the existing G2 cache."""
import argparse
import csv
from pathlib import Path

import torch
from torch.nn.functional import avg_pool2d, one_hot

from bcia_feature_audit import (
    CacheSplit, M1, SEMANTIC_RIDGE, WIDTH, add_intercept, as_rows, cache_path, confusion_matrix,
    fit_linear, moment_values, patch_metrics, read_json, semantic_design, weighted_cross_products,
    weighted_moments,
)
from bcia_peft.model import IGNORE_INDEX, NUM_CLASSES
from bcia_peft.train import initialize_rng, write_log
from m1_diagnostics import CLASSES


BASE = ("z_rgb", "z_dsm")
PREDICTION = ("pred_rgb", "pred_dsm")
RESIDUAL = ("u_rgb", "u_dsm")
POOL = ("pool_rgb", "pool_dsm")
FEATURES = (*BASE, *PREDICTION, *POOL, *RESIDUAL)
CACHE_FIELDS = (*BASE, *PREDICTION, *RESIDUAL, "coverage", "patch_label")
PROBES = {
    "P0": BASE,
    "P1": (*BASE, "pred_rgb"),
    "P2": (*BASE, "pred_dsm"),
    "P3": (*BASE, *PREDICTION),
    "P4": (*BASE, "pool_dsm"),
    "P5": (*BASE, "pool_rgb"),
    "P6": (*BASE, *POOL),
    "U": (*BASE, *RESIDUAL),
}
METRICS = ("mIoU5", "mIoU6", "OA", *CLASSES)
PRIOR_SEMANTIC_CSV = Path("BCIA_SEMANTIC_PROBES.csv")


def probe_features(batch):
    support = batch["coverage"][:, None] > 0
    z_rgb = batch["z_rgb"]
    z_dsm = torch.where(support, batch["z_dsm"], 0.0)
    pred_rgb = torch.where(support, batch["pred_rgb"], 0.0)
    pred_dsm = torch.where(support, batch["pred_dsm"], 0.0)
    if (not torch.equal(z_dsm, batch["z_dsm"]) or not torch.equal(pred_rgb, batch["pred_rgb"]) or
            not torch.equal(z_rgb - pred_rgb, batch["u_rgb"]) or
            not torch.equal(z_dsm - pred_dsm, batch["u_dsm"])):
        raise ValueError("Cached features violate the G2 residual or DSM masking identities")
    pool_rgb = avg_pool2d(z_rgb, kernel_size=3, stride=1, padding=1, count_include_pad=True)
    pool_dsm = avg_pool2d(z_dsm, kernel_size=3, stride=1, padding=1, count_include_pad=True)
    return dict(z_rgb=z_rgb, z_dsm=z_dsm, pred_rgb=pred_rgb, pred_dsm=pred_dsm,
                pool_rgb=pool_rgb, pool_dsm=torch.where(support, pool_dsm, 0.0),
                u_rgb=batch["u_rgb"], u_dsm=batch["u_dsm"])


def transform_raw_weights(weight):
    """Swap predictions and residuals for raw coordinates [base RGB, base DSM, extra RGB, extra DSM]."""
    zr, zh, extra_r, extra_h = weight.split(WIDTH, dim=0)
    return torch.cat((zr + extra_r, zh + extra_h, -extra_r, -extra_h), dim=0)


def raw_affine(standardized, features, moments):
    means, scales = zip(*(moments[field] for field in features))
    weight = standardized[:-1] / torch.cat(scales).clamp_min(1e-6)[:, None]
    bias = standardized[-1] - torch.cat(means) @ weight
    return weight, bias


def train_moments(train, device):
    moments = {key: torch.zeros(2, WIDTH, dtype=torch.float64, device=device) for key in FEATURES}
    counts = torch.zeros(NUM_CLASSES, dtype=torch.int64, device=device)
    no_dsm = torch.zeros(2, dtype=torch.int64, device=device)
    for _, batch in train.batches(CACHE_FIELDS, device):
        features = probe_features(batch)
        labels = batch["patch_label"].flatten().long()
        keep = labels != IGNORE_INDEX
        counts += torch.bincount(labels[keep], minlength=NUM_CLASSES)
        missing = batch["coverage"].flatten() == 0
        no_dsm += torch.stack((missing.sum(), (missing & keep).sum()))
        weight = torch.ones_like(labels[keep], dtype=torch.float64)
        for key, value in features.items():
            weighted_moments(moments[key], as_rows(value)[keep], weight)
    if (counts <= 0).any().item():
        raise ValueError("Training cache lacks a required majority class")
    return {key: moment_values(value, counts.sum()) for key, value in moments.items()}, counts, no_dsm.tolist()


def fit_probes(train, device, moments, counts):
    size = len(FEATURES) * WIDTH + 1
    gram = torch.zeros(size, size, dtype=torch.float64, device=device)
    cross = torch.zeros(size, NUM_CLASSES, dtype=torch.float64, device=device)
    frequency_weight = counts.sum() / (NUM_CLASSES * counts.double())
    for _, batch in train.batches(CACHE_FIELDS, device):
        labels = batch["patch_label"].flatten().long()
        keep = labels != IGNORE_INDEX
        design = semantic_design(probe_features(batch), moments)[keep]
        labels = labels[keep]
        g, h = weighted_cross_products(design, one_hot(labels, NUM_CLASSES).double(),
                                       frequency_weight[labels])
        gram += g
        cross += h
    fits = {}
    for name, fields in PROBES.items():
        columns = [i for key in fields
                   for i in range(FEATURES.index(key) * WIDTH, (FEATURES.index(key) + 1) * WIDTH)]
        index = torch.tensor([*columns, size - 1], device=device)
        weight = fit_linear(gram[index][:, index], cross[index], counts.sum(), SEMANTIC_RIDGE)
        fits[name] = (index, weight)
    return fits


def check_equivalence(features, selected, scores, mapped, stats):
    use = {name: as_rows(features[name])[selected].double() for name in (*BASE, *PREDICTION, *RESIDUAL)}
    rgb, dsm = (use[key] for key in BASE)
    raw_p = torch.cat((rgb, dsm, use["pred_rgb"], use["pred_dsm"]), dim=1)
    raw_u = torch.cat((rgb, dsm, use["u_rgb"], use["u_dsm"]), dim=1)
    exact_u = torch.cat((rgb, dsm, rgb - use["pred_rgb"], dsm - use["pred_dsm"]), dim=1)
    for name, source, target in (("P3_to_U", raw_p, exact_u), ("U_to_P3", exact_u, raw_p)):
        weight, bias, transferred = mapped[name]
        algebra = (source @ weight + bias - (target @ transferred + bias)).abs().max()
        reconstructed = target if name == "U_to_P3" else raw_u
        shifted = reconstructed @ transferred + bias
        native = scores["P3" if name == "P3_to_U" else "U"]
        stats[name][0] = torch.maximum(stats[name][0], algebra)
        stats[name][1] = torch.maximum(stats[name][1], (native - shifted).abs().max())
        stats[name][2] += (native.argmax(1) != shifted.argmax(1)).sum()


def evaluate(val, device, moments, fits):
    matrices = {name: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
                for name in fits}
    stats = {name: torch.zeros(3, dtype=torch.float64, device=device) for name in ("P3_to_U", "U_to_P3")}
    mapped = {}
    for probe, name in (("P3", "P3_to_U"), ("U", "U_to_P3")):
        raw, bias = raw_affine(fits[probe][1], PROBES[probe], moments)
        mapped[name] = (raw, bias, transform_raw_weights(raw))
    no_dsm = torch.zeros(2, dtype=torch.int64, device=device)
    for _, batch in val.batches(CACHE_FIELDS, device):
        features = probe_features(batch)
        design = semantic_design(features, moments)
        labels = batch["patch_label"].flatten()
        keep = labels != IGNORE_INDEX
        missing = batch["coverage"].flatten() == 0
        no_dsm += torch.stack((missing.sum(), (missing & keep).sum()))
        if not keep.any().item():
            continue
        design = design[keep]
        scores = {name: design[:, columns] @ coef for name, (columns, coef) in fits.items()}
        for name, result in scores.items():
            matrices[name] += confusion_matrix(labels[keep], result.argmax(1))
        check_equivalence(features, keep, scores, mapped, stats)
    ref = {name: dict(algebraic_max_abs_logit=value[0].item(),
                      cached_max_abs_logit=value[1].item(), argmax_disagreements=int(value[2].item()))
           for name, value in stats.items()}
    if any(item["algebraic_max_abs_logit"] > 1e-9 for item in ref.values()):
        raise ValueError(f"Prediction and residual coefficient mapping failed: {ref}")
    return matrices, ref, no_dsm.tolist()


def verify_previous(rows):
    with PRIOR_SEMANTIC_CSV.open(encoding="utf-8", newline="") as stream:
        previous = {row["probe"]: row for row in csv.DictReader(stream) if row["subset"] == "all_labeled"}
    for name, known in (("P0", "z_rgb+z_dsm"), ("U", "z_rgb+z_dsm+u_rgb+u_dsm")):
        observed = rows[name]
        earlier = previous[known]
        if (observed["val_patches"] != int(earlier["labeled_patches"]) or
                any(abs(observed[key] - float(earlier[key])) > 1e-10 for key in METRICS)):
            raise ValueError(f"{name} did not reproduce the published G2 probe: {observed}")


@torch.inference_mode()
def run(cache, output, device):
    meta = read_json(cache / "metadata.json")
    prior = read_json(cache / "audit.json")
    m1 = read_json(M1)
    protocol = meta["data_protocol"]
    if (protocol != m1["data_protocol"] or meta["model"]["checkpoint_sha256"] != m1["checkpoints"]["G2"]["sha256"]
            or meta["model"]["best"] != m1["checkpoints"]["G2"]["best"]
            or tuple(meta["classes"]) != tuple(CLASSES) or meta["precision"] != "FP32" or meta["tta"]):
        raise ValueError("Cache is not the verified formal G2 feature snapshot")
    if set(protocol["train_scenes"]) & set(protocol["val_scenes"]):
        raise ValueError("Train and validation scenes overlap")
    if output.exists():
        raise FileExistsError(output)

    train = CacheSplit(cache / "train", meta["splits"]["train"], protocol["train_scenes"])
    moments, counts, train_no_dsm = train_moments(train, device)
    observed_counts = dict(zip(CLASSES, counts.tolist()))
    if observed_counts != prior["semantic_train_counts"]:
        raise ValueError("Cached labels do not reproduce the prior class weights")
    print(f"Verified train cache and identical class weights on {counts.sum().item()} patches", flush=True)
    fits = fit_probes(train, device, moments, counts)
    print("Fitted all probes and residual equivalence control before reading validation arrays", flush=True)
    val = CacheSplit(cache / "val", meta["splits"]["val"], protocol["val_scenes"])
    confusion, equivalent, val_no_dsm = evaluate(val, device, moments, fits)
    rows = {name: dict(probe=name, features="+".join(PROBES[name]), dimensions=len(PROBES[name]) * WIDTH,
                       train_patches=int(counts.sum().item()), val_patches=matrix.sum().item(),
                       **patch_metrics(matrix)) for name, matrix in confusion.items()}
    verify_previous(rows)
    for row in rows.values():
        row.update({f"delta_{key}_pp": row[key] - rows["P0"][key] for key in METRICS})
    output.mkdir(parents=True)
    with (output / "BCIA_DIRECTIONAL_CONTEXT_PROBES.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows["P0"]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows.values())
    write_log(output / "summary.json", dict(probes=rows, equivalence=equivalent,
              class_counts=observed_counts, q_zero_patches=dict(train=train_no_dsm, val=val_no_dsm),
              checkpoint_sha256=meta["model"]["checkpoint_sha256"], data_protocol=protocol,
              cache_sha256={split: meta["splits"][split]["sha256"] for split in ("train", "val")},
              confusion={k: v.cpu().tolist() for k,v in confusion.items()}))
    for name, row in rows.items():
        print(f"{name}: mIoU5={row['mIoU5']:.6f}, mIoU6={row['mIoU6']:.6f}, OA={row['OA']:.6f}", flush=True)
    print(equivalent, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache, output = cache_path(args.cache), cache_path(args.output)
    if output == cache or cache in output.parents:
        parser.error("Use a distinct local output directory outside the source cache")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    initialize_rng(42, device)
    run(cache, output, device)


if __name__ == "__main__":
    main()
