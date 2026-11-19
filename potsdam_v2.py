"""Build or verify the local Potsdam v2 candidate beside its read-only v1."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from PIL import Image


SOURCE_HASH = "eed5e1c6900c58ceb3054bbdc5e414eb11e941814e818cfde741b1ab6b757947"
MEMBER = "5_Labels_for_participants_no_Boundary/top_potsdam_4_12_label_noBoundary.tif"
PALETTE = ((0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 255),
           (0, 255, 0), (255, 255, 0), (255, 0, 0))
ASSETS = ("rgb", "dsm", "valid", "label")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_labels(rgb):
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB labels")
    labels = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for index, color in enumerate(PALETTE):
        labels[np.all(rgb == color, axis=-1)] = index
    if (labels == 255).any():
        raise ValueError("Noncanonical color in replacement TIFF")
    return labels


def raster(raw, mode=None):
    with Image.open(io.BytesIO(raw)) as image:
        if mode is not None and image.mode != mode:
            raise ValueError(f"Expected {mode}, got {image.mode}")
        return np.array(image)


def checked(root, row, asset):
    raw = (root / row[f"{asset}_path"]).read_bytes()
    if sha(raw) != row[f"{asset}_sha256"]:
        raise ValueError(f"{row['id']} {asset} fingerprint mismatch")
    return raw


def read_rows(root, split):
    return [json.loads(line) for line in (root / "manifests" / f"{split}.jsonl")
            .read_text(encoding="utf-8").splitlines()]


def process(v1, root, archive, *, build):
    with zipfile.ZipFile(archive) as source:
        raw = source.read(MEMBER)
    if sha(raw) != SOURCE_HASH:
        raise ValueError("Replacement member SHA-256 differs from audited content")
    replacement = exact_labels(raster(raw, "RGB"))
    if replacement.shape != (6000, 6000):
        raise ValueError("Unexpected source grid")
    metadata_raw = (v1 / "metadata.json").read_bytes()
    stats_raw = (v1 / "normalization.json").read_bytes()
    if json.loads(metadata_raw)["status"] != "verified":
        raise ValueError("Expected verified v1")
    if build:
        (root / "manifests").mkdir()
        (root / "labels" / "train").mkdir(parents=True)
        (root / "normalization.json").write_bytes(stats_raw)
    elif (root / "normalization.json").read_bytes() != stats_raw:
        raise ValueError("Normalization differs from v1")
    result = {"splits": {}, "changed_tiles": 0, "candidate_4_12_source_counts":
              np.bincount(replacement.ravel(), minlength=7).tolist()}
    by_split = {}
    ids = set()
    for split, expected_count, expected_scenes in (("train", 3456, 24), ("val", 2016, 14)):
        originals = read_rows(v1, split)
        scenes = {row["scene"] for row in originals}
        by_split[split] = scenes
        if len(originals) != expected_count or len(scenes) != expected_scenes:
            raise ValueError(f"Unexpected split {split}")
        observed = read_rows(root, split) if not build else originals
        if len(observed) != len(originals):
            raise ValueError("Manifest count mismatch")
        before = np.zeros(7, dtype=np.int64)
        after = before.copy()
        focus_before, focus_after = before.copy(), before.copy()
        fractions = []
        all_ignore, before_all_ignore = [], []
        focus_fractions = []
        manifest = []
        for old, actual in zip(originals, observed):
            tile_id = old["id"]
            scene = old["scene"]
            grid = old["xyxy"]
            if (tile_id in ids or old["split"] != split or
                    tile_id != "_".join(map(str, (scene, *grid)))):
                raise ValueError(f"Unexpected sample identity: {tile_id}")
            ids.add(tile_id)
            left, top, right, bottom = grid
            if not (0 <= left < right <= 6000 and 0 <= top < bottom <= 6000
                    and right - left == bottom - top == 512):
                raise ValueError(f"Invalid crop: {tile_id}")
            row = dict(old)
            row.pop("source_dsm_path", None)
            for asset in ASSETS:
                path = (v1 / old[f"{asset}_path"]).resolve(strict=True)
                row[f"{asset}_path"] = Path(os.path.relpath(path, root)).as_posix()
            rgb = raster(checked(root, row, "rgb"), "RGB")
            dsm = np.load(io.BytesIO(checked(root, row, "dsm")), allow_pickle=False)
            valid = raster(checked(root, row, "valid"))
            original = raster(checked(root, row, "label"))
            if (rgb.shape != (512, 512, 3) or dsm.shape != (512, 512) or
                    dsm.dtype != np.float32 or not np.isfinite(dsm).all() or
                    valid.shape != dsm.shape or not np.isin(valid, (0, 1)).all() or
                    np.count_nonzero(valid) != row["valid_pixels"] or
                    original.shape != dsm.shape or original.dtype != np.uint8 or original.max() > 6):
                raise ValueError(f"Invalid asset: {tile_id}")
            label = original
            old_counts = np.bincount(original.ravel(), minlength=7)
            if old_counts[0] == original.size:
                before_all_ignore.append(tile_id)
            if scene == "4_12":
                if split != "train":
                    raise ValueError("Replacement scene must be train-only")
                row.update(label_path=f"labels/train/{tile_id}.png",
                           v1_label_sha256=old["label_sha256"],
                           label_source_sha256=SOURCE_HASH,
                           label_mapping="exact_rgb_palette_0_to_6")
                output = root / row["label_path"]
                expected = replacement[top:bottom, left:right]
                if build:
                    Image.fromarray(expected).save(output)
                output_bytes = output.read_bytes()
                label = raster(output_bytes)
                if label.dtype != np.uint8 or not np.array_equal(label, expected):
                    raise ValueError(f"Incorrect exact-palette crop: {tile_id}")
                row["label_sha256"] = sha(output_bytes)
                result["changed_tiles"] += 1
                focus_before += old_counts
                focus_after += np.bincount(label.ravel(), minlength=7)
                focus_fractions.append(np.count_nonzero(label) / label.size)
            if not build and actual != row:
                raise ValueError(f"Unexpected manifest contents: {tile_id}")
            before += old_counts
            after += np.bincount(label.ravel(), minlength=7)
            fractions.append(np.count_nonzero(label) / label.size)
            if not fractions[-1]:
                all_ignore.append(tile_id)
            manifest.append(row)
        if build:
            (root / "manifests" / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest), encoding="utf-8")
        result["splits"][split] = dict(before=before.tolist(), after=after.tolist(),
                                     focus_before=focus_before.tolist(), focus_after=focus_after.tolist(),
                                     valid_fraction_min_median_max=np.quantile(fractions, [0, .5, 1]).tolist(),
                                     below_one_percent=sum(value < .01 for value in fractions),
                                     focus_valid_fraction_min=min(focus_fractions) if focus_fractions else None,
                                     before_all_ignore=before_all_ignore,
                                     all_ignore=all_ignore)
    if (result["changed_tiles"] != 144 or by_split["train"] & by_split["val"] or
            by_split["train"] != set(json.loads(stats_raw)["scene_ids"])):
        raise ValueError("Unexpected replacement coverage or scene split")
    metadata = dict(status="verified", version="potsdam_rgb_dsm_v2", role="formal_candidate",
                    legacy_version="potsdam_rgb_dsm_v1", replacement_archive=archive.name,
                    replacement_archive_sha256=file_sha(archive),
                    replacement_member=MEMBER, replacement_member_sha256=SOURCE_HASH,
                    palette_rgb=PALETTE, v1_metadata_sha256=sha(metadata_raw),
                    normalization_sha256=sha(stats_raw),
                    v1_manifest_sha256={split: sha((v1 / "manifests" / f"{split}.jsonl").read_bytes())
                                        for split in by_split},
                    manifest_sha256={split: sha((root / "manifests" / f"{split}.jsonl").read_bytes())
                                     for split in by_split},
                    unchanged_labels="v1 legacy mapping, including 6_7",
                    verification=result)
    if build:
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    elif json.loads((root / "metadata.json").read_text(encoding="utf-8")) != json.loads(json.dumps(metadata)):
        raise ValueError("Metadata does not match verified assets")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    v1, v2 = args.v1.resolve(), args.v2.resolve()
    if v1.name != "potsdam_rgb_dsm_v1" or v2.name != "potsdam_rgb_dsm_v2" or v1.parent != v2.parent:
        parser.error("Use sibling potsdam_rgb_dsm_v1 and potsdam_rgb_dsm_v2 directories")
    if args.action == "build":
        if v2.exists():
            raise FileExistsError(v2)
        with tempfile.TemporaryDirectory(prefix=".potsdam-v2-", dir=v2.parent) as staging:
            root = Path(staging)
            report = process(v1, root, args.archive, build=True)
            root.rename(v2)
    else:
        report = process(v1, v2, args.archive, build=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
