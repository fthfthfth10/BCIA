"""Build and independently verify the frozen v3 dataset beside v2."""

import argparse
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from PIL import Image

from potsdam_v2 import ASSETS, PALETTE, checked, exact_labels, file_sha, raster, read_rows, sha


ERODED_HASH = "7bd08018f36a1836c0de7ecb256f7fd677a1ba1e7a27440fdcbad64ce80e383e"
FULL_HASH = "522a36d1196f041bb8d260e0c189f459639e590ad60659e70c81ca4829184670"
LEGACY_HASH = "efacadba7e790d9cf2b84bdb679f98a9e5f4deccef37817bebeccf23fcf12e25"
V2_MANIFESTS = {
    "train": "3508f4f50ea70ef4e8cb1fc14016e263b9aa0483bdbf2099fd09fc2fb0995614",
    "val": "f9ba0fb5fb379693009cee3fc3c5817dc72945f663d5b4a655f7aee5735b4172",
}
NORMALIZATION = "02da0e6cec2ffbe240fed78eb1bc0f70484147e8da8be7a73fd9b9ad8dcd1c3b"


def member(path, name, expected):
    with zipfile.ZipFile(path) as archive:
        names = [entry for entry in archive.namelist() if entry.rsplit("/", 1)[-1] == name]
        if len(names) != 1:
            raise ValueError(f"Expected one {name} in {path.name}")
        raw = archive.read(names[0])
    if sha(raw) != expected:
        raise ValueError(f"Unexpected upstream SHA-256 for {name}")
    return raster(raw, "RGB")


def verify_source(eroded, full, legacy_rgb):
    if any(image.shape != (6000, 6000, 3) for image in (eroded, full, legacy_rgb)):
        raise ValueError("Unexpected source grid")
    candidate, full_labels = exact_labels(eroded), exact_labels(full)
    unknown = np.all(legacy_rgb == (252, 255, 0), axis=-1)
    if int(unknown.sum()) != 183600 or not np.all(full_labels[unknown] == 5):
        raise ValueError("Unexpected noncanonical label region")
    legacy_rgb[unknown] = 0
    legacy = exact_labels(legacy_rgb)
    valid = candidate != 0
    shared = valid & (legacy != 0)
    if not np.array_equal(candidate[valid], full_labels[valid]) or not np.array_equal(
            candidate[shared], legacy[shared]):
        raise ValueError("Unexplained 6_7 class disagreement")
    changes = int((candidate != legacy).sum())
    if changes != 183776 or int(valid.sum()) != 33625472 or int(shared.sum()) != 33441748:
        raise ValueError("Unexpected class or ignore transition counts")
    return candidate, legacy, dict(changed_source_pixels=changes, shared_valid=int(shared.sum()),
                                   full_agreement=int(valid.sum()), noncanonical_legacy=int(unknown.sum()),
                                   transitions=np.bincount((7 * legacy + candidate).ravel(),
                                                           minlength=49).reshape(7, 7).tolist())


def process(v2, root, eroded_archive, full_archive, legacy_tiff, *, build):
    normalization = (v2 / "normalization.json").read_bytes()
    if sha(normalization) != NORMALIZATION:
        raise ValueError("v2 normalization fingerprint mismatch")
    before_metadata = file_sha(v2 / "metadata.json")
    if before_metadata != "3e20ab72bf7d7e8fd9e39d8d37722bbcbd2439e7656f502bf5fe430ae81d2806":
        raise ValueError("Unexpected v2 metadata")
    for split, expected in V2_MANIFESTS.items():
        if file_sha(v2 / "manifests" / f"{split}.jsonl") != expected:
            raise ValueError(f"v2 {split} manifest fingerprint mismatch")
    old_raw = legacy_tiff.read_bytes()
    if sha(old_raw) != LEGACY_HASH:
        raise ValueError("Legacy TIFF fingerprint mismatch")
    candidate, legacy, source = verify_source(
        member(eroded_archive, "top_potsdam_6_7_label_noBoundary.tif", ERODED_HASH),
        member(full_archive, "top_potsdam_6_7_label.tif", FULL_HASH), raster(old_raw, "RGB"))
    if build:
        (root / "manifests").mkdir()
        (root / "labels" / "train").mkdir(parents=True)
        (root / "normalization.json").write_bytes(normalization)
    elif (root / "normalization.json").read_bytes() != normalization:
        raise ValueError("v3 normalization differs")
    scenes, identities, statistics = {}, set(), {}
    replacements = 0
    for split, count, scene_count in (("train", 3456, 24), ("val", 2016, 14)):
        originals = read_rows(v2, split)
        actual = originals if build else read_rows(root, split)
        scenes[split] = {row["scene"] for row in originals}
        if len(originals) != count or len(actual) != count or len(scenes[split]) != scene_count:
            raise ValueError(f"Unexpected {split} split")
        histogram = np.zeros(7, dtype=np.int64)
        before = histogram.copy()
        fractions, expected_rows = [], []
        for old, observed in zip(originals, actual):
            row = dict(old)
            name, scene = row["id"], row["scene"]
            left, top, right, bottom = row["xyxy"]
            if (name in identities or row["split"] != split or
                    name != "_".join(map(str, (scene, left, top, right, bottom))) or
                    not (0 <= left < right <= 6000 and 0 <= top < bottom <= 6000) or
                    (right - left, bottom - top) != (512, 512)):
                raise ValueError(f"Invalid identity or coordinates: {name}")
            identities.add(name)
            for asset in ASSETS:
                path = Path(row[f"{asset}_path"])
                if path.is_absolute():
                    raise ValueError(f"Absolute asset path: {name}")
                row[f"{asset}_path"] = (Path("..") / v2.name / path).as_posix()
            rgb = raster(checked(root, row, "rgb"), "RGB")
            dsm = np.load(io.BytesIO(checked(root, row, "dsm")), allow_pickle=False)
            valid = raster(checked(root, row, "valid"))
            label = raster(checked(root, row, "label"))
            if (rgb.shape != (512, 512, 3) or dsm.shape != (512, 512) or
                    dsm.dtype != np.float32 or not np.isfinite(dsm).all() or
                    valid.shape != dsm.shape or not np.isin(valid, (0, 1)).all() or
                    np.count_nonzero(valid) != row["valid_pixels"] or label.shape != dsm.shape or
                    label.dtype != np.uint8 or label.max() > 6):
                raise ValueError(f"Invalid v2 asset: {name}")
            before += np.bincount(label.ravel(), minlength=7)
            if scene == "6_7":
                if split != "train" or not np.array_equal(label, legacy[top:bottom, left:right]):
                    raise ValueError(f"Legacy 6_7 crop mismatch: {name}")
                row.update(label_path=f"labels/train/{name}.png", v2_label_sha256=old["label_sha256"],
                           label_source_sha256=ERODED_HASH, label_mapping="exact_rgb_palette_0_to_6")
                path = root / row["label_path"]
                expected = candidate[top:bottom, left:right]
                if build:
                    Image.fromarray(expected).save(path)
                raw = path.read_bytes()
                label = raster(raw)
                if label.dtype != np.uint8 or not np.array_equal(label, expected):
                    raise ValueError(f"Incorrect candidate crop: {name}")
                row["label_sha256"] = sha(raw)
                replacements += 1
            if not build and row != observed:
                raise ValueError(f"Unexpected v3 manifest record: {name}")
            if scene == "4_12" and row.get("label_source_sha256") != (
                    "eed5e1c6900c58ceb3054bbdc5e414eb11e941814e818cfde741b1ab6b757947"):
                raise ValueError("Missing verified 4_12 provenance")
            histogram += np.bincount(label.ravel(), minlength=7)
            fractions.append(np.count_nonzero(label) / label.size)
            expected_rows.append(row)
        if build:
            (root / "manifests" / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in expected_rows),
                encoding="utf-8")
        statistics[split] = dict(samples=count, scenes=scene_count, before=before.tolist(),
                                 after=histogram.tolist(), all_ignore=sum(value == 0 for value in fractions),
                                 valid_fraction_min_median_max=np.quantile(fractions, [0, .5, 1]).tolist())
    if (replacements != 144 or scenes["train"] & scenes["val"] or
            scenes["train"] != set(json.loads(normalization)["scene_ids"]) or
            any(item["all_ignore"] for item in statistics.values())):
        raise ValueError("Unexpected v3 replacement coverage, split or all-ignore tile")
    metadata = dict(version="potsdam_rgb_dsm_v3", status="verified", role="formal",
                    base="potsdam_rgb_dsm_v2", base_metadata_sha256=before_metadata,
                    base_manifest_sha256=V2_MANIFESTS, normalization_sha256=NORMALIZATION,
                    manifest_sha256={split: file_sha(root / "manifests" / f"{split}.jsonl")
                                     for split in scenes},
                    replacement=dict(scene="6_7", tiles=replacements, palette=PALETTE,
                                     eroded_archive=eroded_archive.name, full_archive=full_archive.name,
                                     eroded_sha256=ERODED_HASH, full_sha256=FULL_HASH,
                                     legacy_sha256=LEGACY_HASH, **source), statistics=statistics)
    if build:
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    elif json.loads((root / "metadata.json").read_text(encoding="utf-8")) != json.loads(json.dumps(metadata)):
        raise ValueError("v3 metadata mismatch")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    for flag in ("v2", "v3", "eroded-archive", "full-archive", "legacy-tiff"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args()
    v2, v3 = args.v2.resolve(), args.v3.resolve()
    if v2.name != "potsdam_rgb_dsm_v2" or v3.name != "potsdam_rgb_dsm_v3" or v2.parent != v3.parent:
        parser.error("Expected sibling potsdam_rgb_dsm_v2 and potsdam_rgb_dsm_v3 directories")
    inputs = args.eroded_archive, args.full_archive, args.legacy_tiff
    if args.action == "build":
        if v3.exists():
            raise FileExistsError(v3)
        with tempfile.TemporaryDirectory(prefix=".potsdam-v3-", dir=v3.parent) as staging:
            root = Path(staging)
            report = process(v2, root, *inputs, build=True)
            root.rename(v3)
    else:
        report = process(v2, v3, *inputs, build=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
