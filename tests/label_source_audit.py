"""Read-only Potsdam label audit. JSON includes aggregates, never image data."""

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


PALETTE = ((0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 255),
           (0, 255, 0), (255, 255, 0), (255, 0, 0))


def packed(rgb):
    return (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]


def decode(rgb):
    numbers = packed(rgb)
    exact = np.full(numbers.shape, 7, np.uint8)
    legacy = np.zeros(numbers.shape, np.uint8)
    scores = sum(rgb[..., channel].astype(np.uint16) * weight
                 for channel, weight in enumerate((4, 3, 2)))
    for index, color in enumerate(PALETTE):
        red, green, blue = color
        exact[numbers == (red << 16 | green << 8 | blue)] = index
        legacy[scores == 4 * red + 3 * green + 2 * blue] = index
    return exact, legacy


def tif_info(raw):
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode != "RGB":
            raise ValueError(f"Unexpected TIFF mode {image.mode}")
        return np.array(image), dict(size=image.size, mode=image.mode, bytes=len(raw),
                                     sha256=hashlib.sha256(raw).hexdigest(),
                                     grid={key: repr(image.tag_v2[key]) for key in
                                           (274, 33550, 33922, 34264, 34735) if key in image.tag_v2})


def source_stats(rgb):
    exact, legacy = decode(rgb)
    return exact, legacy, dict(exact_counts=np.bincount(exact.ravel(), minlength=8).tolist(),
                               legacy_counts=np.bincount(legacy.ravel(), minlength=7).tolist(),
                               collisions=int(((exact == 7) & (legacy != 0)).sum()))


def alignment(candidate, eroded):
    if candidate.shape != eroded.shape:
        raise ValueError("Different grids")
    eroded_valid = (eroded > 0) & (eroded < 7)
    candidate_valid = (candidate > 0) & (candidate < 7)
    both = eroded_valid & candidate_valid
    return dict(shared=int(both.sum()), equal=int((both & (candidate == eroded)).sum()),
                full_only=int((candidate_valid & ~eroded_valid).sum()),
                eroded_only=int((eroded_valid & ~candidate_valid).sum()),
                disagreements=np.bincount((eroded[both]*7 + candidate[both]), minlength=49)
                .reshape(7, 7).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    scenes, grouped = {}, {}
    for split in ("train", "val"):
        manifest = args.prepared / "manifests" / f"{split}.jsonl"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            grouped.setdefault((split, row["scene"]), []).append(row)
    focus, focus_legacy = None, None
    for (split, scene), rows in sorted(grouped.items()):
        name = f"top_potsdam_{scene}_label_noBoundary.tif"
        rgb, info = tif_info((args.source / "Label" / name).read_bytes())
        exact, legacy, counts = source_stats(rgb)
        tiles, all_ignore, severe = [], [], []
        for row in rows:
            path = args.prepared / row["label_path"]
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["label_sha256"]:
                raise ValueError(f"Label fingerprint mismatch: {row['id']}")
            with Image.open(io.BytesIO(raw)) as image:
                png = np.array(image)
            left, top, right, bottom = row["xyxy"]
            if not np.array_equal(png, legacy[top:bottom, left:right]):
                raise ValueError(f"Legacy mapping mismatch: {row['id']}")
            fraction = np.count_nonzero(png) / png.size
            tiles.append(fraction)
            if fraction == 0:
                all_ignore.append(row["id"])
            if fraction < .01:
                severe.append(row["id"])
        scenes[scene] = dict(split=split, info=info, **counts, tiles=tiles,
                             all_ignore=all_ignore, severe=severe)
        if scene == "4_12":
            focus, focus_legacy = exact, legacy
            photo, photo_info = tif_info((args.source / "Top/top_potsdam_4_12_RGB.tif").read_bytes())
            unknown = exact == 7
            values, frequencies = np.unique(packed(rgb)[unknown], return_counts=True)
            top_colors = sorted(zip(frequencies.tolist(), values.tolist()), reverse=True)[:12]
            quadrants = [[np.count_nonzero(block) for block in np.hsplit(quad, 2)]
                         for quad in np.vsplit(unknown, 2)]
            overlay = dict(unique_unknown=len(values),
                           frequent_unknown=[(f"{value:06x}", count) for count, value in top_colors],
                           unknown_quadrants=quadrants, rgb=photo_info,
                           same_rgb_pixels=int((np.all(photo == rgb, axis=-1) & unknown).sum()),
                           channel_correlations=[float(np.corrcoef(
                               photo[::4, ::4, channel][unknown[::4, ::4]],
                               rgb[::4, ::4, channel][unknown[::4, ::4]])[0, 1])
                               for channel in range(3)])
    candidates, known = [], {}

    def review(origin, raw):
        digest = hashlib.sha256(raw).hexdigest()
        if digest not in known:
            rgb, info = tif_info(raw)
            exact, _, counts = source_stats(rgb)
            known[digest] = dict(**info, **counts, exact_overlap=alignment(exact, focus),
                                 legacy_overlap=alignment(exact, focus_legacy))
            if not (exact == 7).any() and digest != scenes["4_12"]["info"]["sha256"]:
                horizontal = exact[:, 1:] != exact[:, :-1]
                vertical = exact[1:, :] != exact[:-1, :]
                edge = np.zeros_like(exact, dtype=bool)
                edge[:, 1:] |= horizontal
                edge[:, :-1] |= horizontal
                edge[1:, :] |= vertical
                edge[:-1, :] |= vertical
                valid = (focus > 0) & (focus < 7)
                mismatch = valid & (exact > 0) & (exact != focus)
                known[digest]["boundary"] = dict(edge_pixels=int(edge.sum()),
                                                 eroded_valid_on_edge=int((valid & edge).sum()),
                                                 mismatches_on_edge=int((mismatch & edge).sum()))
        candidates.append(dict(origin=origin, **known[digest]))

    for folder in ("Label", "Label_full", "5_Labels_all"):
        for path in (args.source / folder).glob("top_potsdam_4_12_label*.tif"):
            review(f"{folder}/{path.name}", path.read_bytes())
    for index, folder in enumerate(args.archive_dir, 1):
        for path in sorted(folder.glob("5_Labels*.zip")):
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if re.fullmatch(r"top_potsdam_4_12_label(?:_noBoundary)?\.tif",
                                    member.rsplit("/", 1)[-1], re.IGNORECASE):
                        review(f"archive-dir-{index}/{path.name}/{member}", archive.read(member))
    print(json.dumps(dict(scenes=scenes, candidates=candidates, overlay=overlay), allow_nan=False))


if __name__ == "__main__":
    main()
