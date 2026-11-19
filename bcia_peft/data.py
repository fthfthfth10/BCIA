import json
from pathlib import Path

import mmcv
import numpy as np
import torch
from PIL import Image
from mmseg.datasets.transforms import PhotoMetricDistortion, RandomCrop, RandomFlip, Resize
from torch.utils.data import Dataset

from .model import IGNORE_INDEX, NUM_CLASSES, TILE_SIZE


RGB_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
RGB_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class SynchronizedGeometry:
    """MsRE geometry with mask-aware DSM interpolation and shared crop/flip."""

    def __init__(self, ratio_range=(0.5, 2.0), flip_prob=0.5, cat_max_ratio=0.75):
        self.ratio_range = ratio_range
        self.crop = RandomCrop(crop_size=(TILE_SIZE, TILE_SIZE), cat_max_ratio=cat_max_ratio)
        self.flip = RandomFlip(prob=float(flip_prob), direction="horizontal")

    def __call__(self, rgb, dsm, valid, label):
        extent = int(TILE_SIZE * np.random.uniform(*self.ratio_range))
        result = Resize(scale=(extent, extent), keep_ratio=True)(
            dict(img=rgb, gt_seg_map=label, seg_fields=["gt_seg_map"]))
        height, width = result["img"].shape[:2]
        weights = mmcv.imresize(valid.astype(np.float32), (width, height), interpolation="bilinear")
        numerator = mmcv.imresize(np.where(valid, dsm, 0.0), (width, height), interpolation="bilinear")
        result["dsm"] = np.divide(numerator, weights, out=np.zeros_like(numerator),
                                  where=weights > 1e-7)
        result["valid"] = (weights >= 1.0 - 1e-6).astype(np.uint8)
        bbox = self.crop.crop_bbox(result)
        result["seg_fields"] = ["gt_seg_map", "dsm", "valid"]
        for field in ("img", *result["seg_fields"]):
            result[field] = self.crop.crop(result[field], bbox)
        result["img_shape"] = result["img"].shape[:2]
        result = self.flip(result)
        return result["img"], result["dsm"], result["valid"].astype(bool), result["gt_seg_map"]


class PotsdamSlices(Dataset):
    """Read existing MsRE Eroded tiles and apply training-only augmentation."""

    def __init__(self, root, split, *, geometry=None, photometric=True):
        if split not in ("train", "val"):
            raise ValueError("Expected train or val")
        self.split = split
        self.root = Path(root).resolve()
        metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "verified":
            raise ValueError("Expected a verified prepared MsRE dataset")
        manifest = self.root / "manifests" / f"{split}.jsonl"
        self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        stats = json.loads((self.root / "normalization.json").read_text(encoding="utf-8"))
        self.mean, self.std = np.float32(stats["mean"]), np.float32(stats["std"])
        self.geometry = (geometry or SynchronizedGeometry()) if split == "train" else None
        self.photo = PhotoMetricDistortion() if split == "train" and photometric else None
        if (stats["split"] != "train" or not np.isfinite(self.mean) or
                not np.isfinite(self.std) or self.std <= 0):
            raise ValueError("Expected finite, train-only normalization statistics")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, seed = index
            state = np.random.get_state()
            try:
                np.random.seed(seed)
                return self._load(index)
            finally:
                np.random.set_state(state)
        return self._load(index)

    def _load(self, index):
        row = self.rows[index]
        rgb = mmcv.imread(str(self.root / row["rgb_path"]), flag="color", channel_order="bgr")
        dsm = np.load(self.root / row["dsm_path"], allow_pickle=False)
        with Image.open(self.root / row["valid_path"]) as image:
            valid = np.array(image)
        with Image.open(self.root / row["label_path"]) as image:
            label = np.array(image)
        shape = (TILE_SIZE, TILE_SIZE)
        if rgb is None or rgb.shape != (*shape, 3) or dsm.shape != shape or dsm.dtype != np.float32:
            raise ValueError(f"Invalid RGB or DSM: {row['id']}")
        if valid.shape != shape or not np.isin(valid, [0, 1]).all():
            raise ValueError(f"Invalid DSM mask: {row['id']}")
        if label.shape != shape or label.dtype != np.uint8 or not np.isin(label, range(NUM_CLASSES + 1)).all():
            raise ValueError(f"Invalid label: {row['id']}")
        label = np.where(label == 0, IGNORE_INDEX, label.astype(np.int64) - 1).astype(np.uint8)
        supported = valid.astype(bool) & np.isfinite(dsm)
        if self.geometry is not None:
            rgb, dsm, supported, label = self.geometry(rgb, dsm, supported, label)
        if self.photo is not None:
            rgb = self.photo(dict(img=rgb))["img"]
        rgb = (rgb[..., ::-1].astype(np.float32) - RGB_MEAN) / RGB_STD
        height = np.zeros_like(dsm)
        with np.errstate(over="ignore", invalid="ignore"):
            height[supported] = (dsm[supported] - self.mean) / self.std
        supported &= np.isfinite(height)
        height[~supported] = 0.0
        rows, cols = height.shape
        padding = ((0, TILE_SIZE - rows), (0, TILE_SIZE - cols))
        rgb = np.pad(rgb, (*padding, (0, 0)))
        height = np.pad(height, padding)
        supported = np.pad(supported, padding)
        label = np.pad(label.astype(np.int64), padding, constant_values=IGNORE_INDEX)
        return dict(rgb=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
                    dsm=torch.from_numpy(height[None]),
                    valid=torch.from_numpy(supported[None]),
                    label=torch.from_numpy(label), id=row["id"])


class PotsdamValidation(PotsdamSlices):
    def __init__(self, root):
        super().__init__(root, "val")
