import json
from pathlib import Path

import mmcv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .model import IGNORE_INDEX, NUM_CLASSES, TILE_SIZE


RGB_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
RGB_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class PotsdamValidation(Dataset):
    """Adapted deterministic validation subset of MsRE's PotsdamRGBDSMDataset."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "verified":
            raise ValueError("Expected a verified prepared MsRE dataset")
        manifest = self.root / "manifests" / "val.jsonl"
        self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        stats = json.loads((self.root / "normalization.json").read_text(encoding="utf-8"))
        self.mean, self.std = np.float32(stats["mean"]), np.float32(stats["std"])
        if (stats["split"] != "train" or not np.isfinite(self.mean) or
                not np.isfinite(self.std) or self.std <= 0):
            raise ValueError("Expected finite, train-only normalization statistics")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
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
        rgb = (rgb[..., ::-1].astype(np.float32) - RGB_MEAN) / RGB_STD
        supported = valid.astype(bool) & np.isfinite(dsm)
        height = np.zeros_like(dsm)
        with np.errstate(over="ignore", invalid="ignore"):
            height[supported] = (dsm[supported] - self.mean) / self.std
        supported &= np.isfinite(height)
        height[~supported] = 0.0
        label = np.where(label == 0, IGNORE_INDEX, label.astype(np.int64) - 1)
        return dict(rgb=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
                    dsm=torch.from_numpy(height[None]),
                    valid=torch.from_numpy(supported[None]),
                    label=torch.from_numpy(label))
