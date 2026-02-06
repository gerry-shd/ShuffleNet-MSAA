from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SplitSpec:
    """Container for stratified split ratios and RNG seed.
    """
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


def discover_images(data_root: str) -> Tuple[List[str], List[int], Dict[str, int]]:
    """Scan a classification folder structure and yield sorted image paths.

    Args:
        data_root: Root folder that contains one subfolder per class.

    Returns:
        Tuple of (paths, label indices, class name -> index map).
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())  # Deterministic ordering for label assignment.
    if len(class_dirs) == 0:
        raise ValueError(f"No class subfolders found under: {data_root}")

    label_map = {p.name: i for i, p in enumerate(class_dirs)}

    paths: List[str] = []
    labels: List[int] = []
    for cname, idx in label_map.items():
        for p in sorted((root / cname).rglob("*")):
            # Only keep files with supported image suffixes.
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))
                labels.append(idx)

    if len(paths) == 0:
        raise ValueError(f"No images found under: {data_root}")

    return paths, labels, label_map


def make_splits(labels: List[int], spec: SplitSpec) -> Dict[str, List[int]]:
    """Create stratified train/val/test index lists based on class labels.

    Args:
        labels: Integer-encoded target labels.
        spec: Ratios and random seed for reproducible shuffles.

    Returns:
        Dictionary with keys train/val/test mapped to index lists.
    """
    y = np.array(labels)
    n = len(y)
    if abs(spec.train_ratio + spec.val_ratio + spec.test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    idx_all = np.arange(n)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - spec.train_ratio), random_state=spec.seed)
    train_idx, temp_idx = next(sss1.split(idx_all, y))

    y_temp = y[temp_idx]
    val_portion = spec.val_ratio / (spec.val_ratio + spec.test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_portion), random_state=spec.seed)
    val_rel, test_rel = next(sss2.split(np.arange(len(temp_idx)), y_temp))

    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    return {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}


class SonarFolderDataset(Dataset):
    """Minimal Dataset wrapper around the discovered file list.
    """
    def __init__(self, paths: List[str], labels: List[int], indices: List[int], transform=None) -> None:
        """Store references to the shared path/label lists and split indices.
        """
        self.paths = paths
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the addressed split.
        """
        return len(self.indices)

    def __getitem__(self, i: int):
        """Load and transform the image located at the requested index.
        """
        idx = self.indices[i]
        path = self.paths[idx]
        y = int(self.labels[idx])
        img = Image.open(path).convert("RGB")  # Normalize channel count even for grayscale inputs.
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def build_transforms(image_size: int = 224, augment: bool = True,
                     augment_cfg: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
    """Create torchvision transform pipelines for training and evaluation.

    Args:
        image_size: Final square size applied to the images.
        augment: Whether heavy augmentations should be enabled.
        augment_cfg: Dictionary toggles for individual augmentation primitives.

    Returns:
        (train_transform, eval_transform).
    """
    augment_cfg = augment_cfg or {}
    mean = augment_cfg.get("mean", [0.485, 0.456, 0.406])
    std = augment_cfg.get("std", [0.229, 0.224, 0.225])

    train_ops = []
    if augment:
        rrc = augment_cfg.get("random_resized_crop", {})
        scale = tuple(rrc.get("scale", (0.72, 1.0)))
        ratio = tuple(rrc.get("ratio", (0.9, 1.1)))
        train_ops.append(transforms.RandomResizedCrop(image_size, scale=scale, ratio=ratio))

        ra_cfg = augment_cfg.get("randaugment", {})
        if ra_cfg.get("enabled", False):
            train_ops.append(transforms.RandAugment(num_ops=int(ra_cfg.get("num_ops", 2)),
                                                    magnitude=int(ra_cfg.get("magnitude", 9))))
        else:
            flips = augment_cfg.get("flips", {})
            p_h = float(flips.get("horizontal", 0.5))
            p_v = float(flips.get("vertical", 0.2))
            if p_h > 0:
                train_ops.append(transforms.RandomHorizontalFlip(p=p_h))
            if p_v > 0:
                train_ops.append(transforms.RandomVerticalFlip(p=p_v))

            rot_deg = float(augment_cfg.get("rotation", 15.0))
            if rot_deg > 0:
                train_ops.append(transforms.RandomRotation(degrees=rot_deg))

            cj_cfg = augment_cfg.get("color_jitter", {})
            if any(cj_cfg.get(k, 0.0) > 0 for k in ("brightness", "contrast", "saturation", "hue")):
                train_ops.append(transforms.ColorJitter(
                    brightness=float(cj_cfg.get("brightness", 0.15)),
                    contrast=float(cj_cfg.get("contrast", 0.20)),
                    saturation=float(cj_cfg.get("saturation", 0.10)),
                    hue=float(cj_cfg.get("hue", 0.02)),
                ))

        if augment_cfg.get("gaussian_blur", {}).get("enabled", False):
            gb_cfg = augment_cfg["gaussian_blur"]
            train_ops.append(transforms.GaussianBlur(kernel_size=int(gb_cfg.get("kernel_size", 3)),
                                                     sigma=tuple(gb_cfg.get("sigma", (0.1, 2.0)))))
    else:
        train_ops.append(transforms.Resize((image_size, image_size)))

    train_ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    re_cfg = augment_cfg.get("random_erasing", {})
    if re_cfg.get("prob", 0.0) > 0:
        # Random erasing is applied at the very end to simulate occlusions.
        train_ops.append(transforms.RandomErasing(p=float(re_cfg.get("prob", 0.0)),
                                                  scale=tuple(re_cfg.get("scale", (0.02, 0.2))),
                                                  ratio=tuple(re_cfg.get("ratio", (0.3, 3.3))),
                                                  value=re_cfg.get("value", "random")))

    test_ops = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(train_ops), test_ops

def resolve_split_dirs(data_root: str, split_folders: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Resolve relative split folder names against the dataset root.
    """
    split_folders = split_folders or {}
    root = Path(data_root)
    resolved: Dict[str, str] = {}
    for key in ("train", "val", "test"):
        entry = split_folders.get(key, key)
        p = Path(entry)
        if not p.is_absolute():
            p = root / entry
        resolved[key] = str(p)
    return resolved


def discover_predefined_splits(split_dirs: Dict[str, str]) -> Tuple[List[str], List[int], Dict[str, int], Dict[str, List[int]]]:
    """Read explicit train/val/test subfolders and keep their indices.
    """
    if not split_dirs:
        raise ValueError("split_dirs must include entries for train/val/test")
    resolved = {k: Path(v) for k, v in split_dirs.items()}
    missing = [k for k, p in resolved.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Split folders missing: {missing}")

    class_names = sorted({p.name for split_path in resolved.values() for p in split_path.iterdir() if p.is_dir()})
    if not class_names:
        raise ValueError("No class folders found inside provided split directories")
    label_map = {name: idx for idx, name in enumerate(class_names)}

    paths: List[str] = []
    labels: List[int] = []
    splits: Dict[str, List[int]] = {k: [] for k in resolved}

    for split_name, split_path in resolved.items():
        for cname in class_names:
            class_dir = split_path / cname
            # Missing class folders are skipped but keep label positions consistent.
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                    paths.append(str(img_path))
                    labels.append(label_map[cname])
                    splits[split_name].append(len(paths) - 1)

    for split_name, idxs in splits.items():
        if len(idxs) == 0:
            raise ValueError(f"Split '{split_name}' contains no images")
    return paths, labels, label_map, splits
