from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (discover_images, SonarFolderDataset, build_transforms,
                         resolve_split_dirs, discover_predefined_splits)
from models import build_model, ModelSpec
from metrics import compute_metrics, save_confusion_matrix
from utils import ensure_dir, load_json, save_json, load_checkpoint, get_device, load_config


def parse_args() -> argparse.Namespace:
    """Set up the CLI arguments for offline evaluation runs.
    """
    p = argparse.ArgumentParser(description="Evaluate ShuffleNet-MSAA classifier")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--image_size", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    """Run inference over the provided loader and capture predictions/targets.
    """
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for x, y in tqdm(loader, desc="test", leave=False):
        x = x.to(device, non_blocking=True)  # Async transfers to amortize dataloader latency.
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
    return y_true, y_pred


def main() -> None:
    """Load a checkpoint, build the dataloaders, and report evaluation metrics.
    """
    args = parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    data_root = args.data_root or paths_cfg.get("data_root", "datasets")
    out_base = args.out_dir or paths_cfg.get("output_dir", "outputs/experiment")
    out_dir = ensure_dir(Path(out_base) / "eval")

    batch_size = args.batch_size or int(train_cfg.get("batch_size", 64))
    num_workers = args.num_workers or int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    image_size = args.image_size or int(data_cfg.get("image_size", 224))

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    label_map = ckpt.get("label_map")
    if label_map is None:
        raise ValueError("label_map missing in checkpoint.")
    class_names = [None] * len(label_map)
    for k, v in label_map.items():
        class_names[v] = k

    split_mode = str(data_cfg.get("split_mode", "auto")).lower()
    if split_mode not in {"auto", "folders"}:
        raise ValueError("data.split_mode must be 'auto' or 'folders'")
    split_dirs_cfg = data_cfg.get("split_folders", {})
    resolved_split_dirs = resolve_split_dirs(data_root, split_dirs_cfg)
    split_dir_paths = {k: Path(v) for k, v in resolved_split_dirs.items()}
    has_all_split_dirs = all(p.exists() for p in split_dir_paths.values())
    if split_mode == "folders" and not has_all_split_dirs:
        missing = [k for k, p in split_dir_paths.items() if not p.exists()]
        raise FileNotFoundError(f"Expected data split folders missing: {missing}")
    use_predefined = (split_mode == "folders") or (split_mode == "auto" and has_all_split_dirs)

    if use_predefined:
        paths, labels, _, _ = discover_predefined_splits(resolved_split_dirs)
    else:
        paths, labels, _ = discover_images(data_root)

    split_path = Path(out_base) / "splits.json"
    if not split_path.exists():
        raise FileNotFoundError(f"splits.json not found in out_dir: {split_path}")
    splits = load_json(str(split_path))

    _, test_tfms = build_transforms(image_size=image_size, augment=False, augment_cfg=data_cfg.get("augment", {}))
    test_ds = SonarFolderDataset(paths, labels, splits["test"], transform=test_tfms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    device_pref = str(train_cfg.get("device", "auto"))
    device = get_device(device_pref)

    ckpt_args = ckpt.get("args", {})
    if isinstance(ckpt_args, dict):
        model_args = ckpt_args.get("model", {}) if isinstance(ckpt_args.get("model"), dict) else {}
        width_mult = float(model_args.get("width_mult", ckpt_args.get("width_mult", 1.0)))
        drop_path_rate = float(model_args.get("drop_path_rate", ckpt_args.get("drop_path_rate", 0.0)))
    else:
        width_mult = 1.0
        drop_path_rate = 0.0
    model = build_model(ModelSpec(name="shufflenet_msaa", num_classes=len(label_map), width_mult=width_mult, drop_path_rate=drop_path_rate))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    y_true, y_pred = predict_all(model, test_loader, device)
    res = compute_metrics(y_true, y_pred, class_names)

    save_json(str(out_dir / "test_metrics.json"), {
        "top1": res["top1"],
        "macro_f1": res["macro_f1"],
        "weighted_f1": res["weighted_f1"],
        "classification_report": res["report"],
    })

    cm_png = out_dir / "confusion_matrix.png"
    cm_csv = out_dir / "confusion_matrix.csv"
    save_confusion_matrix(res["cm"], class_names, str(cm_png), str(cm_csv))

    print(f"Top-1 accuracy: {res['top1']*100:.2f}%")
    print(f"Macro-F1:       {res['macro_f1']*100:.2f}%")
    print(f"Weighted-F1:    {res['weighted_f1']*100:.2f}%")
    print(f"Saved: {out_dir / 'test_metrics.json'}")
    print(f"Saved: {cm_png}")
    print(f"Saved: {cm_csv}")


if __name__ == "__main__":
    main()
