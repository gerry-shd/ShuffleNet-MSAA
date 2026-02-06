from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Beta
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataset import (discover_images, make_splits, SplitSpec, SonarFolderDataset, build_transforms,
                     resolve_split_dirs, discover_predefined_splits)
from models import build_model, ModelSpec
from losses import FocalLoss, DWFLoss
from utils import (seed_everything, ensure_dir, save_json, load_json, AverageMeter,
                   accuracy_top1, cosine_with_warmup_lr, save_checkpoint, get_device,
                   load_config, ModelEMA)
from config import resolve_pretrained, load_pretrained_weights, freeze_pretrained_layers


class MixupCutmix:
    """Callable helper that applies mixup/cutmix based on configured probabilities."""
    def __init__(self, mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0,
                 prob: float = 0.0, switch_prob: float = 0.5) -> None:
        """Cache augmentation hyper-parameters so we can adjust at runtime."""
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.prob = float(prob)
        self.switch_prob = float(switch_prob)

    def set_prob(self, prob: float) -> None:
        """Update the probability of applying mixup/cutmix for the current epoch."""
        self.prob = float(max(0.0, min(1.0, prob)))

    def _sample_lambda(self, alpha: float) -> float:
        """Draw a lambda value from the Beta distribution (or 1.0 if disabled)."""
        if alpha <= 0:
            return 1.0
        lam = Beta(alpha, alpha).sample().item()
        return float(max(0.0, min(1.0, lam)))

    def _apply_cutmix(self, x: torch.Tensor, indices: torch.Tensor, lam: float) -> tuple[torch.Tensor, float]:
        """Perform CutMix by replacing a random patch and return the adjusted lambda."""
        b, c, h, w = x.size()
        if h < 2 or w < 2:
            return x, 1.0
        cut_ratio = math.sqrt(max(1e-7, 1.0 - lam))
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        cx = torch.randint(0, w, (1,), device=x.device).item()
        cy = torch.randint(0, h, (1,), device=x.device).item()
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, h)
        if x1 >= x2 or y1 >= y2:
            return x, 1.0
        x[:, :, y1:y2, x1:x2] = x[indices, :, y1:y2, x1:x2]
        lam_new = 1.0 - ((x2 - x1) * (y2 - y1) / float(w * h))
        return x, float(lam_new)

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """Apply mixup or cutmix to a batch and return mixed targets when used."""
        if x.size(0) < 2 or self.prob <= 0.0 or (self.mixup_alpha <= 0.0 and self.cutmix_alpha <= 0.0):
            return x, y, None, None
        if torch.rand(1).item() > self.prob:
            return x, y, None, None

        use_cutmix = self.cutmix_alpha > 0.0 and (self.mixup_alpha <= 0.0 or torch.rand(1).item() < self.switch_prob)
        indices = torch.randperm(x.size(0), device=x.device)
        y_perm = y[indices]

        if use_cutmix:
            lam = self._sample_lambda(self.cutmix_alpha)
            x, lam = self._apply_cutmix(x, indices, lam)
        else:
            lam = self._sample_lambda(self.mixup_alpha)
            lam = float(lam)
            x = x * lam + x[indices] * (1.0 - lam)

        return x, y, y_perm, lam


def parse_args() -> argparse.Namespace:
    """Set up CLI arguments for the training entry point."""
    p = argparse.ArgumentParser(description="Train ShuffleNet-MSAA classifier with advanced augments")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    p.add_argument("--no_amp", action="store_true", help="Force disable AMP regardless of config")
    return p.parse_args()


def mixup_criterion(criterion: nn.Module, logits: torch.Tensor,
                    target_a: torch.Tensor, target_b: Optional[torch.Tensor], lam: Optional[float]) -> torch.Tensor:
    """Blend losses when mixup/cutmix produced two targets."""
    if target_b is None or lam is None:
        return criterion(logits, target_a)
    return lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler: torch.cuda.amp.GradScaler,
        use_amp: bool,
        mixup_fn: Optional[MixupCutmix],
        ema: Optional[ModelEMA],
        mixup_active: bool,
) -> Dict[str, float]:
    """Run a full training epoch including optional mixup and EMA tracking."""
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for x, y in tqdm(loader, desc="train", leave=True):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        targets_b = None
        lam = None
        apply_mix = mixup_fn is not None and mixup_active
        if apply_mix:
            x, y, targets_b, lam = mixup_fn(x, y)

        optimizer.zero_grad(set_to_none=True)  # Cheaper than setting grads to zero manually.
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            logits = model(x)
            loss = mixup_criterion(criterion, logits, y, targets_b, lam)

        scaler.scale(loss).backward()  # Scale gradients to keep FP16 numerically stable.
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        loss_m.update(loss.item(), n=x.size(0))
        if not apply_mix or targets_b is None or lam is None:
            acc_m.update(accuracy_top1(logits.detach(), y), n=x.size(0))

    return {"loss": loss_m.avg, "acc": acc_m.avg}


def evaluate(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        use_amp: bool,
) -> Dict[str, float]:
    """Validate the model on the held-out split without gradient tracking."""
    model.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    with torch.no_grad():
        for x, y in tqdm(loader, desc="val", leave=True):
            # Validation never modifies weights, so gradients remain disabled.
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            loss_m.update(loss.item(), n=x.size(0))
            acc_m.update(accuracy_top1(logits, y), n=x.size(0))

    return {"loss": loss_m.avg, "acc": acc_m.avg}


def main() -> None:
    """High-level training orchestration covering data prep, optimizers, and checkpoints."""
    args = parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    data_root = paths_cfg.get("data_root", "datasets")
    out_dir = ensure_dir(paths_cfg.get("output_dir", "outputs/experiment"))
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed)

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

    split_path = out_dir / "splits.json"
    if use_predefined:
        paths, labels, label_map, splits = discover_predefined_splits(resolved_split_dirs)
        save_json(str(split_path), splits)
    else:
        paths, labels, label_map = discover_images(data_root)
        if split_path.exists():
            splits = load_json(str(split_path))
        else:
            split_cfg = data_cfg.get("splits", {})
            spec = SplitSpec(train_ratio=float(split_cfg.get("train", 0.7)),
                             val_ratio=float(split_cfg.get("val", 0.15)),
                             test_ratio=float(split_cfg.get("test", 0.15)),
                             seed=seed)
            splits = make_splits(labels, spec)
            save_json(str(split_path), splits)

    save_json(str(out_dir / "label_map.json"), label_map)

    image_size = int(data_cfg.get("image_size", 224))
    augment_cfg = dict(data_cfg.get("augment", {}))
    augment_enabled = augment_cfg.pop("enabled", True)
    train_tfms, test_tfms = build_transforms(image_size=image_size, augment=augment_enabled, augment_cfg=augment_cfg)

    train_ds = SonarFolderDataset(paths, labels, splits["train"], transform=train_tfms)
    val_ds = SonarFolderDataset(paths, labels, splits["val"], transform=test_tfms)

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", train_cfg.get("num_workers", 4)))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    device_pref = str(train_cfg.get("device", "auto"))
    device = get_device(device_pref)

    model_cfg = cfg.get("model", {})
    model = build_model(ModelSpec(name=model_cfg.get("name", "shufflenet_msaa"),
                                  num_classes=len(label_map),
                                  width_mult=float(model_cfg.get("width_mult", 1.0)),
                                  drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0))))
    pretrained_name = model_cfg.get("pretrained", "none")
    pretrained_cfg = resolve_pretrained(pretrained_name)
    pretrained_loaded = False
    if pretrained_cfg.get("enabled", False) and pretrained_cfg.get("name") != "none":
        pretrained_loaded = load_pretrained_weights(model, pretrained_cfg)
        if pretrained_loaded:
            freeze_pretrained_layers(model, pretrained_cfg)
    model.to(device)

    loss_name = train_cfg.get("loss", "dwfl").lower()
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    if loss_name == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        extra = {"loss": "ce", "label_smoothing": label_smoothing}
    elif loss_name == "focal":
        criterion = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)))
        extra = {"loss": "focal", "focal_gamma": train_cfg.get("focal_gamma", 2.0)}
    else:
        class_counts = [0] * len(label_map)
        for idx in splits["train"]:
            class_counts[int(labels[idx])] += 1
        criterion = DWFLoss(class_counts=class_counts,
                            alpha=float(train_cfg.get("dwfl_alpha", 0.25)),
                            gamma=float(train_cfg.get("dwfl_gamma", 2.0)))
        extra = {
            "loss": "dwfl",
            "dwfl_alpha": train_cfg.get("dwfl_alpha", 0.25),
            "dwfl_gamma": train_cfg.get("dwfl_gamma", 2.0),
            "class_counts": class_counts,
        }

    extra["pretrained"] = {"name": pretrained_cfg.get("name"), "loaded": bool(pretrained_loaded)}

    criterion = criterion.to(device)

    opt_cfg = train_cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(opt_cfg.get("lr", 3e-4)),
                                  weight_decay=float(opt_cfg.get("weight_decay", 1e-4)))

    epochs = int(train_cfg.get("epochs", 120))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
    scheduler = cosine_with_warmup_lr(optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs)

    amp_from_cfg = bool(train_cfg.get("amp", True))
    use_amp = (device.type == "cuda") and amp_from_cfg and (not args.no_amp)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    mixup_cfg = train_cfg.get("mixup", {})
    mixup_fn: Optional[MixupCutmix] = None
    mixup_base_prob = 0.0
    mixup_start_epoch = 1
    mixup_stop_epoch = epochs
    mixup_final_prob = 0.0
    mixup_schedule = "constant"
    if mixup_cfg.get("enabled", False):
        mixup_base_prob = float(mixup_cfg.get("prob", 0.7))
        mixup_start_epoch = int(mixup_cfg.get("start_epoch", 1))
        mixup_stop_epoch = int(mixup_cfg.get("stop_epoch", epochs))
        mixup_final_prob = float(mixup_cfg.get("final_prob", mixup_base_prob))
        mixup_schedule = str(mixup_cfg.get("schedule", "constant")).lower()
        mixup_fn = MixupCutmix(mixup_alpha=float(mixup_cfg.get("mixup_alpha", 0.8)),
                               cutmix_alpha=float(mixup_cfg.get("cutmix_alpha", 1.0)),
                               prob=mixup_base_prob,
                               switch_prob=float(mixup_cfg.get("switch_prob", 0.5)))
        extra.update({"mixup": mixup_cfg})

    ema_cfg = train_cfg.get("ema", {})
    ema: Optional[ModelEMA] = None
    if ema_cfg.get("enabled", False):
        ema = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.9997)),
                       device=device, warmup_steps=int(ema_cfg.get("warmup_steps", 100)))
        extra.update({"ema": {k: v for k, v in ema_cfg.items() if k in {"decay", "warmup_steps"}}})

    csv_path = out_dir / "train_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    best_val = -1.0
    best_epoch = -1
    patience = int(train_cfg.get("early_stop_patience", 25))
    patience_left = patience

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        mix_prob_now = 0.0
        mix_active = False
        if mixup_fn is not None:
            if mixup_start_epoch <= epoch <= mixup_stop_epoch:
                mix_active = True
                if mixup_schedule == "linear":
                    span = max(1, mixup_stop_epoch - mixup_start_epoch)
                    frac = min(1.0, max(0.0, (epoch - mixup_start_epoch) / span))
                    mix_prob_now = mixup_base_prob + frac * (mixup_final_prob - mixup_base_prob)
                else:
                    mix_prob_now = mixup_base_prob
            mix_prob_now = max(0.0, min(1.0, mix_prob_now))
            mixup_fn.set_prob(mix_prob_now)
            mix_active = mix_active and (mix_prob_now > 0.0)
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, mixup_fn, ema, mix_active)
        val_model = ema.ema if ema is not None else model
        va = evaluate(val_model, val_loader, criterion, device, use_amp=False)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([epoch, tr["loss"], tr["acc"], va["loss"], va["acc"], lr_now])

        print(f"  train loss {tr['loss']:.4f} acc {tr['acc']*100:6.2f}% | "
              f"val loss {va['loss']:.4f} acc {va['acc']*100:6.2f}% | lr {lr_now:.6f} | mix {mix_prob_now:.2f}")

        if va["acc"] > best_val:
            best_val = float(va["acc"])
            best_epoch = epoch
            patience_left = patience
            model_to_save = ema.ema if ema is not None else model
            save_checkpoint(str(out_dir / "best.pt"), model_to_save, optimizer, epoch, best_val,
                            label_map, cfg, extra=extra)
        else:
            patience_left -= 1

        save_checkpoint(str(out_dir / "last.pt"), model, optimizer, epoch, best_val, label_map, cfg, extra=extra)

        if patience_left <= 0:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best val acc {best_val*100:.2f}% at epoch {best_epoch}.")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()



