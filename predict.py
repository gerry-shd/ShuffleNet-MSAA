from __future__ import annotations

import argparse
import torch
from PIL import Image

from dataset import build_transforms
from models import build_model, ModelSpec
from utils import load_checkpoint, get_device, load_config


def parse_args() -> argparse.Namespace:
    """Build the CLI needed for single-image inference.

    Returns:
        Parsed namespace produced by argparse.
    """
    p = argparse.ArgumentParser(description="Predict a single image with ShuffleNet-MSAA")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--image_size", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    """Run end-to-end prediction for one image using a trained checkpoint.
    """
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    image_size = args.image_size or int(data_cfg.get("image_size", 224))

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    label_map = ckpt.get("label_map")
    if label_map is None:
        raise ValueError("label_map missing in checkpoint.")
    idx_to_name = {v: k for k, v in label_map.items()}

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

    device_pref = str(train_cfg.get("device", "auto"))
    device = get_device(device_pref)
    model.to(device).eval()

    _, test_tfms = build_transforms(image_size=image_size, augment=False, augment_cfg=data_cfg.get("augment", {}))
    # Ensure the inference transform receives a 3-channel tensor.
    img = Image.open(args.image).convert("RGB")
    x = test_tfms(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)  # Convert logits to normalized probabilities.
    topk = min(args.topk, probs.numel())
    vals, idxs = torch.topk(probs, k=topk)

    print(f"Image: {args.image}")
    for v, i in zip(vals.tolist(), idxs.tolist()):
        print(f"{idx_to_name[i]:>20s}  prob={v:.4f}")


if __name__ == "__main__":
    main()
