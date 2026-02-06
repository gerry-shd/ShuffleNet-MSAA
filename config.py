from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"

PRETRAINED_MODELS: Dict[str, Dict[str, Any]] = {
    "none": {
        "enabled": False,
        "description": "Train from scratch without loading external weights.",
    },
    "torchvision_resnet50": {
        "enabled": True,
        "arch": "resnet50",
        "source": "torchvision",
        "torchvision_url": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "strict_loading": False,
        "reset_classifier": True,
        "freeze_stem": True,
        "description": "Torchvision ResNet-50 ImageNet weights.",
    },
    "torchvision_mobilenetv3_large": {
        "enabled": True,
        "arch": "mobilenet_v3_large",
        "source": "torchvision",
        "torchvision_url": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        "strict_loading": False,
        "reset_classifier": True,
        "freeze_stem": False,
        "description": "Torchvision MobileNetV3-Large ImageNet weights.",
    },
    "torchvision_shufflenetv2_x1_0": {
        "enabled": True,
        "arch": "shufflenetv2_x1.0",
        "source": "torchvision",
        "torchvision_url": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
        "strict_loading": False,
        "reset_classifier": True,
        "freeze_stem": False,
        "description": "Torchvision ShuffleNetV2 x1.0 ImageNet weights.",
    },
    "torchvision_vit_b16": {
        "enabled": True,
        "arch": "vit_b_16",
        "source": "torchvision",
        "torchvision_url": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
        "strict_loading": False,
        "reset_classifier": True,
        "freeze_stem": False,
        "description": "Torchvision ViT-B/16 Transformer weights.",
    },
    "torchvision_swin_tiny": {
        "enabled": True,
        "arch": "swin_t",
        "source": "torchvision",
        "torchvision_url": "https://download.pytorch.org/models/swin_t-704ceda7.pth",
        "strict_loading": False,
        "reset_classifier": True,
        "freeze_stem": False,
        "description": "Torchvision Swin Transformer Tiny weights.",
    },
}


def resolve_pretrained(name: str | None) -> Dict[str, Any]:
    """Return a normalized pretrained configuration entry.

    Args:
        name: Friendly key provided via CLI/config (case-insensitive).

    Returns:
        Dict that mirrors PRETRAINED_MODELS plus the resolved key.
    """
    key = (name or "none").lower()
    cfg = PRETRAINED_MODELS.get(key)
    if cfg is None:
        raise KeyError(f"Unknown pretrained model key: {name}")
    out = dict(cfg)
    out["name"] = key
    return out


def _load_state_from_path(path: Path) -> Dict[str, Any]:
    """Load a checkpoint from disk and unwrap the 'model' key if present.
    """
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def _download_state_dict(url: str) -> Dict[str, Any]:
    """Download weights directly from a Torch Hub URL.
    """
    return torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)


def load_pretrained_weights(model: torch.nn.Module, cfg: Dict[str, Any]) -> bool:
    """Attempt to hydrate the model with pretrained weights according to cfg.

    Returns:
        True if any weights were loaded successfully, False otherwise.
    """
    if not cfg.get("enabled", False):
        return False
    candidates = []
    if cfg.get("path"):
        candidates.append(("local", Path(cfg["path"])))  # User-provided checkpoint takes priority.
    if cfg.get("torchvision_url"):
        candidates.append(("torchvision", cfg["torchvision_url"]))  # Fallback to official weights.

    state_dict = None
    source_desc = None
    for kind, src in candidates:
        # Try each candidate until one succeeds, preferring local files.
        try:
            if kind == "local":
                if src.is_file():
                    state_dict = _load_state_from_path(src)
                    source_desc = str(src)
                    break
                else:
                    print(f"[pretrained] local file missing: {src}")
            else:
                print(f"[pretrained] downloading torchvision weights: {src}")
                state_dict = _download_state_dict(src)
                source_desc = src
                break
        except Exception as exc:
            print(f"[pretrained] Failed to load from {src}: {exc}")
            state_dict = None
            continue

    if state_dict is None:
        print("[pretrained] no weights loaded.")
        return False

    if cfg.get("reset_classifier", True):
        for key in list(state_dict.keys()):
            if key.startswith("classifier.") or key.startswith("fc."):
                state_dict.pop(key, None)  # Force a fresh classifier head for new datasets.
    strict = bool(cfg.get("strict_loading", False))
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if not strict and (missing or unexpected):
        print(f"[pretrained] loaded with missing={missing} unexpected={unexpected}")
    print(f"[pretrained] Loaded {cfg.get('arch', 'model')} weights from {source_desc}")
    return True


def freeze_pretrained_layers(model: torch.nn.Module, cfg: Dict[str, Any]) -> None:
    """Freeze early layers when fine-tuning to avoid destroying pretrained filters.
    """
    if not cfg.get("enabled", False):
        return
    if cfg.get("freeze_stem", False) and hasattr(model, "stem"):
        for p in model.stem.parameters():
            p.requires_grad_(False)
        print("[pretrained] Stem frozen.")
