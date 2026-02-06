import copy
import os
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve the preferred compute device while honoring availability."""
    pref = (preferred or "auto").strip().lower()
    if pref not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown device preference: {preferred}")
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(p: str) -> Path:
    """Create the directory if it does not exist and return a Path object."""
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str, obj: Any) -> None:
    """Serialize obj to JSON with UTF-8 encoding, creating parent folders as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Read a UTF-8 encoded JSON file and deserialize it into Python objects."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


class AverageMeter:
    """Track running sums/counts so we can report averages without storing all values."""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Zero out the accumulated statistics."""
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Accumulate val * n into the running sum and increment the counter."""
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        """Return the average value accumulated so far."""
        return self.sum / max(1, self.count)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy for a batch without tracking gradients."""
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        return float((pred == targets).float().mean().item())


def cosine_with_warmup_lr(optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int):
    """Build a LambdaLR schedule that performs linear warmup followed by cosine decay."""
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, best_val: float, label_map: Dict[str, int], args: Dict[str, Any],
                    extra: Optional[Dict[str, Any]] = None) -> None:
    """Persist model/optimizer state for later resumption or evaluation."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "label_map": label_map,
        "args": args,
        "extra": extra or {},
        "torch_version": torch.__version__,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)

def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a checkpoint dictionary from disk with an optional map_location."""
    return torch.load(path, map_location=map_location)


class ModelEMA:
    """Maintain an exponential moving average copy of the model for evaluation stability."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.9997,
                 device: Optional[torch.device] = None, warmup_steps: int = 100) -> None:
        """Clone model parameters and configure EMA decay/warmup behaviour."""
        self.decay = float(decay)
        self.warmup_steps = int(max(1, warmup_steps)) if warmup_steps > 0 else 0
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.ema.to(device)
        self.num_updates = 0

    def _get_decay(self) -> float:
        """Gradually ramp the EMA decay factor during the warmup period."""
        if self.warmup_steps <= 0:
            return self.decay
        warm = 1.0 - math.exp(-self.num_updates / float(self.warmup_steps))
        return self.decay * warm

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Blend EMA parameters with the latest model weights using the computed decay."""
        self.num_updates += 1
        d = self._get_decay()
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1.0 - d)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)

    def state_dict(self) -> Dict[str, Any]:
        """Return a serialization-friendly snapshot of the EMA configuration/state."""
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "num_updates": self.num_updates,
            "ema_state": self.ema.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore EMA parameters and metadata from a serialized snapshot."""
        self.decay = float(state.get("decay", self.decay))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.num_updates = int(state.get("num_updates", 0))
        self.ema.load_state_dict(state["ema_state"])
