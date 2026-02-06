from pathlib import Path

def add_doc(text, header, doc, indent):
    pattern = header + "\n" + indent
    replacement = f"{header}\n{indent}\"\"\"{doc}\"\"\"\n{indent}"
    if pattern not in text:
        raise RuntimeError(f"Pattern not found for {header}")
    return text.replace(pattern, replacement, 1)

text = Path("train.py").read_text(encoding="utf-8")
entries = [
    ("class MixupCutmix:", "Callable helper that applies mixup/cutmix based on configured probabilities.", "    "),
    ("    def __init__(self, mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0,", "Cache augmentation hyper-parameters so we can adjust at runtime.", "        "),
    ("    def set_prob(self, prob: float) -> None:", "Update the probability of applying mixup/cutmix for the current epoch.", "        "),
    ("    def _sample_lambda(self, alpha: float) -> float:", "Draw a lambda value from the Beta distribution (or 1.0 if disabled).", "        "),
    ("    def _apply_cutmix(self, x: torch.Tensor, indices: torch.Tensor, lam: float) -> tuple[torch.Tensor, float]:", "Perform CutMix by replacing a random patch and return the adjusted lambda.", "        "),
    ("    def __call__(self, x: torch.Tensor, y: torch.Tensor):", "Apply mixup or cutmix to a batch and return mixed targets when used.", "        "),
    ("def parse_args() -> argparse.Namespace:", "Set up CLI arguments for the training entry point.", "    "),
    ("def mixup_criterion(criterion: nn.Module, logits: torch.Tensor,", "Blend losses when mixup/cutmix produced two targets.", "    "),
    ("def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,", "Run a full training epoch including optional mixup and EMA tracking.", "    "),
    ("def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,", "Validate the model on the held-out split without gradient tracking.", "    "),
    ("def main() -> None:", "High-level training orchestration covering data prep, optimizers, and checkpoints.", "    "),
]
for header, doc, indent in entries:
    text = add_doc(text, header, doc, indent)
text = text.replace("        optimizer.zero_grad(set_to_none=True)\n", "        optimizer.zero_grad(set_to_none=True)  # Cheaper than setting grads to zero manually.\n")
text = text.replace("        scaler.scale(loss).backward()\n", "        scaler.scale(loss).backward()  # Scale gradients to keep FP16 numerically stable.\n")
text = text.replace("    with torch.no_grad():\n        for x, y in tqdm(loader, desc=\"val\", leave=True):\n", "    with torch.no_grad():\n        for x, y in tqdm(loader, desc=\"val\", leave=True):\n            # Validation never modifies weights, so gradients remain disabled.\n")
Path("train.py").write_text(text, encoding="utf-8")
