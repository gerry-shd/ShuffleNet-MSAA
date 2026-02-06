from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWFLoss(nn.Module):
    """Dual-Weighted Focal Loss that down-weights majority classes dynamically.
    """
    def __init__(self, class_counts, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """Pre-compute class-specific alpha weights from the training histogram.

        Args:
            class_counts: Number of samples per label in the training split.
            alpha: Global scaling factor applied to each alpha term.
            gamma: Controls the focusing effect on hard samples.
            reduction: Either 'mean', 'sum', or 'none'.
        """
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        n_total = counts.sum().clamp(min=1.0)
        alpha_cls = (n_total / counts.clamp(min=1.0)) * float(alpha)
        self.register_buffer("alpha_cls", alpha_cls)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the DW-FL loss for a batch of logits and labels.
        """
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)

        alpha_t = self.alpha_cls.gather(0, targets)
        gamma_t = self.gamma * torch.log(1.0 / p_t)

        loss = -alpha_t * torch.pow(1.0 - p_t, gamma_t) * logp_t

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Standard focal loss implementation used as a baseline.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        """Store focusing and class-balancing hyper-parameters.
        """
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss using the configured gamma/alpha values.
        """
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -(1.0 - p_t) ** self.gamma * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device).gather(0, targets)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
