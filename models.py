from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Re-arrange channel groups to enable information mixing between branches."""
    b, c, h, w = x.size()
    if c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ConvBNAct(nn.Module):
    """Convenience block that applies Conv -> BatchNorm -> Activation."""
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: Optional[int] = None,
                 groups: int = 1, act: bool = True):
        """Build the convolutional stack with optional grouped convs and activation."""
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        """Apply the conv-bn-act sequence to the input tensor."""
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """SE-style attention module that re-weights channels via global pooling."""
    def __init__(self, ch: int, r: int = 8):
        """Create the bottleneck fully connected layers used to score channels."""
        super().__init__()
        mid = max(8, ch // r)
        self.fc1 = nn.Conv2d(ch, mid, 1, bias=True)
        self.fc2 = nn.Conv2d(mid, ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention-weighted activations."""
        g = F.adaptive_avg_pool2d(x, 1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(g), inplace=True)))
        return x * w


class SpatialMultiOrderAdaptiveWeight(nn.Module):
    """Approximate pixel-wise attention using min/median/max filters at multiple scales."""
    def __init__(self):
        super().__init__()

    @staticmethod
    def _min_max(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return local sliding-window minima and maxima for kernel size k."""
        x_min = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)
        x_max = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        return x_min, x_max

    @staticmethod
    def _median_approx(x: torch.Tensor, k: int) -> torch.Tensor:
        """Approximate median filtering via average pooling in the neighborhood."""
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse scale-specific masks into a spatial gating tensor."""
        w_sum = 0.0
        for k in (3, 5, 7):
            x_min, x_max = self._min_max(x, k)
            x_med = self._median_approx(x, k)
            cond = (x_med > x_min).to(x.dtype) * (x_med < x_max).to(x.dtype)
            w_sum = w_sum + cond
        w_norm = torch.sigmoid(w_sum - 1.5)
        return x * w_norm


class MSAA(nn.Module):
    """Multi-Scale Adaptive Attention block combining channel and spatial cues."""
    def __init__(self, ch: int, compress_ratio: int = 2, r: int = 8):
        """Create parallel depthwise branches and attention heads."""
        super().__init__()
        c_mid = max(8, ch // compress_ratio)
        self.compress = ConvBNAct(ch, c_mid, k=1, s=1, p=0)
        self.dw3 = ConvBNAct(c_mid, c_mid, k=3, s=1, groups=c_mid)
        self.dw5 = ConvBNAct(c_mid, c_mid, k=5, s=1, groups=c_mid)
        self.dw7 = ConvBNAct(c_mid, c_mid, k=7, s=1, groups=c_mid)
        self.fuse = ConvBNAct(c_mid * 3, ch, k=1, s=1, p=0, act=True)
        self.ca = ChannelAttention(ch, r=r)
        self.spatial = SpatialMultiOrderAdaptiveWeight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return residual multi-scale features."""
        y = self.compress(x)
        f = torch.cat([self.dw3(y), self.dw5(y), self.dw7(y)], dim=1)
        f = self.fuse(f)
        f = self.ca(f)
        f = self.spatial(f)
        return f + x


class MultiScaleExtractor(nn.Module):
    """Simplified multi-kernel depthwise extractor without attention."""
    def __init__(self, ch: int, compress_ratio: int = 2):
        """Create three depthwise branches whose outputs are fused residually."""
        super().__init__()
        c_mid = max(8, ch // compress_ratio)
        self.compress = ConvBNAct(ch, c_mid, k=1, s=1, p=0)
        self.dw3 = ConvBNAct(c_mid, c_mid, k=3, s=1, groups=c_mid)
        self.dw5 = ConvBNAct(c_mid, c_mid, k=5, s=1, groups=c_mid)
        self.dw7 = ConvBNAct(c_mid, c_mid, k=7, s=1, groups=c_mid)
        self.fuse = ConvBNAct(c_mid * 3, ch, k=1, s=1, p=0, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return residual attention-enhanced features."""
        y = self.compress(x)
        f = torch.cat([self.dw3(y), self.dw5(y), self.dw7(y)], dim=1)
        f = self.fuse(f)
        return f + x




class StochasticDepth(nn.Module):
    """Implements per-sample drop-path regularization."""
    def __init__(self, drop_prob: float = 0.0):
        """Store the drop probability, clamped to [0, 1]."""
        super().__init__()
        self.drop_prob = float(max(0.0, min(1.0, drop_prob)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop entire residual paths during training."""
        if (not self.training) or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = random_tensor.floor()
        return x.div(keep_prob) * binary_mask




class MultiLevelAggregationHead(nn.Module):
    """Project feature maps from multiple stages into a shared embedding."""
    def __init__(self, in_channels, embed_dim: int, dropout: float = 0.2):
        """Instantiate per-stage projections and the final token mixer."""
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.SiLU(inplace=True),
            ) for ch in in_channels
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim * len(in_channels)),
            nn.Linear(embed_dim * len(in_channels), embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, feats):
        """Pool every feature map and concatenate tokens for classification."""
        tokens = []
        for f, proj in zip(feats, self.proj):
            pooled = F.adaptive_avg_pool2d(proj(f), 1).flatten(1)
            tokens.append(self.dropout(pooled))
        x = torch.cat(tokens, dim=1)
        return self.fc(x)


class ShuffleNetMSAABlock(nn.Module):
    """ShuffleNet block augmented with multi-scale extraction and MSAA attention."""
    def __init__(self, in_ch: int, out_ch: int, stride: int, groups: int = 2, drop_prob: float = 0.0):
        """Configure the twin branches used for stride-1 and stride-2 cases."""
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        self.stride = stride
        self.groups = groups

        if stride == 1:
            assert in_ch == out_ch
            c = out_ch // 2
            self.pw1 = ConvBNAct(c, c, k=1, s=1, p=0, groups=groups)
            self.ms = MultiScaleExtractor(c)
            self.dw = ConvBNAct(c, c, k=3, s=1, groups=c, act=False)
            self.pw2 = ConvBNAct(c, c, k=1, s=1, p=0, groups=groups)
            self.attn = MSAA(c)
            self.drop = StochasticDepth(drop_prob)
        else:
            c = out_ch // 2
            self.drop = nn.Identity()
            self.branch1 = nn.Sequential(
                ConvBNAct(in_ch, in_ch, k=3, s=2, groups=in_ch, act=False),
                ConvBNAct(in_ch, c, k=1, s=1, p=0, groups=groups),
                MSAA(c),
            )
            self.branch2 = nn.Sequential(
                ConvBNAct(in_ch, c, k=1, s=1, p=0, groups=groups),
                MultiScaleExtractor(c),
                ConvBNAct(c, c, k=3, s=2, groups=c, act=False),
                ConvBNAct(c, c, k=1, s=1, p=0, groups=groups),
                MSAA(c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Switch between residual and downsampling behaviours depending on stride."""
        if self.stride == 1:
            x1, x2 = torch.chunk(x, 2, dim=1)
            out = self.pw1(x2)
            out = self.ms(out)
            out = channel_shuffle(out, self.groups)
            out = self.dw(out)
            out = self.pw2(out)
            out = self.attn(out)
            out = self.drop(out)
            return torch.cat([x1, out], dim=1)
        out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_shuffle(out, self.groups)


@dataclass
class ModelSpec:
    """Configuration object passed into build_model."""
    name: str
    num_classes: int
    width_mult: float = 1.0
    drop_path_rate: float = 0.0


class ShuffleNetMSAA(nn.Module):
    """Main classification network with MSAA-enhanced ShuffleNet stages."""
    def __init__(self, num_classes: int, width_mult: float = 1.0, drop_path_rate: float = 0.0):
        """Construct the backbone, aggregation head, and classifier."""
        super().__init__()
        def c(ch):
            return max(16, int(ch * width_mult))

        self.stem = nn.Sequential(
            ConvBNAct(3, c(24), k=3, s=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        stage_out = [c(116), c(232), c(464)]
        repeats = [4, 8, 4]
        total_blocks = sum(repeats)
        if drop_path_rate > 0:
            drop_rates = torch.linspace(0, drop_path_rate, steps=total_blocks).tolist()
        else:
            drop_rates = [0.0] * total_blocks

        in_ch = c(24)
        stages = []
        dp_idx = 0
        for out_ch, rep in zip(stage_out, repeats):
            blocks = [ShuffleNetMSAABlock(in_ch, out_ch, stride=2, drop_prob=drop_rates[dp_idx])]
            dp_idx += 1
            for _ in range(rep - 1):
                blocks.append(ShuffleNetMSAABlock(out_ch, out_ch, stride=1, drop_prob=drop_rates[dp_idx]))
                dp_idx += 1
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        head_ch = c(1024)
        self.head_proj = ConvBNAct(in_ch, head_ch, k=1, s=1, p=0)
        agg_in = stage_out + [head_ch]
        agg_dim = c(512)
        self.agg = MultiLevelAggregationHead(agg_in, embed_dim=agg_dim, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(agg_dim),
            nn.Dropout(0.2),
            nn.Linear(agg_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the input through the stem, stages, and classifier head."""
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        head_feat = self.head_proj(x)
        agg = self.agg(feats + [head_feat])
        return self.classifier(agg)


def build_model(spec: ModelSpec) -> nn.Module:
    """Factory that builds a model using the provided ModelSpec."""
    name = spec.name.lower()
    if name == "shufflenet_msaa":
        return ShuffleNetMSAA(num_classes=spec.num_classes, width_mult=spec.width_mult, drop_path_rate=spec.drop_path_rate)
    raise ValueError("Unknown model. Use --model shufflenet_msaa")
