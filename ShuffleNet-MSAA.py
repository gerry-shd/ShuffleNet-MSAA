# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_activation_layer
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.utils import channel_shuffle, make_divisible
from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone
import torch.nn.functional as F



# MSAA
class AdaptiveAttentionModule(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[3, 5, 7], reduction_ratio=8):
        super(AdaptiveAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.reduction_ratio = reduction_ratio

        # Reducing the number of channels using 1x1 convolution improves computational efficiency
        self.reduce_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)

        # Multi-scale local feature extraction convolutional layer
        self.local_convs = nn.ModuleList([
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=k, padding=k // 2, groups=in_channels // 2,
                      bias=False)
            for k in kernel_sizes
        ])

        # Channel Attention Mechanisms
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels // 2, in_channels // (2 * reduction_ratio), kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // (2 * reduction_ratio), in_channels // 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Point Convolution Adaptive Weight Fusion
        self.pointwise_conv = nn.Conv2d(in_channels // 2 * len(kernel_sizes), in_channels // 2, kernel_size=1,
                                        bias=False)

        # The final 1x1 convolution is used to recover the channel
        self.expand_conv = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False)

        # activation function
        self.swish = nn.SiLU()  # May be replaced by GELU

    def spatial_adaptive_attention(self, feature, kernel_size):
        """Multi-scale spatial adaptive attention mechanisms"""
        batch_size, channels, height, width = feature.size()
        pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False).to(feature.device)
        local_min = -pool(-feature)
        local_max = pool(feature)
        local_median = torch.median(feature.view(batch_size, channels, -1), dim=2).values.view(batch_size, channels, 1,1).to(feature.device)
        # Adaptive weight calculation
        spatial_weight = (local_median > local_min).float() * (local_median < local_max).float()
        return spatial_weight

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Reduce the number of channels using 1x1 convolution
        x_reduced = self.reduce_conv(x)

        # channel attention
        channel_weights = self.channel_attention(x_reduced)
        x_reduced = x_reduced * channel_weights

        # Applying convolutional and spatially adaptive attention mechanisms to multi-scale feature maps respectively
        multi_scale_features = []
        for i, conv in enumerate(self.local_convs):
            feature = conv(x_reduced)

            # Applying spatial adaptive attention mechanisms
            spatial_weight = self.spatial_adaptive_attention(feature, self.kernel_sizes[i])
            feature = feature * spatial_weight

            multi_scale_features.append(feature)

        # Stitching multi-scale features together and fusing them using point convolution
        combined_features = torch.cat(multi_scale_features, dim=1)
        out = self.pointwise_conv(combined_features)

        # Enhancing nonlinearity and recovering channel counts using activation functions
        out = self.swish(out)
        out = self.expand_conv(out)

        # Adding a residual connection
        out += x

        return out





class MultiScaleFeatureExtractor(nn.Module):
    """Multi-Scale Feature Extraction Module with various kernel sizes."""

    def __init__(self, in_channels, out_channels, kernel_sizes=(1, 3, 5)):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.convs = nn.ModuleList([
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channels // len(kernel_sizes),
                kernel_size=k,
                stride=1,
                padding=k // 2,
                act_cfg=None,
                norm_cfg=dict(type='BN'))
            for k in kernel_sizes
        ])

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        return torch.cat(features, dim=1)


class ShuffleUnit(BaseModule):
    """ShuffleUnit block with multi-scale feature extraction and attention."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=3,
                 first_block=True,
                 combine='add',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        super(ShuffleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_block = first_block
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        self.with_cp = with_cp

        if self.combine == 'add':
            self.depthwise_stride = 1
            self._combine_func = self._add
            assert in_channels == out_channels, (
                'in_channels must be equal to out_channels when combine '
                'is add')
        elif self.combine == 'concat':
            self.depthwise_stride = 2
            self._combine_func = self._concat
            self.out_channels -= self.in_channels
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f'Cannot combine tensors with {self.combine}. '
                             'Only "add" and "concat" are supported')

        self.first_1x1_groups = 1 if first_block else self.groups
        self.g_conv_1x1_compress = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            groups=self.first_1x1_groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.multi_scale_conv = MultiScaleFeatureExtractor(
            in_channels=self.bottleneck_channels,
            out_channels=self.bottleneck_channels)

        self.depthwise_conv3x3_bn = ConvModule(
            in_channels=self.bottleneck_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=3,
            stride=self.depthwise_stride,
            padding=1,
            groups=self.bottleneck_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.g_conv_1x1_expand = ConvModule(
            in_channels=self.bottleneck_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            groups=self.groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.attention = AdaptiveAttentionModule(self.out_channels)
        self.act = build_activation_layer(act_cfg)

    @staticmethod
    def _add(x, out):
        return x + out

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, x):

        def _inner_forward(x):
            residual = x

            out = self.g_conv_1x1_compress(x)
            out = self.multi_scale_conv(out)  # Multi-scale feature extraction
            out = self.depthwise_conv3x3_bn(out)

            if self.groups > 1:
                out = channel_shuffle(out, self.groups)

            out = self.g_conv_1x1_expand(out)
            out = self.attention(out)  # Apply adaptive attention

            if self.combine == 'concat':
                residual = self.avgpool(residual)
                out = self.act(out)
                out = self._combine_func(residual, out)
            else:
                out = self._combine_func(residual, out)
                out = self.act(out)
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


@MODELS.register_module()
class ShuffleNetV1(BaseBackbone):
    """ShuffleNetV1 backbone with optimized ShuffleUnits."""

    def __init__(self,
                 groups=3,
                 widen_factor=1.0,
                 out_indices=(2,),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=None):
        super(ShuffleNetV1, self).__init__(init_cfg)
        self.init_cfg = init_cfg
        self.stage_blocks = [4, 8, 4]
        self.groups = groups

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        if groups == 1:
            channels = (144, 288, 576)
        elif groups == 2:
            channels = (200, 400, 800)
        elif groups == 3:
            channels = (240, 480, 960)
        elif groups == 4:
            channels = (272, 544, 1088)
        elif groups == 8:
            channels = (384, 768, 1536)
        else:
            raise ValueError(f'{groups} groups is not supported for 1x1 '
                             'Grouped Convolutions')

        channels = [make_divisible(ch * widen_factor, 8) for ch in channels]
        self.in_channels = int(24 * widen_factor)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            first_block = True if i == 0 else False
            layer = self.make_layer(channels[i], num_blocks, first_block)
            self.layers.append(layer)

    def make_layer(self, out_channels, num_blocks, first_block=False):
        layers = []
        for i in range(num_blocks):
            first_block = first_block if i == 0 else False
            combine_mode = 'concat' if i == 0 else 'add'
            layers.append(
                ShuffleUnit(
                    self.in_channels,
                    out_channels,
                    groups=self.groups,
                    first_block=first_block,
                    combine=combine_mode,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        super(ShuffleNetV1, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
