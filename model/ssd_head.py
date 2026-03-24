"""
SSD Detection Head.

Implements extra convolutional layers and per-scale prediction heads
as described in:
  "SSD: Single Shot MultiBox Detector"
  Liu et al., 2015 (https://arxiv.org/abs/1512.02325)

The head takes two backbone feature maps (stride 16, stride 32) and
appends four extra feature maps produced by strided convolutions,
giving a total of six prediction scales.
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class ExtraConvBlock(nn.Module):
    """
    Extra convolutional block that halves spatial dimensions.

    Architecture: 1×1 conv (channel compression) → 3×3 conv, stride 2
    (same pattern used in SSD VGG extras, adapted for MobileNetV2 channels).

    Args:
        in_channels:  Number of input channels.
        mid_channels: Bottleneck channel count for the 1×1 conv.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels, mid_channels, out_channels,  kernel_size = 3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PredictionHead(nn.Module):
    """
    Per-scale localization and classification prediction head.

    Produces two parallel 1×1 convolution branches:
      - loc_head:  predicts 4 * num_anchors offsets  (Δcx, Δcy, Δw, Δh)
      - cls_head:  predicts num_classes * num_anchors class scores

    Args:
        in_channels:  Feature map channels at this scale.
        num_anchors:  Number of default boxes per spatial location.
        num_classes:  Total number of classes (including background).
    """

    def __init__(
        self, in_channels: int, num_anchors: int, num_classes: int
    ) -> None:
        super().__init__()
        self.loc_head = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1
        )
        self.cls_head = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=1
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loc = self.loc_head(x)  # (B, num_anchors*4, H, W)
        cls = self.cls_head(x)  # (B, num_anchors*num_classes, H, W)
        return loc, cls


class SSDHead(nn.Module):
    """
    SSD multi-scale prediction head for MobileNetV2.

    Takes two backbone feature maps and produces predictions across six
    feature map scales.  For a 300×300 input the spatial sizes are:
      Scale 0: 19×19  (from backbone, stride 16, channels=96)
      Scale 1: 10×10  (from backbone, stride 32, channels=320)
      Scale 2:  5×5   (extra conv)
      Scale 3:  3×3   (extra conv)
      Scale 4:  2×2   (extra conv)
      Scale 5:  1×1   (extra conv)

    Args:
        num_classes:         Total number of classes (including background).
        num_anchors_per_loc: Number of anchors per spatial location.
        in_channels_s16:     Channels of the stride-16 backbone feature map.
        in_channels_s32:     Channels of the stride-32 backbone feature map.
    """

    # Extra layer configurations:
    # (in_ch, mid_ch, out_ch, kernel_size, stride, padding)
    _EXTRA_CONFIG = [
        (320, 256, 512, 3, 2, 1), # Scale 2: 10x10 -> 5x5
        (512, 128, 256, 3, 2, 1), # Scale 3: 5x5  -> 3x3
        (256, 128, 256, 3, 2, 1), # Scale 4: 3x3  -> 2x2
        (256, 64,  128, 2, 1, 0), # Scale 5: 2x2  -> 1x1 (K=2, S=1, P=0)
    ]

    # Channels at each of the 6 prediction scales
    _SCALE_CHANNELS = [96, 320, 512, 256, 256, 128]

    def __init__(
        self,
        num_classes: int,
        num_anchors_per_loc: int = 6,
        in_channels_s16: int = 96,
        in_channels_s32: int = 320,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_loc = num_anchors_per_loc

        # Extra convolutional layers (4 stages, each halves spatial size)
        self.extra_layers = nn.ModuleList([
            ExtraConvBlock(in_ch, mid_ch, out_ch,  kernel_size, stride, padding)
            for in_ch, mid_ch, out_ch, kernel_size, stride, padding 
            in self._EXTRA_CONFIG
        ])

        # Per-scale prediction heads
        scale_channels = [in_channels_s16, in_channels_s32] + [
            cfg[2] for cfg in self._EXTRA_CONFIG
        ]
        self.heads = nn.ModuleList([
            PredictionHead(ch, num_anchors_per_loc, num_classes)
            for ch in scale_channels
        ])

    def forward(
        self,
        feat_s16: torch.Tensor,
        feat_s32: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_s16: (B, C_s16, 19, 19) backbone stride-16 feature map
            feat_s32: (B, C_s32, 10, 10) backbone stride-32 feature map

        Returns:
            loc_preds: (B, N_anchors, 4)  — raw location offsets
            cls_preds: (B, N_anchors, num_classes) — raw class logits
        """
        feature_maps: List[torch.Tensor] = [feat_s16, feat_s32]

        # Build extra feature maps
        x = feat_s32
        for extra in self.extra_layers:
            x = extra(x)
            feature_maps.append(x)

        loc_list: List[torch.Tensor] = []
        cls_list: List[torch.Tensor] = []

        for feat, head in zip(feature_maps, self.heads):
            loc_raw, cls_raw = head(feat)  # (B, k*4, H, W), (B, k*C, H, W)
            B = loc_raw.size(0)
            # Reshape to (B, H*W*k, 4) and (B, H*W*k, num_classes)
            loc_list.append(
                loc_raw.permute(0, 2, 3, 1).contiguous()
                       .view(B, -1, 4)
            )
            cls_list.append(
                cls_raw.permute(0, 2, 3, 1).contiguous()
                       .view(B, -1, self.num_classes)
            )

        loc_preds = torch.cat(loc_list, dim=1)  # (B, N, 4)
        cls_preds = torch.cat(cls_list, dim=1)  # (B, N, num_classes)
        return loc_preds, cls_preds
