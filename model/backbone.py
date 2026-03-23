"""
MobileNetV2 backbone implemented from scratch in PyTorch.

Architecture follows the original paper:
  "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
  Sandler et al., 2018 (https://arxiv.org/abs/1801.04381)

No torchvision model imports are used.
The backbone exposes two intermediate feature maps at stride 16 and stride 32,
which are fed into the SSD detection head.
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class ConvBNReLU6(nn.Module):
    """
    Convolution → Batch Normalization → ReLU6 block.

    Optionally uses depth-wise (groups=in_channels) convolution to
    implement the depth-wise separable convolution used in inverted residuals.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        use_activation: bool = True,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if use_activation:
            layers.append(nn.ReLU6(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    """
    MobileNetV2 Inverted Residual (bottleneck) block.

    The block expands the input channels by `expand_ratio`, applies a
    depth-wise 3×3 convolution, then projects back to `out_channels` with
    a linear (no activation) 1×1 convolution.

    A residual (skip) connection is applied when stride == 1 and
    in_channels == out_channels.

    Args:
        in_channels:   Number of input channels.
        out_channels:  Number of output channels.
        stride:        Convolution stride (1 or 2).
        expand_ratio:  Channel expansion factor t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        assert stride in (1, 2), f"stride must be 1 or 2, got {stride}"
        self.use_skip = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        layers: List[nn.Module] = []
        # Expand phase (omitted when expand_ratio == 1)
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))
        # Depth-wise convolution
        layers.append(
            ConvBNReLU6(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim)
        )
        # Linear projection (no activation)
        layers.append(
            ConvBNReLU6(hidden_dim, out_channels, kernel_size=1, use_activation=False)
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_skip:
            out = out + x
        return out


class MobileNetV2(nn.Module):
    """
    MobileNetV2 backbone for SSD feature extraction.

    Returns two feature tensors used by the SSD head:
      - feat_s16: output of the 14th bottleneck stage (stride 16 relative to input)
      - feat_s32: output of the final bottleneck stage (stride 32 relative to input)

    For a 300×300 input the shapes are approximately:
      feat_s16: (B, 96,  19, 19)
      feat_s32: (B, 320, 10, 10)

    Architecture table (from the paper, Table 2):
      Input | Operator     | t | c   | n | s
      ------+--------------+---+-----+---+---
      3×300 | conv2d       | - | 32  | 1 | 2
      32    | bottleneck   | 1 | 16  | 1 | 1
      16    | bottleneck   | 6 | 24  | 2 | 2
      24    | bottleneck   | 6 | 32  | 3 | 2
      32    | bottleneck   | 6 | 64  | 4 | 2
      64    | bottleneck   | 6 | 96  | 3 | 1   <- feat_s16 extracted here
      96    | bottleneck   | 6 | 160 | 3 | 2
      160   | bottleneck   | 6 | 320 | 1 | 1   <- feat_s32 extracted here

    Args:
        width_mult: Width multiplier α ∈ (0, 1]. Scales all channel counts.
    """

    # (expand_ratio, out_channels, num_blocks, stride)
    _INVERTED_RESIDUAL_CONFIG = [
        (1, 16,  1, 1),  # block 1
        (6, 24,  2, 2),  # block 2
        (6, 32,  3, 2),  # block 3
        (6, 64,  4, 2),  # block 4
        (6, 96,  3, 1),  # block 5  → feat_s16
        (6, 160, 3, 2),  # block 6
        (6, 320, 1, 1),  # block 7  → feat_s32
    ]

    def __init__(self, width_mult: float = 1.0) -> None:
        super().__init__()

        def _make_divisible(v: float, divisor: int = 8) -> int:
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        input_channel = _make_divisible(32 * width_mult)

        # First conv layer: 3 → 32, stride 2
        self.first_conv = ConvBNReLU6(3, input_channel, stride=2)

        # Build inverted-residual stages
        # We keep track of which stage index to split at for multi-scale output.
        stages: List[nn.Module] = []
        for t, c, n, s in self._INVERTED_RESIDUAL_CONFIG:
            out_channel = _make_divisible(c * width_mult)
            blocks: List[nn.Module] = []
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(
                    InvertedResidual(input_channel, out_channel, stride=stride, expand_ratio=t)
                )
                input_channel = out_channel
            stages.append(nn.Sequential(*blocks))

        # Stages 0–4 (indices 0..4) → produce feat_s16
        self.stage_s16 = nn.Sequential(*stages[:5])  # up to & including block 5
        # Stages 5–6 (indices 5..6) → produce feat_s32
        self.stage_s32 = nn.Sequential(*stages[5:])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.first_conv(x)
        feat_s16 = self.stage_s16(x)
        feat_s32 = self.stage_s32(feat_s16)
        return feat_s16, feat_s32


if __name__ == "__main__":
    model = MobileNetV2()
    dummy = torch.randn(2, 3, 300, 300)
    f16, f32 = model(dummy)
    print(f"feat_s16: {f16.shape}")   # (2, 96, 19, 19)
    print(f"feat_s32: {f32.shape}")   # (2, 320, 10, 10)
