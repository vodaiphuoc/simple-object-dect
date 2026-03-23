"""
SSD Anchor (Default Box) Generator.

Generates default boxes for each feature map scale following the SSD paper:
  "SSD: Single Shot MultiBox Detector"
  Liu et al., 2015 (https://arxiv.org/abs/1512.02325)

The generator produces anchors in (cx, cy, w, h) normalized format
(values in [0, 1] relative to the input image size).
"""

import math
import torch
from typing import List, Tuple


class AnchorGenerator(torch.nn.Module):
    """
    Generates SSD default boxes (anchors) for a fixed set of feature maps.

    For each feature map cell we generate k anchors, where k = len(aspect_ratios)
    + (1 if add_extra_scale else 0), i.e. one anchor per aspect ratio and one
    additional square anchor at the geometric mean scale.

    Args:
        image_size:    Input image side length (assumes square input).
        feature_maps:  List of (height, width) for each feature map level.
        min_scale:     Smallest anchor scale as a fraction of image_size.
        max_scale:     Largest anchor scale as a fraction of image_size.
        aspect_ratios: Aspect ratios w/h for each level (shared across levels).
        add_extra_scale: Whether to add a square anchor at sqrt(s_k * s_{k+1}).
    """

    def __init__(
        self,
        image_size: int = 300,
        feature_maps: List[Tuple[int, int]] = None,
        min_scale: float = 0.2,
        max_scale: float = 0.95,
        aspect_ratios: List[float] = None,
        add_extra_scale: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size

        if feature_maps is None:
            # Default: backbone (19×19, 10×10) + 4 extra SSD layers
            feature_maps = [
                (19, 19),
                (10, 10),
                (5,  5),
                (3,  3),
                (2,  2),
                (1,  1),
            ]
        self.feature_maps = feature_maps

        if aspect_ratios is None:
            aspect_ratios = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]
        self.aspect_ratios = aspect_ratios
        self.add_extra_scale = add_extra_scale

        num_levels = len(feature_maps)
        # Linearly interpolate scales from min_scale to max_scale
        self.scales = [
            min_scale + (max_scale - min_scale) * k / (num_levels - 1)
            for k in range(num_levels)
        ]
        # Append one extra scale for the geometric-mean anchor of the last level
        self.scales.append(1.0)

        # Pre-compute and register anchors as a buffer so they move with the model
        anchors = self._generate_anchors()
        self.register_buffer("anchors", anchors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_anchors(self) -> torch.Tensor:
        """
        Build all default boxes across all feature maps.

        Returns:
            Tensor of shape (N, 4) in (cx, cy, w, h) normalized format.
        """
        all_anchors: List[torch.Tensor] = []

        for level_idx, (fh, fw) in enumerate(self.feature_maps):
            scale_k = self.scales[level_idx]
            scale_k1 = self.scales[level_idx + 1]

            for i in range(fh):
                for j in range(fw):
                    cx = (j + 0.5) / fw
                    cy = (i + 0.5) / fh

                    for ar in self.aspect_ratios:
                        w = scale_k * math.sqrt(ar)
                        h = scale_k / math.sqrt(ar)
                        all_anchors.append([cx, cy, w, h])

                    if self.add_extra_scale:
                        # Extra square anchor at geometric mean scale
                        scale_extra = math.sqrt(scale_k * scale_k1)
                        all_anchors.append([cx, cy, scale_extra, scale_extra])

        anchors = torch.tensor(all_anchors, dtype=torch.float32)
        # Clamp to [0, 1]
        anchors.clamp_(0.0, 1.0)
        return anchors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, dummy: torch.Tensor = None) -> torch.Tensor:  # type: ignore[override]
        """Return the pre-computed anchors buffer."""
        return self.anchors

    @property
    def num_anchors_per_location(self) -> int:
        """Number of anchors generated per spatial location."""
        n = len(self.aspect_ratios)
        if self.add_extra_scale:
            n += 1
        return n

    @property
    def total_anchors(self) -> int:
        """Total number of anchors across all feature maps."""
        return self.anchors.shape[0]


if __name__ == "__main__":
    gen = AnchorGenerator()
    print(f"Total anchors: {gen.total_anchors}")
    print(f"Anchors per location: {gen.num_anchors_per_location}")
    print(f"Anchors shape: {gen.anchors.shape}")
    print(f"Sample anchor: {gen.anchors[0]}")
