"""
Full MobileNetV2-SSD Object Detection Model.

Combines:
  - MobileNetV2 backbone (mobilenetv2.py)
  - SSD multi-scale prediction head (ssd_head.py)
  - Default box / anchor generator (anchors.py)

Usage:
    model = MobileNetV2SSD(num_classes=2)  # background + car
    loc, cls, anchors = model(images)       # training mode
    detections = model.predict(images)      # inference with NMS
"""

import torch
import torch.nn as nn
import torchvision.ops as ops   # only ops (nms/boxes), not model imports
from typing import List, Dict, Tuple, Optional

from model.backbone import MobileNetV2
from model.ssd_head import SSDHead
from anchors import AnchorGenerator
from losses import decode_boxes, cxcywh_to_xyxy


class MobileNetV2SSD(nn.Module):
    """
    MobileNetV2-SSD single-shot object detector.

    Architecture:
      Input (300×300) → MobileNetV2 backbone
                      → SSD head (6 prediction scales)
                      → (loc_preds, cls_preds, anchors)

    Args:
        num_classes:         Total number of classes **including background** (idx 0).
        width_mult:          MobileNetV2 width multiplier.
        image_size:          Input image side length (square assumed).
        score_threshold:     Minimum confidence score for inference detections.
        nms_iou_threshold:   IoU threshold for non-maximum suppression.
        max_detections:      Maximum number of detections to return per image.
        variances:           Encoding variances matching those used in MultiBoxLoss.
    """

    def __init__(
        self,
        num_classes: int = 2,
        width_mult: float = 1.0,
        image_size: int = 300,
        score_threshold: float = 0.3,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 100,
        variances: Tuple[float, float] = (0.1, 0.2),
    ) -> None:
        super().__init__()
        self.num_classes       = num_classes
        self.image_size        = image_size
        self.score_threshold   = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections    = max_detections
        self.variances         = variances

        # --- Backbone ---
        self.backbone = MobileNetV2(width_mult=width_mult)

        # Determine backbone output channels for the given width_mult
        def _make_divisible(v: float, divisor: int = 8) -> int:
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        ch_s16 = _make_divisible(96  * width_mult)  # stride-16 feature channels
        ch_s32 = _make_divisible(320 * width_mult)  # stride-32 feature channels

        # --- Anchor generator ---
        self.anchor_generator = AnchorGenerator(
            image_size=image_size,
            feature_maps=[(19,19),(10,10),(5,5),(3,3),(2,2),(1,1)],
            min_scale=0.2,
            max_scale=0.95,
            aspect_ratios=[1.0, 2.0, 0.5, 3.0, 1.0/3.0],
            add_extra_scale=True,
        )
        num_anchors_per_loc = self.anchor_generator.num_anchors_per_location

        # --- SSD head ---
        self.ssd_head = SSDHead(
            num_classes=num_classes,
            num_anchors_per_loc=num_anchors_per_loc,
            in_channels_s16=ch_s16,
            in_channels_s32=ch_s32,
        )

        # Weight initialization
        self._init_head_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_head_weights(self) -> None:
        """Xavier-uniform init for SSD head conv layers, zero bias."""
        for m in self.ssd_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass (training mode).

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            loc_preds: (B, N_anchors, 4)            — raw location offsets
            cls_preds: (B, N_anchors, num_classes)  — raw class logits
            anchors:   (N_anchors, 4)               — default boxes (cx,cy,w,h)
        """
        feat_s16, feat_s32 = self.backbone(x)
        loc_preds, cls_preds = self.ssd_head(feat_s16, feat_s32)
        anchors = self.anchor_generator()
        return loc_preds, cls_preds, anchors

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        score_threshold: Optional[float] = None,
        nms_iou_threshold: Optional[float] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference and return per-image detections after NMS.

        Args:
            x:                  Input image tensor (B, 3, H, W).
            score_threshold:    Override instance score_threshold.
            nms_iou_threshold:  Override instance nms_iou_threshold.

        Returns:
            List of dicts (one per image), each with:
              'boxes'  (K, 4)  — absolute pixel boxes in (x1, y1, x2, y2)
              'scores' (K,)    — confidence scores
              'labels' (K,)    — predicted class indices (1-indexed)
        """
        score_thr = score_threshold   or self.score_threshold
        nms_thr   = nms_iou_threshold or self.nms_iou_threshold

        self.eval()
        loc_preds, cls_preds, anchors = self.forward(x)

        B = x.size(0)
        H, W = x.shape[2], x.shape[3]
        results: List[Dict[str, torch.Tensor]] = []

        for i in range(B):
            # Decode box offsets → (N, 4) cx,cy,w,h normalized
            decoded_cxcywh = decode_boxes(loc_preds[i], anchors, self.variances)
            decoded_xyxy   = cxcywh_to_xyxy(decoded_cxcywh).clamp(0.0, 1.0)

            # Class scores (softmax, skip background at index 0)
            scores_all = torch.softmax(cls_preds[i], dim=-1)  # (N, C)
            scores, labels = scores_all[:, 1:].max(dim=-1)    # ignore background
            labels = labels + 1                                # 1-indexed

            # Score threshold filter
            keep = scores >= score_thr
            scores, labels = scores[keep], labels[keep]
            boxes_xyxy     = decoded_xyxy[keep]

            if boxes_xyxy.numel() == 0:
                results.append({
                    "boxes":  torch.zeros(0, 4),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                })
                continue

            # Scale to pixel coordinates
            scale = torch.tensor([W, H, W, H], dtype=torch.float32,
                                  device=boxes_xyxy.device)
            boxes_pixel = boxes_xyxy * scale

            # Per-class NMS
            keep_idx = ops.batched_nms(boxes_pixel, scores, labels, nms_thr)
            keep_idx = keep_idx[: self.max_detections]

            results.append({
                "boxes":  boxes_pixel[keep_idx],
                "scores": scores[keep_idx],
                "labels": labels[keep_idx],
            })

        return results


# if __name__ == "__main__":
#     model = MobileNetV2SSD(num_classes=2)
#     dummy = torch.randn(2, 3, 300, 300)
#     loc, cls, anchors = model(dummy)
#     print(f"loc_preds : {loc.shape}")      # (2, N, 4)
#     print(f"cls_preds : {cls.shape}")      # (2, N, 2)
#     print(f"anchors   : {anchors.shape}")  # (N, 4)
#     print(f"Total anchors: {anchors.shape[0]}")

#     detections = model.predict(dummy)
#     for i, det in enumerate(detections):
#         print(f"Image {i}: {det['boxes'].shape[0]} detections")
