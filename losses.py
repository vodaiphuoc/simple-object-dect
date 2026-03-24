"""
MultiBox Loss for SSD object detection.

Implements the combined localization + classification loss from:
  "SSD: Single Shot MultiBox Detector" — Liu et al., 2015

Key components:
  1. Anchor-to-ground-truth matching via IoU threshold
  2. Smooth L1 localization loss on positive anchors
  3. Softmax cross-entropy classification loss with Hard Negative Mining
     (keeps negative:positive ratio ≤ neg_pos_ratio:1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------------

def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes in (x1, y1, x2, y2) format.

    Args:
        boxes_a: (N, 4)
        boxes_b: (M, 4)

    Returns:
        iou: (N, M)
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / union_area.clamp(min=1e-6)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    return torch.cat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2,
    ], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) → (cx, cy, w, h)."""
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2],
    ], dim=-1)


def encode_boxes(
    gt_boxes_cxcywh: torch.Tensor,
    anchors_cxcywh: torch.Tensor,
    variances: Tuple[float, float] = (0.1, 0.2),
) -> torch.Tensor:
    """
    Encode ground-truth boxes relative to matched anchors (SSD convention).

    Δcx = (gt_cx - anc_cx) / (var[0] * anc_w)
    Δcy = (gt_cy - anc_cy) / (var[0] * anc_h)
    Δw  = log(gt_w / anc_w) / var[1]
    Δh  = log(gt_h / anc_h) / var[1]
    """
    eps = 1e-7
    delta_cxy = (gt_boxes_cxcywh[:, :2] - anchors_cxcywh[:, :2]) / (
        variances[0] * anchors_cxcywh[:, 2:]
    )
    delta_wh = torch.log(
        (gt_boxes_cxcywh[:, 2:] / anchors_cxcywh[:, 2:].clamp(min=eps)).clamp(min=eps)
    ) / variances[1]
    return torch.cat([delta_cxy, delta_wh], dim=-1)


def decode_boxes(
    loc_preds: torch.Tensor,
    anchors_cxcywh: torch.Tensor,
    variances: Tuple[float, float] = (0.1, 0.2),
) -> torch.Tensor:
    """
    Decode predicted offsets back to absolute (cx, cy, w, h) boxes.
    Inverse of encode_boxes.
    """
    pred_cxy = loc_preds[:, :2] * variances[0] * anchors_cxcywh[:, 2:] + anchors_cxcywh[:, :2]
    pred_wh  = torch.exp(loc_preds[:, 2:] * variances[1]) * anchors_cxcywh[:, 2:]
    return torch.cat([pred_cxy, pred_wh], dim=-1)


# ---------------------------------------------------------------------------
# Main loss class
# ---------------------------------------------------------------------------

class MultiBoxLoss(nn.Module):
    """
    SSD MultiBox Loss = Localization Loss + α × Classification Loss.

    Localization loss: Smooth L1 on positive-matched anchors.
    Classification loss: Cross-entropy with hard negative mining.

    Args:
        num_classes:    Total number of classes (including background at index 0).
        iou_threshold:  IoU threshold for positive anchor matching.
        neg_pos_ratio:  Maximum ratio of hard negatives to positives.
        alpha:          Weight for localization loss term.
        variances:      Encoding variances for (Δcxy, Δwh).
    """

    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
        neg_pos_ratio: int = 3,
        alpha: float = 1.0,
        variances: Tuple[float, float] = (0.1, 0.2),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.variances = variances

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _match_anchors(
        self,
        gt_boxes_xyxy: torch.Tensor,   # (M, 4) ground-truth boxes, normalized xyxy
        gt_labels: torch.Tensor,        # (M,)   ground-truth class labels (1-indexed)
        anchors_cxcywh: torch.Tensor,   # (N, 4) anchors
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match anchors to ground-truth boxes (bipartite + threshold strategy).

        Returns:
            loc_targets: (N, 4)  encoded offsets (zero for unmatched)
            cls_targets: (N,)    class index per anchor; 0 = background
        """
        N = anchors_cxcywh.size(0)
        anchors_xyxy = cxcywh_to_xyxy(anchors_cxcywh)

        if gt_boxes_xyxy.numel() == 0:
            loc_targets = torch.zeros(N, 4, device=anchors_cxcywh.device)
            cls_targets = torch.zeros(N, dtype=torch.long, device=anchors_cxcywh.device)
            return loc_targets, cls_targets

        iou = box_iou(anchors_xyxy, gt_boxes_xyxy)  # (N, M)

        # For each anchor: best GT match
        best_gt_iou, best_gt_idx = iou.max(dim=1)  # (N,)
        # For each GT: ensure at least one anchor is matched
        best_anc_idx = iou.argmax(dim=0)             # (M,)
        best_gt_idx[best_anc_idx] = torch.arange(gt_boxes_xyxy.size(0),
                                                   device=gt_boxes_xyxy.device)
        best_gt_iou[best_anc_idx] = 1.0

        # Assign labels
        cls_targets = torch.zeros(N, dtype=torch.long, device=anchors_cxcywh.device)
        pos_mask = best_gt_iou >= self.iou_threshold
        cls_targets[pos_mask] = gt_labels[best_gt_idx[pos_mask]]

        # Encode box targets
        matched_gt_cxcywh = xyxy_to_cxcywh(gt_boxes_xyxy[best_gt_idx])
        loc_targets = encode_boxes(matched_gt_cxcywh, anchors_cxcywh, self.variances)
        # Zero out loc targets for negatives (not used in loss, but cleaner)
        loc_targets[~pos_mask] = 0.0
        print('cls_targets:', cls_targets.max())
        return loc_targets, cls_targets

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        loc_preds: torch.Tensor,            # (B, N, 4)
        cls_preds: torch.Tensor,            # (B, N, num_classes)
        anchors: torch.Tensor,              # (N, 4)  cx,cy,w,h
        targets: List[Dict[str, torch.Tensor]],  # list of {'boxes': (M,4), 'labels': (M,)}
    ) -> torch.Tensor:
        """
        Compute the total SSD multi-box loss.

        Args:
            loc_preds: Predicted localization offsets (B, N, 4).
            cls_preds: Predicted class logits (B, N, num_classes).
            anchors:   Default box anchors in (cx, cy, w, h) format (N, 4).
            targets:   Per-image ground-truth dicts with keys:
                         'boxes'  — (M, 4) normalized xyxy boxes
                         'labels' — (M,)  integer class labels, 1-indexed

        Returns:
            Scalar loss tensor.
        """
        B, N, _ = loc_preds.shape
        device = loc_preds.device

        loc_target_batch = torch.zeros(B, N, 4, device=device)
        cls_target_batch = torch.zeros(B, N, dtype=torch.long, device=device)

        for i, target in enumerate(targets):
            gt_boxes  = target["boxes"].to(device)   # (M, 4)
            gt_labels = target["labels"].to(device)  # (M,)
            loc_t, cls_t = self._match_anchors(gt_boxes, gt_labels, anchors)
            loc_target_batch[i] = loc_t
            cls_target_batch[i] = cls_t

        pos_mask = cls_target_batch > 0  # (B, N)
        num_pos  = pos_mask.sum().clamp(min=1)

        # ---- Localization loss (Smooth L1, positives only) ----
        loc_loss = F.smooth_l1_loss(
            loc_preds[pos_mask],
            loc_target_batch[pos_mask],
            reduction="sum",
        )

        # ---- Classification loss with Hard Negative Mining ----
        cls_preds_flat = cls_preds.view(B * N, self.num_classes)

        print(cls_preds_flat, cls_target_batch.view(-1))
        print(cls_preds_flat.max())
        print(cls_target_batch.view(-1).max())
        cls_loss_all   = F.cross_entropy(
            cls_preds_flat, cls_target_batch.view(-1), reduction="none"
        ).view(B, N)

        # Mask out positives for mining; mine among negatives
        cls_loss_neg = cls_loss_all.clone()
        cls_loss_neg[pos_mask] = 0.0

        # Sort negatives by descending loss and keep top neg_pos_ratio * num_pos
        num_neg = (self.neg_pos_ratio * pos_mask.sum(dim=1)).clamp(max=N - 1)
        sorted_idx = cls_loss_neg.argsort(dim=1, descending=True)
        rank       = sorted_idx.argsort(dim=1)
        neg_mask   = rank < num_neg.unsqueeze(1)

        cls_loss = (
            cls_loss_all[pos_mask].sum() + cls_loss_all[neg_mask].sum()
        )

        total_loss = (self.alpha * loc_loss + cls_loss) / num_pos
        return total_loss
