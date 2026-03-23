"""
Detection Metrics: IoU and Mean Average Precision (mAP).

Implements:
  - compute_iou: pairwise Intersection-over-Union between box sets
  - MeanAveragePrecision: Pascal VOC-style mAP at IoU=0.5

Usage:
    metric = MeanAveragePrecision(num_classes=2, iou_threshold=0.5)
    metric.update(predictions, targets)   # per batch
    results = metric.compute()            # at epoch end
    metric.reset()
"""

import torch
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def compute_iou(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes in (x1, y1, x2, y2) format.

    Args:
        boxes_a: (N, 4) — first set of boxes
        boxes_b: (M, 4) — second set of boxes

    Returns:
        iou: (N, M) — pairwise IoU matrix
    """
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros(boxes_a.size(0), boxes_b.size(0))

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(0) * \
             (boxes_a[:, 3] - boxes_a[:, 1]).clamp(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(0) * \
             (boxes_b[:, 3] - boxes_b[:, 1]).clamp(0)

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union_area  = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / union_area.clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Mean Average Precision
# ---------------------------------------------------------------------------

class MeanAveragePrecision:
    """
    Pascal VOC mean Average Precision (mAP) metric.

    Computes per-class AP using the 11-point interpolation method and averages
    across all foreground classes.  Background (class 0) is always ignored.

    Args:
        num_classes:   Total number of classes including background.
        iou_threshold: IoU threshold for a prediction to count as TP.

    Usage:
        metric = MeanAveragePrecision(num_classes=2)

        # During validation loop:
        metric.update(predictions, targets)

        # After each epoch:
        results = metric.compute()
        print(results["mAP"])   # scalar
        metric.reset()
    """

    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
    ) -> None:
        self.num_classes   = num_classes
        self.iou_threshold = iou_threshold
        # Per-class accumulators: list of (score, is_tp) tuples
        self._detections: Dict[int, List[Tuple[float, int]]] = {
            c: [] for c in range(1, num_classes)
        }
        # Per-class ground-truth counts
        self._num_gts: Dict[int, int] = {c: 0 for c in range(1, num_classes)}

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def update(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets:     List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Accumulate predictions and ground-truths for a batch.

        Args:
            predictions: List of per-image dicts with keys:
                           'boxes'  (K, 4) pixel xyxy
                           'scores' (K,)
                           'labels' (K,)
            targets:     List of per-image dicts with keys:
                           'boxes'  (M, 4) normalized xyxy  ← scaled to pixels
                           'labels' (M,)
                         Note: targets from dataset.py are normalized; caller must
                         scale to pixels or pass pixel-level targets here.
        """
        for pred, tgt in zip(predictions, targets):
            pred_boxes  = pred["boxes"].cpu()    # (K, 4)
            pred_scores = pred["scores"].cpu()   # (K,)
            pred_labels = pred["labels"].cpu()   # (K,)
            gt_boxes    = tgt["boxes"].cpu()     # (M, 4)
            gt_labels   = tgt["labels"].cpu()    # (M,)

            # Count ground-truths per class
            for cls_id in range(1, self.num_classes):
                self._num_gts[cls_id] += int((gt_labels == cls_id).sum())

            if pred_boxes.numel() == 0:
                continue

            # For each class, evaluate detections
            for cls_id in range(1, self.num_classes):
                pred_mask = pred_labels == cls_id
                gt_mask   = gt_labels   == cls_id

                cls_pred_boxes  = pred_boxes[pred_mask]
                cls_pred_scores = pred_scores[pred_mask]
                cls_gt_boxes    = gt_boxes[gt_mask]

                if cls_pred_boxes.numel() == 0:
                    continue

                # Sort predictions by descending score
                sort_idx   = cls_pred_scores.argsort(descending=True)
                cls_pred_boxes  = cls_pred_boxes[sort_idx]
                cls_pred_scores = cls_pred_scores[sort_idx]

                matched_gt = torch.zeros(cls_gt_boxes.size(0), dtype=torch.bool)

                for k in range(cls_pred_boxes.size(0)):
                    score = float(cls_pred_scores[k])
                    if cls_gt_boxes.numel() == 0:
                        self._detections[cls_id].append((score, 0))
                        continue

                    iou = compute_iou(
                        cls_pred_boxes[k:k+1], cls_gt_boxes
                    ).squeeze(0)            # (M_cls,)
                    best_iou, best_j = iou.max(dim=0)

                    if best_iou >= self.iou_threshold and not matched_gt[best_j]:
                        matched_gt[best_j] = True
                        self._detections[cls_id].append((score, 1))  # TP
                    else:
                        self._detections[cls_id].append((score, 0))  # FP

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self) -> Dict[str, float]:
        """
        Compute mAP and per-class AP from accumulated detections.

        Returns:
            dict with:
              'mAP'          — mean average precision across foreground classes
              'AP_class_{c}' — per-class AP  (c = 1 … num_classes-1)
        """
        results: Dict[str, float] = {}
        aps: List[float] = []

        for cls_id in range(1, self.num_classes):
            dets     = self._detections[cls_id]
            num_gt   = self._num_gts[cls_id]

            if num_gt == 0:
                results[f"AP_class_{cls_id}"] = 0.0
                continue

            if len(dets) == 0:
                results[f"AP_class_{cls_id}"] = 0.0
                aps.append(0.0)
                continue

            # Sort by descending score
            dets_sorted = sorted(dets, key=lambda x: -x[0])
            tp_cumsum = torch.cumsum(
                torch.tensor([d[1] for d in dets_sorted], dtype=torch.float32), dim=0
            )
            fp_cumsum = torch.cumsum(
                torch.tensor([1 - d[1] for d in dets_sorted], dtype=torch.float32), dim=0
            )

            precision = tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-6)
            recall    = tp_cumsum / num_gt

            # 11-point interpolated AP (VOC 2007)
            ap = self._voc_11point_ap(recall, precision)
            results[f"AP_class_{cls_id}"] = ap
            aps.append(ap)

        results["mAP"] = float(torch.tensor(aps).mean()) if aps else 0.0
        return results

    @staticmethod
    def _voc_11point_ap(
        recall: torch.Tensor,
        precision: torch.Tensor,
    ) -> float:
        """Compute 11-point interpolated Average Precision (VOC 2007)."""
        ap = 0.0
        for thr in torch.linspace(0, 1, 11):
            mask = recall >= thr
            if mask.any():
                ap += float(precision[mask].max())
        return ap / 11.0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated detections and GT counts."""
        self._detections = {c: [] for c in range(1, self.num_classes)}
        self._num_gts    = {c: 0  for c in range(1, self.num_classes)}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MeanAveragePrecision("
            f"num_classes={self.num_classes}, "
            f"iou_threshold={self.iou_threshold})"
        )
