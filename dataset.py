"""
Expected directory layout (Kaggle):
    <root>/
        images/
            train/   *.jpg / *.png / ...
            val/     *.jpg / *.png / ...
        labels/
            train/   *.txt   (YOLO format, one line per box)
            val/     *.txt

YOLO annotation format (each line in a .txt file):
    <class_id> <x_center> <y_center> <width> <height>
    All values are normalized to [0, 1] relative to image size.
    Example: "1 0.700807 0.851667 0.166719 0.30037"
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not found. Install with: pip install albumentations")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 300) -> "A.Compose":
    """
    Augmentation pipeline for training.

    Includes:
      - Random horizontal flip
      - Color jitter (brightness, contrast, saturation)
      - Random scale + crop with bbox clipping
      - Resize to fixed image_size × image_size
      - Normalize with ImageNet mean/std
      - Convert to torch Tensor (C, H, W)
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.3),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",      # (x1,y1,x2,y2) absolute pixel coords
            min_visibility=0.2,
            label_fields=["labels"],
        ),
    )


def get_val_transforms(image_size: int = 300) -> "A.Compose":
    """Minimal validation transforms: resize + normalize."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_visibility=0.2,
            label_fields=["labels"],
        ),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class YOLODetectionDataset(Dataset):
    """
    Object-detection dataset that reads images and YOLO-format label files
    from a Kaggle-style directory structure.

    Directory layout expected:
        <root>/images/<split>/  → image files
        <root>/labels/<split>/  → .txt annotation files (YOLO format)

    YOLO format per line:
        <class_id> <x_center> <y_center> <width> <height>   (all normalized 0-1)

    The dataset returns dicts with:
        "image"  : (3, H, W) float32 tensor, normalized (ImageNet stats)
        "boxes"  : (M, 4)    float32 tensor, normalized xyxy [0..1]
        "labels" : (M,)      int64 tensor,   class IDs from the annotation files
                             (NOTE: labels are 0-indexed as stored in YOLO files;
                              if your pipeline expects 1-indexed, pass
                              label_offset=1 to shift all labels by +1)

    Args:
        root:         Path to the dataset root directory on disk.
        split:        "train" or "val".
        transforms:   albumentations Compose pipeline.
        image_size:   Target image size (default 300 for SSD).
        label_offset: Integer added to every raw class id (default 0).
                      Set to 1 if background is class 0 and your YOLO files
                      use 0-indexed foreground classes.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        image_size: int = 300,
        label_offset: int = 0,
    ) -> None:
        self.root         = Path(root)
        self.split        = split
        self.transforms   = transforms
        self.image_size   = image_size
        self.label_offset = label_offset

        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        # Collect all image paths that also have a corresponding label file
        self.samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            label_path = self.label_dir / (img_path.stem + ".txt")
            # Include image even if label file is absent (treat as no objects)
            self.samples.append((img_path, label_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}. "
                f"Supported extensions: {_IMAGE_EXTENSIONS}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_yolo_label(
        label_path: Path,
        img_w: int,
        img_h: int,
        label_offset: int,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Parse a YOLO .txt label file.

        Returns:
            bboxes_pvoc: List of [x1, y1, x2, y2] in absolute pixel coords.
            labels:      Corresponding class ids (with label_offset applied).
        """
        bboxes_pvoc: List[List[float]] = []
        labels: List[int] = []

        if not label_path.exists():
            return bboxes_pvoc, labels  # image with no annotations

        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue  # malformed line

                cls_id = int(parts[0]) + label_offset
                xc     = float(parts[1])   # normalized x_center
                yc     = float(parts[2])   # normalized y_center
                bw     = float(parts[3])   # normalized width
                bh     = float(parts[4])   # normalized height

                # Convert YOLO normalized xywh → absolute pascal_voc xyxy
                x1 = max(0.0, (xc - bw / 2) * img_w)
                y1 = max(0.0, (yc - bh / 2) * img_h)
                x2 = min(float(img_w), (xc + bw / 2) * img_w)
                y2 = min(float(img_h), (yc + bh / 2) * img_h)

                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate box, skip

                bboxes_pvoc.append([x1, y1, x2, y2])
                labels.append(cls_id)

        return bboxes_pvoc, labels

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label_path = self.samples[idx]

        # ---- Load image ----
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w  = image.shape[:2]

        # ---- Parse annotations ----
        bboxes_pvoc, labels = self._parse_yolo_label(
            label_path, w, h, self.label_offset
        )

        # ---- Apply transforms ----
        if self.transforms is not None and len(bboxes_pvoc) > 0:
            augmented  = self.transforms(image=image, bboxes=bboxes_pvoc, labels=labels)
            image_t    = augmented["image"]               # (3, H, W) tensor
            aug_boxes  = list(augmented["bboxes"])        # pascal_voc absolute
            aug_labels = list(augmented["labels"])
        elif self.transforms is not None:
            # No boxes — still transform image
            augmented  = self.transforms(image=image, bboxes=[], labels=[])
            image_t    = augmented["image"]
            aug_boxes  = []
            aug_labels = []
        else:
            # Fallback: manual to-tensor (no normalization)
            image_t    = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            aug_boxes  = bboxes_pvoc
            aug_labels = labels

        # ---- Normalize boxes to [0, 1] ----
        _, img_h, img_w = image_t.shape
        if len(aug_boxes) > 0:
            boxes_tensor = torch.tensor(aug_boxes, dtype=torch.float32)
            boxes_tensor[:, [0, 2]] /= img_w
            boxes_tensor[:, [1, 3]] /= img_h
            boxes_tensor = boxes_tensor.clamp(0.0, 1.0)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        labels_tensor = torch.tensor(aug_labels, dtype=torch.long)

        return {
            "image":  image_t,
            "boxes":  boxes_tensor,
            "labels": labels_tensor,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function for variable-length bounding box lists.

    Returns:
        images:  (B, 3, H, W) stacked tensor
        targets: list of {'boxes': (Mi, 4), 'labels': (Mi,)} dicts
    """
    images  = torch.stack([item["image"]  for item in batch], dim=0)
    targets = [{"boxes": item["boxes"], "labels": item["labels"]} for item in batch]
    return images, targets


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root: str,
    image_size: int = 300,
    batch_size: int = 16,
    num_workers: int = 4,
    label_offset: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a Kaggle-style YOLO dataset.

    Args:
        data_root:    Absolute path to the dataset root (contains images/ and labels/).
        image_size:   Resize target (pixels).
        batch_size:   Samples per batch.
        num_workers:  DataLoader worker processes.
        label_offset: Added to every raw YOLO class id (use 1 if background=0).

    Returns:
        (train_loader, val_loader)
    """
    train_ds = YOLODetectionDataset(
        root=data_root,
        split="train",
        transforms=get_train_transforms(image_size),
        image_size=image_size,
        label_offset=label_offset,
    )
    val_ds = YOLODetectionDataset(
        root=data_root,
        split="val",
        transforms=get_val_transforms(image_size),
        image_size=image_size,
        label_offset=label_offset,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Dataset root : {data_root}")
    print(f"  Train      : {len(train_ds)} samples")
    print(f"  Val        : {len(val_ds)} samples")
    return train_loader, val_loader
