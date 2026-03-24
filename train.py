import os
import math
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model   import MobileNetV2SSD
from losses  import MultiBoxLoss
from dataset import build_dataloaders
from metrics import MeanAveragePrecision


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: Dict,
    checkpoint_dir: str,
    filename: str,
) -> None:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    print(f"  [Checkpoint] Saved → {path}")


def get_lr_with_warmup(
    optimizer: optim.Optimizer,
    warmup_iters: int,
    current_iter: int,
    base_lr: float,
) -> None:
    """Linear warm-up: scale LR from 0 → base_lr over warmup_iters steps."""
    if current_iter < warmup_iters:
        lr = base_lr * (current_iter + 1) / warmup_iters
        for pg in optimizer.param_groups:
            pg["lr"] = lr


def pixel_scale_targets(
    targets: list,
    image_size: int,
) -> list:
    """
    Scale normalized [0,1] xyxy boxes to absolute pixel coords for mAP metric.
    """
    scaled = []
    for t in targets:
        boxes = t["boxes"].clone()
        boxes = boxes * image_size          # both axes are image_size (square)
        scaled.append({"boxes": boxes, "labels": t["labels"]})
    return scaled


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: MobileNetV2SSD,
    criterion: MultiBoxLoss,
    optimizer: optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    warmup_iters: int,
    base_lr: float,
    global_step: int,
    grad_clip: float = 5.0,
) -> Tuple[float, int]:
    """
    Train for one epoch.

    Returns:
        avg_loss:    Mean loss over the epoch.
        global_step: Updated global iteration counter.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)

        # Warm-up LR
        get_lr_with_warmup(optimizer, warmup_iters, global_step, base_lr)

        # Forward
        loc_preds, cls_preds, anchors = model(images)

        # Move anchors and targets to device
        anchors_dev = anchors.to(device)
        targets_dev = [
            {k: v.to(device) for k, v in t.items()} for t in targets
        ]

        loss = criterion(loc_preds, cls_preds, anchors_dev, targets_dev)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss  += loss.item()
        global_step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.5f}")

    return total_loss / num_batches, global_step


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: MobileNetV2SSD,
    criterion: MultiBoxLoss,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    image_size: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Validate for one epoch and compute mAP.

    Returns:
        avg_loss: Mean validation loss.
        metrics:  Dict containing 'mAP' and per-class APs.
    """
    model.eval()
    total_loss = 0.0
    metric = MeanAveragePrecision(
        num_classes=model.num_classes, iou_threshold=0.5
    )

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        anchors_dev = None

        # Forward pass (loss)
        loc_preds, cls_preds, anchors = model(images)
        anchors_dev = anchors.to(device)
        targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = criterion(loc_preds, cls_preds, anchors_dev, targets_dev)
        total_loss += loss.item()

        # Decode predictions for mAP
        detections = model.predict(images, score_threshold=0.3)

        # Scale GT boxes from normalized [0,1] to pixels
        scaled_targets = pixel_scale_targets(targets, image_size)
        metric.update(detections, scaled_targets)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    results = metric.compute()
    return total_loss / len(loader), results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # ---- Device ----
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    # ---- Dataloaders ----
    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_offset=args.label_offset,
    )

    # ---- Model ----
    model = MobileNetV2SSD(
        num_classes=args.num_classes,
        width_mult=args.width_mult,
        image_size=args.image_size,
        score_threshold=0.3,
        nms_iou_threshold=0.45,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # ---- Load checkpoint if resuming ----
    start_epoch  = 1
    best_map     = 0.0
    global_step  = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_map    = ckpt.get("best_map", 0.0)
        global_step = ckpt.get("global_step", 0)

    # ---- Loss ----
    criterion = MultiBoxLoss(
        num_classes=args.num_classes,
        iou_threshold=0.5,
        neg_pos_ratio=3,
        alpha=1.0,
    )

    # ---- Optimizer ----
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # ---- LR Scheduler: cosine annealing (after warm-up) ----
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 1e-2,
    )

    warmup_iters = args.warmup_epochs * len(train_loader)
    base_lr = args.lr

    print(f"\nStarting training for {args.epochs} epochs")
    print(f"  Dataset root: {args.data_root}")
    print(f"  Num classes : {args.num_classes} (incl. background)")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Base LR     : {base_lr}")
    print(f"  Warm-up     : {args.warmup_epochs} epochs ({warmup_iters} iters)")
    print(f"  Checkpoints : {args.checkpoint_dir}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        train_loss, global_step = train_one_epoch(
            model, criterion, optimizer, train_loader,
            device, epoch, warmup_iters, base_lr, global_step,
        )

        # --- LR step (after warm-up) ---
        if epoch > args.warmup_epochs:
            scheduler.step()

        # --- Validate ---
        val_loss, val_metrics = validate(
            model, criterion, val_loader,
            device, epoch, args.image_size,
        )

        elapsed = time.time() - t0
        map_val = val_metrics["mAP"]

        # --- Logging ---
        print(
            f"Epoch [{epoch:>3}/{args.epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| mAP@0.5: {map_val:.4f} "
            f"| LR: {optimizer.param_groups[0]['lr']:.6f} "
            f"| {elapsed:.0f}s"
        )
        # Print per-class APs
        for k, v in val_metrics.items():
            if k.startswith("AP_class_"):
                print(f"  {k}: {v:.4f}")

        # --- Save checkpoints ---
        checkpoint_state = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "best_map":    best_map,
            "global_step": global_step,
            "args":        vars(args),
        }

        save_checkpoint(checkpoint_state, args.checkpoint_dir, "last.pth")

        if map_val > best_map:
            best_map = map_val
            checkpoint_state["best_map"] = best_map
            save_checkpoint(checkpoint_state, args.checkpoint_dir, "best.pth")
            print(f"  ★ New best mAP: {best_map:.4f}")

    print(f"\nTraining complete. Best mAP@0.5 = {best_map:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MobileNetV2-SSD for car detection"
    )

    # Dataset
    p.add_argument("--data_root",    type=str,   required=True,
                   help="Path to dataset root (must contain images/ and labels/ sub-dirs)")
    p.add_argument("--label_offset", type=int,   default=0,
                   help="Add this value to every YOLO class id "
                        "(e.g. 1 if background is class-0 in your pipeline)")
    p.add_argument("--num_classes",  type=int,   default=1,
                   help="Number of classes INCLUDING background")
    p.add_argument("--image_size",   type=int,   default=300)

    # Model
    p.add_argument("--width_mult",   type=float, default=1.0,
                   help="MobileNetV2 width multiplier")

    # Training
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-2)
    p.add_argument("--weight_decay",  type=float, default=4e-5)
    p.add_argument("--warmup_epochs", type=int,   default=3,
                   help="Epochs for linear LR warm-up")
    p.add_argument("--grad_clip",     type=float, default=5.0)
    p.add_argument("--num_workers",   type=int,   default=4)

    # Checkpoints
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--resume",         type=str, default="",
                   help="Path to checkpoint to resume from")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
