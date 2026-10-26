"""Training loop module for fine-tuning closed-set logo classification models."""

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from invoice_logo_classifier.augmentation import (
    get_eval_transforms,
    get_train_transforms,
)
from invoice_logo_classifier.config import settings
from invoice_logo_classifier.training.models import build_model, set_backbone_trainable

logger = logging.getLogger(__name__)


def compute_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    """Calculate inverse class frequency weights to handle class imbalance in loss computation.

    Strategy Choice: Weighted Loss (nn.CrossEntropyLoss weight tensor).
    Formula: weight_c = total_samples / (num_classes * count_c)
    This penalizes misclassifications on rare logo classes more heavily without throwing away samples.

    Args:
        targets: List of integer class target labels across training dataset.
        num_classes: Total number of target classes.

    Returns:
        torch.FloatTensor of shape (num_classes,) containing normalized weights.
    """
    counts = [0] * num_classes
    for t in targets:
        if 0 <= t < num_classes:
            counts[t] += 1

    total_samples = len(targets)
    weights = []
    for c in range(num_classes):
        cnt = counts[c]
        if cnt > 0:
            w = total_samples / (float(num_classes) * cnt)
        else:
            w = 1.0
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so mean weight is 1.0
    if weights_tensor.sum() > 0:
        weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
    scaler: GradScaler | None = None,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Execute one training or evaluation epoch over a DataLoader.

    Args:
        model: The model being trained or evaluated.
        dataloader: DataLoader producing (images, labels) batches.
        criterion: Loss function.
        optimizer: Optimizer (None for evaluation mode).
        device: Target compute device.
        scaler: Optional GradScaler for mixed-precision training on CUDA.
        max_grad_norm: Max gradient norm for gradient clipping (training only).

    Returns:
        (average_loss, accuracy_percentage)
    """
    resolved_device = device if device is not None else torch.device("cpu")
    is_train = optimizer is not None
    use_amp = scaler is not None and resolved_device.type == "cuda"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in dataloader:
            images = images.to(resolved_device)
            labels = labels.to(resolved_device)

            if is_train:
                optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


def train_model(
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    backbone_name: str = "resnet50",
    batch_size: int = 32,
    epochs: int = 20,
    warmup_epochs: int = 2,
    learning_rate: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Train closed-set logo classifier using multi-phase fine-tuning (warmup + backbone unfreeze).

    Args:
        data_dir: Path to ImageFolder processed dataset root (containing train, val splits).
        output_dir: Directory where trained checkpoints and logs will be saved.
        backbone_name: Backbone model architecture ('resnet50' or 'clip_vit_b32').
        batch_size: DataLoader batch size.
        epochs: Total training epochs (including warmup).
        warmup_epochs: Initial epochs training only classification head with frozen backbone.
        learning_rate: Head learning rate.
        patience: Early stopping patience (validation loss).
        seed: Random seed for PyTorch reproducibility.
        device: 'cuda' or 'cpu' device string.

    Returns:
        Summary metrics dictionary.
    """
    torch.manual_seed(seed)
    target_data_dir = Path(data_dir if data_dir is not None else settings.processed_data_dir)
    target_output_dir = Path(output_dir if output_dir is not None else settings.model_dir)
    target_output_dir.mkdir(parents=True, exist_ok=True)

    target_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {target_device} for training.")

    train_path = target_data_dir / "train"
    val_path = target_data_dir / "val"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Training/Validation ImageFolder datasets not found in '{target_data_dir}'. Build dataset first."
        )

    # Load dataset transforms
    train_transform = get_train_transforms(image_size=settings.image_size)
    eval_transform = get_eval_transforms(image_size=settings.image_size)

    train_ds = ImageFolder(root=str(train_path), transform=train_transform)
    val_ds = ImageFolder(root=str(val_path), transform=eval_transform)

    num_classes = len(train_ds.classes)
    class_list = train_ds.classes
    logger.info(f"Loaded {num_classes} classes and {len(train_ds)} train / {len(val_ds)} val samples.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    # Build model architecture
    model = build_model(backbone_name=backbone_name, num_classes=num_classes, pretrained=True)
    model.to(target_device)

    # Calculate class weights for cross-entropy loss with label smoothing
    targets = [sample[1] for sample in train_ds.samples]
    class_weights = compute_class_weights(targets, num_classes=num_classes).to(target_device)
    # label_smoothing=0.1 reduces overconfidence and prevents memorization of synthetic/limited data
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Prepare CSV logger
    csv_log_path = target_output_dir / "training_log.csv"
    with open(csv_log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # ── Phase 1: Warmup — freeze backbone, train head only ──────────────────
    set_backbone_trainable(model, trainable=False)
    warmup_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=settings.weight_decay,
    )
    warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        warmup_optimizer, T_max=max(warmup_epochs, 1), eta_min=learning_rate * 0.01
    )

    # ── Phase 2: Fine-tune — unfreeze last 2 backbone blocks + head ─────────
    set_backbone_trainable(model, trainable=True, unfreeze_last_n_blocks=2)
    head_params = [p for n, p in model.named_parameters() if "fc" in n and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad]
    finetune_optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": learning_rate},
            {"params": backbone_params, "lr": learning_rate * 0.1},
        ],
        weight_decay=settings.weight_decay,
    )
    finetune_epochs = max(epochs - warmup_epochs, 1)
    finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        finetune_optimizer, T_max=finetune_epochs, eta_min=learning_rate * 0.001
    )

    # Mixed-precision scaler (only effective on CUDA)
    scaler = GradScaler() if target_device.type == "cuda" else None

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    last_epoch = 1
    val_loss = 0.0
    val_acc = 0.0

    for epoch in range(1, epochs + 1):
        last_epoch = epoch

        # Select correct optimizer/scheduler based on phase
        if epoch <= warmup_epochs:
            phase = "warmup"
            set_backbone_trainable(model, trainable=False)
            optimizer = warmup_optimizer
            scheduler = warmup_scheduler
        else:
            phase = "fine_tune"
            if epoch == warmup_epochs + 1:
                # Transition to fine-tune phase on first fine-tune epoch
                set_backbone_trainable(model, trainable=True, unfreeze_last_n_blocks=2)
            optimizer = finetune_optimizer
            scheduler = finetune_scheduler

        # Run train and validation epochs
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer=optimizer,
            device=target_device, scaler=scaler,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer=None, device=target_device
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({phase}) | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}"
        )

        # Log metrics to CSV
        with open(csv_log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, phase, f"{train_loss:.4f}", f"{train_acc:.2f}", f"{val_loss:.4f}", f"{val_acc:.2f}", f"{current_lr:.6f}"])

        # Track best val accuracy
        best_val_acc = max(val_acc, best_val_acc)

        # Early stopping: monitor val_loss improvement (more robust than val_acc)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_checkpoint_path = target_output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "class_list": class_list,
                    "backbone_name": backbone_name,
                },
                best_checkpoint_path,
            )
            logger.info(f"Saved new best model checkpoint (val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%).")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > warmup_epochs:
                logger.info(f"Early stopping triggered at epoch {epoch} (no val_loss improvement for {patience} epochs).")
                break

    # Save final model state
    final_checkpoint_path = target_output_dir / "final_model.pt"
    torch.save(
        {
            "epoch": last_epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "class_list": class_list,
            "backbone_name": backbone_name,
        },
        final_checkpoint_path,
    )

    # Write model metadata JSON
    metadata = {
        "backbone_used": backbone_name,
        "num_classes": num_classes,
        "class_list": class_list,
        "training_date": datetime.now(tz=UTC).isoformat(),
        "best_val_acc": round(best_val_acc, 2),
        "final_val_acc": round(val_acc, 2),
        "image_size": list(settings.image_size),
        "seed": seed,
        "epochs_trained": last_epoch,
    }

    metadata_path = target_output_dir / "model_metadata.json"
    with open(metadata_path, mode="w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training complete. Metadata written to '{metadata_path}'.")
    return metadata
