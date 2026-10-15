"""Module for filtering datasets, performing train/val/test splits, capping class sizes, and building ImageFolder layout."""

import json
import logging
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from PIL import Image

from invoice_logo_classifier.config import settings

logger = logging.getLogger(__name__)


def crop_logo_region(image: Image.Image, bbox: list[int] | tuple[int, ...] | None) -> Image.Image:
    """Crop logo bounding box region from an image with boundary protection.

    Args:
        image: Source PIL Image.
        bbox: Bounding box coordinates [xmin, ymin, xmax, ymax] or None.

    Returns:
        Cropped PIL Image or original image if bbox is invalid/missing.
    """
    if not bbox or len(bbox) < 4:
        return image.copy()

    xmin, ymin, xmax, ymax = bbox[:4]
    w, h = image.size

    # Clamp coordinates to image boundaries
    xmin = max(0, min(int(xmin), w - 1))
    ymin = max(0, min(int(ymin), h - 1))
    xmax = max(xmin + 1, min(int(xmax), w))
    ymax = max(ymin + 1, min(int(ymax), h))

    if xmax <= xmin or ymax <= ymin:
        return image.copy()

    return image.crop((xmin, ymin, xmax, ymax))


def _extract_all_records(dataset: Dataset | DatasetDict) -> list[dict[str, Any]]:
    """Extract standard record dicts from Dataset or DatasetDict."""
    records = []
    if isinstance(dataset, DatasetDict):
        for split_name, ds_split in dataset.items():
            for idx in range(len(ds_split)):
                rec = dict(ds_split[idx])
                rec["_orig_split"] = split_name
                rec["_orig_idx"] = idx
                records.append(rec)
    elif isinstance(dataset, Dataset):
        for idx in range(len(dataset)):
            rec = dict(dataset[idx])
            rec["_orig_idx"] = idx
            records.append(rec)
    else:
        raise TypeError(f"Expected Dataset or DatasetDict, got {type(dataset)}")
    return records


def build_dataset(
    hf_dataset: Dataset | DatasetDict,
    selected_classes: list[str | int],
    output_dir: Path | str,
    val_frac: float | None = None,
    test_frac: float | None = None,
    max_images_per_class: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Filter dataset down to selected classes, perform clean train/val/test split, and save to disk in ImageFolder format.

    Args:
        hf_dataset: Source Hugging Face Dataset or DatasetDict.
        selected_classes: List of allowed class IDs or names.
        output_dir: Destination directory path for ImageFolder layout.
        val_frac: Fraction of data allocated to validation split (default settings.val_split_ratio).
        test_frac: Fraction of data allocated to test split (default settings.test_split_ratio).
        max_images_per_class: Training set per-class ceiling cap (default settings.max_images_per_class_train).
        seed: Random seed for reproducible shuffling (default settings.seed).

    Returns:
        Manifest dictionary summarizing dataset split metrics and class breakdown.
    """
    target_output_dir = Path(output_dir)
    target_val_frac = val_frac if val_frac is not None else settings.val_split_ratio
    target_test_frac = test_frac if test_frac is not None else settings.test_split_ratio
    target_max_train = (
        max_images_per_class
        if max_images_per_class is not None
        else settings.max_images_per_class_train
    )
    rng_seed = seed if seed is not None else settings.seed

    label_col = settings.label_column
    img_col = settings.image_column
    bbox_col = settings.bbox_column

    selected_set = {str(cls) for cls in selected_classes}

    logger.info(f"Extracting dataset records for {len(selected_set):,} selected target classes...")
    all_records = _extract_all_records(hf_dataset)

    # Group records by class
    class_grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in all_records:
        class_val = str(rec.get(label_col, ""))
        if class_val in selected_set:
            class_grouped.setdefault(class_val, []).append(rec)

    logger.info(f"Found matching records across {len(class_grouped):,} active classes.")

    rng = random.Random(rng_seed)

    split_records: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    per_class_split_counts: dict[str, dict[str, int]] = {
        cls_name: {"train": 0, "val": 0, "test": 0} for cls_name in selected_set
    }

    # Process class by class for stratified split
    for cls_name, records in class_grouped.items():
        # Shuffle records deterministically per class
        shuffled = list(records)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_test = max(1 if n_total >= 3 else 0, round(n_total * target_test_frac))
        n_val = max(1 if n_total >= 3 else 0, round(n_total * target_val_frac))
        n_train = n_total - n_test - n_val

        if n_train < 1:
            n_train = 1
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

        train_recs = shuffled[:n_train]
        val_recs = shuffled[n_train : n_train + n_val]
        test_recs = shuffled[n_train + n_val :]

        # Apply ceiling cap to training set
        if target_max_train is not None and target_max_train > 0:
            train_recs = train_recs[:target_max_train]

        for rec in train_recs:
            split_records["train"].append((cls_name, rec))
        for rec in val_recs:
            split_records["val"].append((cls_name, rec))
        for rec in test_recs:
            split_records["test"].append((cls_name, rec))

        per_class_split_counts[cls_name]["train"] = len(train_recs)
        per_class_split_counts[cls_name]["val"] = len(val_recs)
        per_class_split_counts[cls_name]["test"] = len(test_recs)

    # Save images to disk in ImageFolder format
    logger.info(f"Saving processed dataset to '{target_output_dir}'...")

    total_saved_images = 0
    split_totals: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    global_image_id = 0

    for split_name in ("train", "val", "test"):
        recs = split_records[split_name]
        logger.info(f"Writing split '{split_name}' ({len(recs):,} images)...")

        for cls_name, rec in recs:
            cls_dir = target_output_dir / split_name / str(cls_name)
            cls_dir.mkdir(parents=True, exist_ok=True)

            img_data = rec.get(img_col)
            if not isinstance(img_data, Image.Image):
                if hasattr(img_data, "convert"):
                    img_data = img_data.convert("RGB")
                else:
                    logger.warning(f"Skipping record without valid PIL Image: {rec}")
                    continue

            bbox = rec.get(bbox_col)
            cropped_img = crop_logo_region(img_data, bbox)
            if cropped_img.mode != "RGB":
                cropped_img = cropped_img.convert("RGB")

            global_image_id += 1
            orig_idx = rec.get("_orig_idx", global_image_id)
            image_filename = f"img_{orig_idx:06d}_{global_image_id:06d}.jpg"
            save_path = cls_dir / image_filename
            cropped_img.save(save_path, format="JPEG", quality=95)

            total_saved_images += 1
            split_totals[split_name] += 1

    manifest = {
        "manifest_version": "1.0",
        "dataset_name": settings.hf_dataset_name,
        "seed": rng_seed,
        "max_images_per_class_train": target_max_train,
        "val_frac": target_val_frac,
        "test_frac": target_test_frac,
        "total_classes": len(selected_classes),
        "total_images": total_saved_images,
        "split_totals": split_totals,
        "per_class_counts": per_class_split_counts,
        "selected_classes": [str(c) for c in selected_classes],
    }

    # Save manifest inside processed output_dir and settings.dataset_manifest_path
    manifest_in_output = target_output_dir / "dataset_manifest.json"
    manifest_in_output.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_in_output, mode="w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if settings.dataset_manifest_path != manifest_in_output:
        settings.dataset_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.dataset_manifest_path, mode="w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    logger.info(f"Dataset build complete. Manifest written to '{manifest_in_output}'.")
    return manifest
