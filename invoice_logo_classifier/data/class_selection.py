"""Module for counting per-class image frequencies, filtering target classes, and JSON/CSV persistence."""

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from invoice_logo_classifier.config import settings

logger = logging.getLogger(__name__)


def count_images_per_class(
    dataset: Dataset | DatasetDict,
    label_col: str | None = None,
) -> dict[Any, int]:
    """Iterate the dataset once and count row occurrences per class label.

    Args:
        dataset: A Hugging Face Dataset or DatasetDict.
        label_col: Name of column containing class label/ID. Defaults to settings.label_column.

    Returns:
        Dictionary mapping class label to total image count.
    """
    target_col = label_col or settings.label_column
    counter: Counter[Any] = Counter()

    if isinstance(dataset, DatasetDict):
        for split_name, ds_split in dataset.items():
            logger.info(f"Counting class occurrences in split '{split_name}'...")
            if target_col not in ds_split.column_names:
                raise KeyError(
                    f"Label column '{target_col}' not found in dataset split '{split_name}'. "
                    f"Available columns: {ds_split.column_names}"
                )
            # Efficient column extraction
            labels = ds_split[target_col]
            counter.update(labels)
    elif isinstance(dataset, Dataset):
        if target_col not in dataset.column_names:
            raise KeyError(
                f"Label column '{target_col}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )
        labels = dataset[target_col]
        counter.update(labels)
    else:
        raise TypeError(f"Expected Dataset or DatasetDict, got {type(dataset)}")

    logger.info(f"Counted images across {len(counter):,} unique classes.")
    return dict(counter)


def export_class_counts_to_csv(
    class_counts: dict[Any, int],
    csv_path: Path | str | None = None,
) -> Path:
    """Export per-class image counts to CSV sorted descending by image count.

    Args:
        class_counts: Dictionary mapping class label/name to image count.
        csv_path: Target CSV file path. Defaults to settings.class_counts_path.

    Returns:
        Path to generated CSV file.
    """
    target_path = Path(csv_path) if csv_path is not None else settings.class_counts_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_counts = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)

    with open(target_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "count"])
        for cls_name, cnt in sorted_counts:
            writer.writerow([str(cls_name), cnt])

    logger.info(f"Exported {len(sorted_counts):,} class counts to '{target_path}'.")
    return target_path


def select_classes(
    class_counts: dict[Any, int],
    min_images_per_class: int | None = None,
    target_count: int | None = None,
) -> list[Any]:
    """Filter classes meeting a minimum image count threshold, sorted descending.

    Args:
        class_counts: Dictionary mapping class label/name to image count.
        min_images_per_class: Minimum image count required for inclusion. Defaults to settings.min_images_per_class.
        target_count: Optional max limit on number of selected classes. Defaults to settings.target_class_count.

    Returns:
        List of selected class labels/names sorted descending by count.
    """
    min_threshold = (
        min_images_per_class
        if min_images_per_class is not None
        else settings.min_images_per_class
    )
    max_classes = (
        target_count if target_count is not None else settings.target_class_count
    )

    # Sort descending by count
    sorted_items = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)

    # Filter by minimum threshold
    filtered = [cls_name for cls_name, cnt in sorted_items if cnt >= min_threshold]

    # Apply optional cap
    if max_classes is not None and max_classes > 0:
        filtered = filtered[:max_classes]

    logger.info(
        f"Selected {len(filtered):,} classes (min_images={min_threshold}, target_cap={max_classes})."
    )
    return filtered


def save_selected_classes(
    selected_classes: list[Any],
    json_path: Path | str | None = None,
) -> Path:
    """Save selected class list to a human-editable JSON file.

    Args:
        selected_classes: List of selected class labels or names.
        json_path: Target JSON file path. Defaults to settings.selected_classes_path.

    Returns:
        Path to written JSON file.
    """
    target_path = (
        Path(json_path) if json_path is not None else settings.selected_classes_path
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-string labels (e.g. ints) to strings/primitives for clean JSON
    serializable_list = [
        int(c) if isinstance(c, (int, str)) and str(c).isdigit() else str(c)
        for c in selected_classes
    ]

    with open(target_path, mode="w", encoding="utf-8") as f:
        json.dump(serializable_list, f, indent=2)

    logger.info(f"Saved {len(serializable_list):,} selected classes to '{target_path}'.")
    return target_path


def load_selected_classes(
    json_path: Path | str | None = None,
) -> list[Any]:
    """Load selected class list from JSON file.

    Args:
        json_path: Source JSON file path. Defaults to settings.selected_classes_path.

    Returns:
        List of selected class labels/names.

    Raises:
        FileNotFoundError: If the selected classes JSON file does not exist.
    """
    target_path = (
        Path(json_path) if json_path is not None else settings.selected_classes_path
    )

    if not target_path.exists():
        raise FileNotFoundError(
            f"Selected classes file not found at '{target_path}'. "
            f"Run scripts/review_class_selection.py first to generate it."
        )

    with open(target_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"Expected list in '{target_path}', got {type(data)}")

    logger.info(f"Loaded {len(data):,} selected classes from '{target_path}'.")
    return data
