"""Data module: handles dataset download, selection, and train/val/test splitting."""

from invoice_logo_classifier.data.class_selection import (
    count_images_per_class,
    export_class_counts_to_csv,
    load_selected_classes,
    save_selected_classes,
    select_classes,
)
from invoice_logo_classifier.data.dataset_builder import build_dataset, crop_logo_region
from invoice_logo_classifier.data.download import download_and_cache, inspect_schema

__all__ = [
    "build_dataset",
    "count_images_per_class",
    "crop_logo_region",
    "download_and_cache",
    "export_class_counts_to_csv",
    "inspect_schema",
    "load_selected_classes",
    "save_selected_classes",
    "select_classes",
]
