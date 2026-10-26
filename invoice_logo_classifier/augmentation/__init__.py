"""Augmentation module: image augmentation pipeline for dataset robustness."""

from invoice_logo_classifier.augmentation.clutter import (
    generate_clutter_crop,
    generate_clutter_dataset,
)
from invoice_logo_classifier.augmentation.transforms import (
    RandomJPEGCompression,
    RandomLowDPISimulation,
    RandomPhotocopyContrast,
    RandomScannerBanding,
    get_eval_transforms,
    get_train_transforms,
)

__all__ = [
    "RandomJPEGCompression",
    "RandomLowDPISimulation",
    "RandomPhotocopyContrast",
    "RandomScannerBanding",
    "generate_clutter_crop",
    "generate_clutter_dataset",
    "get_eval_transforms",
    "get_train_transforms",
]
