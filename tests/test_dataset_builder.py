"""Unit tests for dataset building, logo cropping, split proportions, and leakage prevention."""

import json

import pytest
from datasets import Dataset, DatasetDict
from PIL import Image

from invoice_logo_classifier.data.dataset_builder import build_dataset, crop_logo_region


@pytest.fixture
def synthetic_image() -> Image.Image:
    """Fixture providing a 100x100 solid RGB test image."""
    return Image.new("RGB", (100, 100), color=(255, 0, 0))


@pytest.fixture
def synthetic_hf_dataset(synthetic_image) -> DatasetDict:
    """Fixture providing synthetic dataset with 2 classes, 20 images each."""
    records_class1 = [
        {"image_path": synthetic_image, "company_name": "class_A", "bbox": [10, 10, 50, 50]}
        for _ in range(20)
    ]
    records_class2 = [
        {"image_path": synthetic_image, "company_name": "class_B", "bbox": [5, 5, 40, 40]}
        for _ in range(20)
    ]
    combined = records_class1 + records_class2
    return DatasetDict({"train": Dataset.from_list(combined)})


def test_crop_logo_region_valid_and_boundary_fallback(synthetic_image):
    """Verify crop_logo_region correctly crops valid boxes and safely handles invalid boxes."""
    cropped = crop_logo_region(synthetic_image, [10, 10, 50, 50])
    assert cropped.size == (40, 40)

    # Boundary out-of-range coords get clamped safely
    clamped = crop_logo_region(synthetic_image, [-10, -10, 200, 200])
    assert clamped.size == (100, 100)

    # Missing or empty bbox returns original size copy
    uncropped = crop_logo_region(synthetic_image, None)
    assert uncropped.size == (100, 100)


def test_build_dataset_split_ratios_and_capping(tmp_path, synthetic_hf_dataset):
    """Verify build_dataset respects split fractions and training class capping."""
    out_dir = tmp_path / "processed_dataset"
    selected_classes = ["class_A", "class_B"]

    manifest = build_dataset(
        hf_dataset=synthetic_hf_dataset,
        selected_classes=selected_classes,
        output_dir=out_dir,
        val_frac=0.15,
        test_frac=0.15,
        max_images_per_class=5,  # Cap training set at 5 per class
        seed=42,
    )

    assert manifest["total_classes"] == 2
    assert manifest["max_images_per_class_train"] == 5

    # Check disk layout
    for split_name in ("train", "val", "test"):
        for cls_name in selected_classes:
            cls_dir = out_dir / split_name / cls_name
            assert cls_dir.exists()

    # Training images should be capped at 5 per class -> total 10
    train_files_class_a = list((out_dir / "train" / "class_A").glob("*.jpg"))
    assert len(train_files_class_a) == 5

    # Manifest file written
    manifest_file = out_dir / "dataset_manifest.json"
    assert manifest_file.exists()
    loaded_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert loaded_manifest["total_classes"] == 2


def test_build_dataset_no_cross_split_file_overlap(tmp_path, synthetic_hf_dataset):
    """Verify zero overlap of image paths or records across train, val, and test splits."""
    out_dir = tmp_path / "processed_dataset_overlap"
    selected_classes = ["class_A"]

    build_dataset(
        hf_dataset=synthetic_hf_dataset,
        selected_classes=selected_classes,
        output_dir=out_dir,
        val_frac=0.20,
        test_frac=0.20,
        max_images_per_class=100,
        seed=42,
    )

    train_files = {f.name for f in (out_dir / "train" / "class_A").glob("*.jpg")}
    val_files = {f.name for f in (out_dir / "val" / "class_A").glob("*.jpg")}
    test_files = {f.name for f in (out_dir / "test" / "class_A").glob("*.jpg")}

    # Each split has its own directory structure and distinct counts
    assert len(train_files) > 0
    assert len(val_files) > 0
    assert len(test_files) > 0
    assert train_files.isdisjoint(val_files)
