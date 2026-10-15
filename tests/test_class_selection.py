"""Unit tests for class counting, selection filtering, and persistence functions."""

import json

import pytest
from datasets import Dataset, DatasetDict

from invoice_logo_classifier.data.class_selection import (
    count_images_per_class,
    export_class_counts_to_csv,
    load_selected_classes,
    save_selected_classes,
    select_classes,
)


@pytest.fixture
def synthetic_dataset_dict() -> DatasetDict:
    """Fixture providing synthetic dataset for testing count_images_per_class."""
    train_data = {"company_name": ["brand_A", "brand_A", "brand_B", "brand_C"]}
    test_data = {"company_name": ["brand_A", "brand_B"]}
    return DatasetDict(
        {
            "train": Dataset.from_dict(train_data),
            "test": Dataset.from_dict(test_data),
        }
    )


@pytest.fixture
def synthetic_class_counts() -> dict[str, int]:
    """Fixture providing synthetic class counts."""
    return {
        "brand_alpha": 100,
        "brand_beta": 50,
        "brand_gamma": 40,
        "brand_delta": 39,
        "brand_epsilon": 10,
    }


def test_count_images_per_class(synthetic_dataset_dict):
    """Verify count_images_per_class accurately tallies class frequencies across splits."""
    counts = count_images_per_class(synthetic_dataset_dict, label_col="company_name")
    assert counts["brand_A"] == 3
    assert counts["brand_B"] == 2
    assert counts["brand_C"] == 1


def test_select_classes_min_count_filter(synthetic_class_counts):
    """Verify select_classes filters classes meeting minimum threshold (default 40)."""
    selected = select_classes(synthetic_class_counts, min_images_per_class=40)
    assert selected == ["brand_alpha", "brand_beta", "brand_gamma"]
    assert "brand_delta" not in selected
    assert "brand_epsilon" not in selected


def test_select_classes_target_count_cap(synthetic_class_counts):
    """Verify select_classes caps result when target_count is specified."""
    selected = select_classes(
        synthetic_class_counts, min_images_per_class=40, target_count=2
    )
    assert selected == ["brand_alpha", "brand_beta"]
    assert len(selected) == 2


def test_export_class_counts_to_csv(tmp_path, synthetic_class_counts):
    """Verify CSV export writes valid header and sorted rows."""
    csv_file = tmp_path / "test_counts.csv"
    export_class_counts_to_csv(synthetic_class_counts, csv_file)

    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8").splitlines()
    assert content[0] == "class_name,count"
    assert content[1] == "brand_alpha,100"
    assert content[2] == "brand_beta,50"


def test_save_and_load_selected_classes_roundtrip(tmp_path):
    """Verify save_selected_classes and load_selected_classes roundtrip persistence."""
    json_file = tmp_path / "selected_classes.json"
    original_classes = ["brand_alpha", "brand_beta", 2020]

    save_selected_classes(original_classes, json_file)
    assert json_file.exists()

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)

    loaded_classes = load_selected_classes(json_file)
    assert loaded_classes == ["brand_alpha", "brand_beta", 2020]
