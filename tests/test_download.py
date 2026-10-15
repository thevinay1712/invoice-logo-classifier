"""Tests for dataset download, caching, and schema inspection module."""

from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from invoice_logo_classifier.data import download_and_cache, inspect_schema


def test_inspect_schema_returns_expected_metadata():
    """Verify inspect_schema correctly extracts splits, columns, and features."""
    sample_data = {"image": ["img1.png", "img2.png"], "label": [0, 1]}
    mock_dataset = Dataset.from_dict(sample_data)
    dataset_dict = DatasetDict({"train": mock_dataset})

    info = inspect_schema(dataset_dict)

    assert info["splits"] == ["train"]
    assert "image" in info["column_names"]
    assert "label" in info["column_names"]
    assert "label" in info["label_candidates"]


@patch("invoice_logo_classifier.data.download.load_dataset")
def test_download_and_cache_success(mock_load_dataset):
    """Verify download_and_cache calls load_dataset with correct parameters."""
    mock_ds = DatasetDict({"train": Dataset.from_dict({"a": [1]})})
    mock_load_dataset.return_value = mock_ds

    res = download_and_cache(dataset_name="dummy/dataset")

    assert res == mock_ds
    mock_load_dataset.assert_called_once_with("dummy/dataset", cache_dir=None)


@patch("invoice_logo_classifier.data.download.load_dataset")
def test_download_and_cache_network_error_raises_runtime_error(mock_load_dataset):
    """Verify download_and_cache catches connection errors and raises RuntimeError."""
    mock_load_dataset.side_effect = ConnectionError("Connection refused")

    with pytest.raises(RuntimeError) as exc_info:
        download_and_cache(dataset_name="nonexistent/dataset")

    assert "Network Connection Error" in str(exc_info.value)
