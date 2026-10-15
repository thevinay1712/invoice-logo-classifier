"""Module for downloading, caching, and inspecting Hugging Face datasets."""

import logging
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

from invoice_logo_classifier.config import settings

logger = logging.getLogger(__name__)


def download_and_cache(
    cache_dir: Path | str | None = None,
    dataset_name: str | None = None,
) -> DatasetDict:
    """Download and cache the dataset programmatically via Hugging Face datasets.

    Args:
        cache_dir: Optional custom path for caching dataset files.
        dataset_name: Hugging Face dataset identifier. Defaults to settings.hf_dataset_name.

    Returns:
        DatasetDict containing dataset splits.

    Raises:
        RuntimeError: If dataset download fails due to network or connection issues.
    """
    target_dataset = dataset_name or settings.hf_dataset_name
    target_cache = cache_dir or settings.dataset_cache_dir

    cache_dir_str = str(target_cache) if target_cache is not None else None

    logger.info(f"Downloading dataset '{target_dataset}' (cache_dir: {cache_dir_str or 'default HF cache'})...")

    try:
        ds = load_dataset(target_dataset, cache_dir=cache_dir_str)
        if not isinstance(ds, DatasetDict):
            # If load_dataset returns a single Dataset (e.g. split specified), wrap in DatasetDict
            ds = DatasetDict({"train": ds})
        logger.info(f"Successfully loaded dataset '{target_dataset}'.")
        return ds
    except ConnectionError as e:
        error_msg = (
            f"Network Connection Error: Unable to connect to huggingface.co to download '{target_dataset}'. "
            f"Please verify internet connectivity or proxy settings. Details: {e}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = (
            f"Failed to download/load dataset '{target_dataset}' from Hugging Face. "
            f"Ensure internet access to huggingface.co is available and the repository exists. Details: {e}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def inspect_schema(dataset_dict: DatasetDict) -> dict[str, Any]:
    """Inspect and print the schema, splits, and column features of a DatasetDict.

    Args:
        dataset_dict: Loaded Hugging Face DatasetDict instance.

    Returns:
        Dictionary containing extracted schema metadata.
    """
    splits = list(dataset_dict.keys())
    first_split_name = splits[0]
    first_split = dataset_dict[first_split_name]

    column_names = first_split.column_names
    features = first_split.features

    label_candidates = [
        col for col in column_names if col.lower() in ("label", "class", "category", "brand", "name", "tag")
    ]

    print("\n" + "=" * 60)
    print(" DATASET SCHEMA INSPECTION")
    print("=" * 60)
    print(f"Available Splits : {splits}")
    print(f"Column Names     : {column_names}")
    print("\nFull Feature Schema:")
    for col, feat in features.items():
        print(f"  - {col}: {feat}")

    if label_candidates:
        print(f"\nPotential Label/Brand Candidates : {label_candidates}")
    else:
        print("\nNo obvious label candidate names detected standard names.")
    print("=" * 60 + "\n")

    return {
        "splits": splits,
        "column_names": column_names,
        "features": features,
        "label_candidates": label_candidates,
    }
