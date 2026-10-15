"""CLI script to build and crop the training dataset into an ImageFolder structure and manifest."""

import sys

from invoice_logo_classifier.config import settings
from invoice_logo_classifier.data import (
    build_dataset,
    download_and_cache,
    load_selected_classes,
)


def main():
    print("=" * 70)
    print(" BUILDING TRAINING DATASET (ImageFolder Layout)")
    print("=" * 70)

    try:
        selected_classes = load_selected_classes()
    except FileNotFoundError as err:
        print(f"\n[ERROR] {err}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(selected_classes):,} target classes from '{settings.selected_classes_path}'.")

    try:
        ds_dict = download_and_cache()
    except RuntimeError as err:
        print(f"\n[ERROR] Failed to load dataset: {err}", file=sys.stderr)
        sys.exit(1)

    output_dir = settings.processed_data_dir
    print(f"Processing dataset and writing to '{output_dir}'...")

    manifest = build_dataset(
        hf_dataset=ds_dict,
        selected_classes=selected_classes,
        output_dir=output_dir,
    )

    split_totals = manifest["split_totals"]
    total_images = manifest["total_images"]

    print("\n" + "-" * 70)
    print(" DATASET BUILD MANIFEST SUMMARY")
    print("-" * 70)
    print(f" Target Classes Included       : {manifest['total_classes']:,}")
    print(f" Training Images (Cap <= {manifest['max_images_per_class_train']}) : {split_totals['train']:,}")
    print(f" Validation Images             : {split_totals['val']:,}")
    print(f" Test Images                   : {split_totals['test']:,}")
    print(f" Total Processed Images        : {total_images:,}")
    print(f" Output ImageFolder Directory  : '{output_dir}'")
    print(f" Manifest Saved To             : '{output_dir / 'dataset_manifest.json'}'")
    print("-" * 70)

    print("=" * 70)
    print(" Dataset build complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
