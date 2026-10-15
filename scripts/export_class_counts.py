"""CLI script to count image frequencies for all LogoDet-3K classes and export to data/class_counts.csv."""

import sys

from invoice_logo_classifier.config import settings
from invoice_logo_classifier.data import (
    count_images_per_class,
    download_and_cache,
    export_class_counts_to_csv,
)


def main():
    print("Starting LogoDet-3K per-class image count export...")

    try:
        ds_dict = download_and_cache()
    except RuntimeError as err:
        print(f"\n[ERROR] Failed to load dataset: {err}", file=sys.stderr)
        sys.exit(1)

    print("Counting image occurrences across all classes...")
    class_counts = count_images_per_class(ds_dict)

    output_path = export_class_counts_to_csv(class_counts, settings.class_counts_path)

    print("-" * 60)
    print(f" SUCCESS: Exported {len(class_counts):,} unique classes to '{output_path}'")
    print("-" * 60)


if __name__ == "__main__":
    main()
