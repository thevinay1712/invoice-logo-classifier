"""CLI script to review class selection, generate selected_classes.json, and print audit breakdown."""

import csv
import sys
from pathlib import Path

from invoice_logo_classifier.config import settings
from invoice_logo_classifier.data import (
    count_images_per_class,
    download_and_cache,
    export_class_counts_to_csv,
    save_selected_classes,
    select_classes,
)


def load_class_counts_from_csv(csv_path: Path) -> dict[str, int]:
    """Load class counts from CSV file."""
    if not csv_path.exists():
        return {}
    counts = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return {}
        for row in reader:
            if len(row) >= 2:
                counts[row[0]] = int(row[1])
    return counts


def main():
    print("=" * 70)
    print(" CLASS SELECTION REVIEW & AUDIT")
    print("=" * 70)

    csv_path = settings.class_counts_path
    if not csv_path.exists():
        print(f"'{csv_path}' not found. Generating class counts from dataset...")
        try:
            ds_dict = download_and_cache()
            class_counts = count_images_per_class(ds_dict)
            export_class_counts_to_csv(class_counts, csv_path)
        except RuntimeError as err:
            print(f"\n[ERROR] Failed to download/count dataset: {err}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Loading class counts from '{csv_path}'...")
        class_counts = load_class_counts_from_csv(csv_path)

    min_threshold = settings.min_images_per_class
    target_cap = settings.target_class_count

    selected = select_classes(
        class_counts,
        min_images_per_class=min_threshold,
        target_count=target_cap,
    )

    json_path = save_selected_classes(selected, settings.selected_classes_path)

    # Calculate audit metrics
    selected_set = {str(c) for c in selected}
    excluded = {str(k): v for k, v in class_counts.items() if str(k) not in selected_set}

    total_classes = len(class_counts)
    total_images = sum(class_counts.values())
    selected_images = sum(class_counts[k] for k in selected if k in class_counts or str(k) in class_counts)

    print("\n" + "-" * 70)
    print(" SELECTION AUDIT SUMMARY")
    print("-" * 70)
    print(f" Total Unique Classes in Dataset : {total_classes:,}")
    print(f" Total Images in Dataset          : {total_images:,}")
    print(f" Minimum Images per Class Filter  : >= {min_threshold}")
    print(f" Selected Class Count             : {len(selected):,}")
    print(f" Images in Selected Classes       : {selected_images:,} ({selected_images / total_images * 100:.1f}% of total dataset)")
    print(f" Excluded Class Count             : {len(excluded):,}")
    print(f" Target Class JSON Saved To       : '{json_path}'")
    print("-" * 70)

    # Print selected classes table (top 20 & summary)
    print(f"\nSELECTED CLASSES TABLE (Showing top 20 of {len(selected)}):")
    print(f"{'Rank':<6} | {'Class Name / ID':<25} | {'Image Count':<12}")
    print("-" * 50)
    for idx, cls_name in enumerate(selected[:20], start=1):
        cnt = class_counts.get(cls_name, class_counts.get(str(cls_name), 0))
        print(f"{idx:<6} | {cls_name!s:<25} | {cnt:<12,}")
    if len(selected) > 20:
        print(f"... and {len(selected) - 20} more selected classes.")

    # Print sample excluded classes breakdown
    print(f"\nEXCLUDED CLASSES BREAKDOWN (Total: {len(excluded)} classes below {min_threshold} images):")
    excluded_sorted = sorted(excluded.items(), key=lambda item: item[1], reverse=True)
    print(f"{'Rank':<6} | {'Class Name / ID':<25} | {'Count':<8} | Reason")
    print("-" * 65)
    for idx, (cls_name, cnt) in enumerate(excluded_sorted[:10], start=1):
        print(f"{idx:<6} | {cls_name:<25} | {cnt:<8} | Count < {min_threshold} min threshold")
    if len(excluded_sorted) > 10:
        print(f"... and {len(excluded_sorted) - 10} more excluded classes.")

    print("\n" + "=" * 70)
    print(" Review & Audit complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
