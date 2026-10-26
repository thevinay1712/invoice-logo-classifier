#!/usr/bin/env python3
"""CLI utility to generate hard negative non-logo document clutter dataset.

Populates `data/processed/{train,val,test}/_Non_Logo_Clutter` with realistic non-logo document
crops (addresses, barcodes, table lines, stamps, blank textured paper) for hard negative training.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure package root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from invoice_logo_classifier.augmentation import generate_clutter_dataset
from invoice_logo_classifier.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_clutter")


def main():
    parser = argparse.ArgumentParser(description="Generate hard negative non-logo document clutter crops.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.processed_data_dir,
        help="Root processed data directory containing train/, val/, test/ splits.",
    )
    parser.add_argument("--train-count", type=int, default=150, help="Number of train clutter crops.")
    parser.add_argument("--val-count", type=int, default=30, help="Number of val clutter crops.")
    parser.add_argument("--test-count", type=int, default=20, help="Number of test clutter crops.")
    parser.add_argument(
        "--class-name",
        type=str,
        default="_Non_Logo_Clutter",
        help="Target folder name for negative class.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" GENERATING HARD NEGATIVE DOCUMENT CLUTTER DATASET")
    print("=" * 70)
    print(f" Target Directory: '{args.output_dir}'")
    print(f" Class Name      : '{args.class_name}'")
    print(f" Counts          : {args.train_count} train, {args.val_count} val, {args.test_count} test")
    print("=" * 70)

    counts = generate_clutter_dataset(
        output_base_dir=args.output_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        class_name=args.class_name,
    )

    for split, count in counts.items():
        print(f" [OK] Created {count} crops under '{args.output_dir / split / args.class_name}'")

    print("=" * 70)
    print(" Hard negative clutter dataset generation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
