#!/usr/bin/env python3
"""CLI script to run bulk invoice vendor classification and document sorting into vendor directories.

Usage:
    python scripts/classify_bulk_invoices.py --input-dir data/unclassified_invoices --output-dir classified_invoices
"""

import argparse
import logging
import sys
from pathlib import Path

from invoice_logo_classifier.inference.vendor_organizer import organize_bulk_invoices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("classify_bulk_invoices")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify bulk unorganized invoices by vendor and sort them into vendor folders at one central location."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing bulk invoice documents (.pdf, .png, .jpg, .docx)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="classified_invoices",
        help="Output target directory for sorted vendor folders (default: classified_invoices)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them (default: copy)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        logger.error(f"Input directory '{input_path}' does not exist.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("      AI-Based Bulk Invoice Vendor Classification & Sorting Engine      ")
    print("=" * 70)
    print(f"Input Directory : {input_path.resolve()}")
    print(f"Output Directory: {output_path.resolve()}")
    print(f"File Operation  : {'Move' if args.move else 'Copy'}")
    print("-" * 70)

    summary = organize_bulk_invoices(
        input_dir=input_path,
        output_dir=output_path,
        copy_mode=not args.move,
    )

    print("\n" + "=" * 70)
    print("                   Vendor Classification Results Summary                  ")
    print("=" * 70)
    print(f"Total Invoices Processed : {summary.total_processed}")
    print(f"Matched Trained Brands   : {summary.matched_trained_count}")
    print(f"Matched Dynamic Gallery  : {summary.matched_gallery_count}")
    print(f"Unassigned / ReviewQueue : {summary.unassigned_count}")
    print("-" * 70)
    print("Vendors Identified:")
    for vendor, count in summary.vendor_breakdown.items():
        print(f"  • {vendor:<30}: {count} invoice(s)")
    print("-" * 70)
    print(f"Summary Report JSON saved to: {output_path / 'vendor_classification_report.json'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
