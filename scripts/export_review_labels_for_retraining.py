"""CLI script to export classifier misses from review queue for Phase 16 model retraining."""

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from invoice_logo_classifier.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="Export classifier-missed trained class items from review queue for retraining."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=settings.review_queue_dir / "review_queue.db",
        help="Path to SQLite review queue database.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=settings.review_queue_dir / "classifier_misses.csv",
        help="Path to classifier_misses.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.data_dir / "retraining_misses_export.json",
        help="Destination path for retraining export JSON.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" EXPORTING CLASSIFIER MISSES FOR MODEL RETRAINING")
    print("=" * 70)
    print(f" Database Path     : '{args.db_path}'")
    print(f" Output Export JSON: '{args.output}'")
    print("=" * 70)

    misses = []

    # Read from SQLite database filtering specifically for 'classifier_missed_trained_class'
    if args.db_path.exists():
        conn = sqlite3.connect(str(args.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, invoice_source_ref, predicted_label, resolved_label, confidence_or_similarity, crop_image_path, resolved_at
            FROM review_queue
            WHERE status = 'resolved' AND resolution_type = 'classifier_missed_trained_class'
            ORDER BY id ASC
            """
        )
        rows = cursor.fetchall()
        conn.close()

        for r in rows:
            misses.append(
                {
                    "item_id": r[0],
                    "invoice_source_ref": r[1],
                    "predicted_label": r[2],
                    "resolved_label": r[3],
                    "confidence_score": r[4],
                    "crop_image_path": r[5],
                    "resolved_at": r[6],
                }
            )
    elif args.csv_path.exists():
        with open(args.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                misses.append(
                    {
                        "item_id": int(row["item_id"]),
                        "invoice_source_ref": row["invoice_source_ref"],
                        "predicted_label": row["predicted_label"],
                        "resolved_label": row["resolved_label"],
                        "confidence_score": float(row["confidence"]),
                        "crop_image_path": row["crop_image_path"],
                        "resolved_at": row["timestamp"],
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, mode="w", encoding="utf-8") as f:
        json.dump(
            {
                "export_version": "1.0",
                "total_classifier_misses": len(misses),
                "note": "Excludes new_vendor gallery-resolved items which are already handled via dynamic gallery lookup.",
                "items": misses,
            },
            f,
            indent=2,
        )

    print(f"\n Exported {len(misses):,} classifier-missed trained class items to '{args.output}'.")
    print("-" * 70)
    print("=" * 70)
    print(" Retraining export complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
