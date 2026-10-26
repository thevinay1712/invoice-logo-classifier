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


def export_classifier_misses(
    db_path: Path | str | None = None,
    csv_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> list[dict]:
    """Export classifier-missed trained class items from review queue SQLite DB or CSV.

    Args:
        db_path: Path to SQLite review queue database.
        csv_path: Fallback path to classifier_misses.csv file.
        output_path: Destination path for retraining export JSON.

    Returns:
        List of exported miss dictionaries.
    """
    target_db = Path(db_path if db_path is not None else settings.review_queue_dir / "review_queue.db")
    target_csv = Path(csv_path if csv_path is not None else settings.review_queue_dir / "classifier_misses.csv")
    target_out = Path(output_path if output_path is not None else settings.data_dir / "retraining_misses_export.json")

    misses = []

    if target_db.exists():
        conn = sqlite3.connect(str(target_db))
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
    elif target_csv.exists():
        with open(target_csv, mode="r", encoding="utf-8") as f:
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

    target_out.parent.mkdir(parents=True, exist_ok=True)
    with open(target_out, mode="w", encoding="utf-8") as f:
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

    return misses


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

    misses = export_classifier_misses(
        db_path=args.db_path,
        csv_path=args.csv_path,
        output_path=args.output,
    )

    print(f"\n Exported {len(misses):,} classifier-missed trained class items to '{args.output}'.")
    print("-" * 70)
    print("=" * 70)
    print(" Retraining export complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
