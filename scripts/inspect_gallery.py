"""CLI script to inspect stored vendor logo embeddings in the gallery."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from invoice_logo_classifier.config import settings
from invoice_logo_classifier.gallery import VendorGallery


def main():
    parser = argparse.ArgumentParser(description="Inspect stored vendor logo embeddings in the gallery database.")
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=settings.gallery_dir,
        help="Path to saved gallery directory containing FAISS index and SQLite db.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" VENDOR EMBEDDING GALLERY INSPECTION")
    print("=" * 70)
    print(f" Target Gallery Path : '{args.gallery_dir}'")
    print("=" * 70)

    if not args.gallery_dir.exists() or not (args.gallery_dir / "gallery_index.faiss").exists():
        print(
            f"\n[INFO] Gallery store not found or empty at '{args.gallery_dir}'. "
            "No vendor embeddings registered yet.",
            file=sys.stderr,
        )
        sys.exit(0)

    try:
        gallery = VendorGallery.load(args.gallery_dir)
    except Exception as err:  # noqa: BLE001 – FAISS/pickle load can raise many types; clean exit is correct
        print(f"\n[ERROR] Failed to load gallery: {err}", file=sys.stderr)
        sys.exit(1)

    summary = gallery.get_vendor_summary()

    print(f"\n Total Stored Embeddings : {gallery.total_entries:,}")
    print(f" Vector Dimension        : {gallery.embedding_dim}")
    print(f" Unique Vendors Stored   : {len(summary):,}")
    print("-" * 70)
    print(" VENDOR EMBEDDING COUNTS BREAKDOWN")
    print("-" * 70)
    print(f" {'Vendor Label':<35} | {'Embedding Count':<15}")
    print("-" * 70)

    for vendor_label, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        print(f" {vendor_label:<35} | {count:<15}")

    print("-" * 70)
    print("=" * 70)
    print(" Inspection complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
