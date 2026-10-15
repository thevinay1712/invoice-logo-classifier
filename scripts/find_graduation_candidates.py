#!/usr/bin/env python3
"""
Gallery Graduation Candidate Finder Script.

Inspects the dynamic VendorGallery vector database and identifies gallery vendors whose
accumulated confirmed embedding count clears a configurable threshold (default >= 40 embeddings,
matching Phase 2's minimum class threshold).

Outputs a human-reviewable candidate list to `data/graduation_candidates.json`.
NOTE: This script does NOT modify the model or dataset automatically — it produces an advisory
list for human review prior to offline retraining.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invoice_logo_classifier.config import settings
from invoice_logo_classifier.gallery import VendorGallery

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("find_graduation_candidates")


def find_graduation_candidates(
    gallery_dir: Path,
    min_embeddings: int = 40,
    output_json: Path | None = None,
) -> list[dict]:
    """Finds gallery vendors with embedding count >= min_embeddings."""
    if not (gallery_dir.exists() and (gallery_dir / "gallery_index.faiss").exists()):
        logger.warning(f"Vendor gallery directory '{gallery_dir}' does not exist or has no index. Returning empty candidates.")
        return []

    gallery = VendorGallery.load(gallery_dir)

    # Tally embedding vector count per vendor label
    counts: dict[str, int] = {}
    source_refs: dict[str, list[str]] = {}
    for meta in gallery.metadata:
        label = meta.get("vendor_label", "Unknown")
        counts[label] = counts.get(label, 0) + 1
        source_refs.setdefault(label, [])
        ref = meta.get("source_invoice_ref")
        if ref and ref not in source_refs[label]:
            source_refs[label].append(ref)

    candidates = []
    for label, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        if count >= min_embeddings:
            candidates.append({
                "vendor_label": label,
                "embedding_count": count,
                "min_threshold": min_embeddings,
                "status": "candidate_for_graduation",
                "sample_source_refs": source_refs[label][:10],
            })

    # Print summary report
    print("\n" + "=" * 65)
    print(f"  GALLERY VENDOR GRADUATION CANDIDATES (Threshold >= {min_embeddings})")
    print("=" * 65)
    if not candidates:
        print("  No gallery vendors have reached the minimum embedding threshold yet.")
        print(f"  Total unique vendors in gallery: {len(counts)}")
    else:
        print(f"{'Vendor Label':<35} | {'Embedding Count':<18} | {'Status'}")
        print("-" * 65)
        for c in candidates:
            print(f"{c['vendor_label']:<35} | {c['embedding_count']:<18} | READY FOR RETRAINING")
    print("=" * 65 + "\n")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({
                "min_embeddings_threshold": min_embeddings,
                "total_candidates": len(candidates),
                "candidates": candidates
            }, f, indent=2)
        logger.info(f"Graduation candidates written to '{output_json}'.")

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Identify gallery vendors eligible for promotion into trained closed-set classes.",
        epilog="Use the output JSON or vendor names with `scripts/retrain_with_new_labels.py --graduate-vendors VendorA,VendorB`."
    )
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=settings.gallery_dir,
        help="Directory containing VendorGallery FAISS index and metadata DB"
    )
    parser.add_argument(
        "--min-embeddings",
        type=int,
        default=40,
        help="Minimum embedding count required to nominate a gallery vendor for graduation (default: 40)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/graduation_candidates.json"),
        help="Path to save candidate JSON output (default: data/graduation_candidates.json)"
    )

    args = parser.parse_args()
    find_graduation_candidates(
        gallery_dir=args.gallery_dir,
        min_embeddings=args.min_embeddings,
        output_json=args.output_json
    )


if __name__ == "__main__":
    main()
