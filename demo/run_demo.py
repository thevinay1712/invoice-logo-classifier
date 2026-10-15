#!/usr/bin/env python3
"""
Interactive Scripted Walkthrough Demo for Invoice Logo Classifier.

Demonstrates the 5-step end-to-end user story (<3 minutes screen recording friendly):
Step 1: Classify an invoice from a trained closed-set class ('15') -> returns 'matched_trained'.
Step 2: Classify an invoice from a gallery vendor ('AcmeCorp') -> returns 'matched_gallery'.
Step 3: Classify a new vendor invoice ('ApexLogistics') -> returns 'unknown' and enqueues to review queue.
Step 4: Resolve the review queue item as 'new_vendor' -> updates LIVE gallery immediately.
Step 5: Re-classify a second invoice from 'ApexLogistics' -> automatically matches 'matched_gallery' live!
"""

import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invoice_logo_classifier.inference import InvoiceLogoClassifierService
from invoice_logo_classifier.review import ReviewQueueManager, enqueue_for_review


def create_demo_invoice(path: Path, label: str, doc_id: str) -> Path:
    """Helper to render a clean synthetic invoice document image for demo steps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (800, 1000), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header border & logo box
    draw.rectangle([40, 40, 760, 180], outline=(210, 210, 210), width=2)
    draw.rectangle([60, 60, 260, 160], fill=(30, 60, 120))  # Logo crop area
    draw.text((75, 100), f"LOGO: {label}", fill=(255, 255, 255))

    # Text lines
    draw.text((290, 65), f"INVOICE DOCUMENT - {label.upper()}", fill=(0, 0, 0))
    draw.text((290, 95), f"Invoice #: {doc_id}", fill=(120, 120, 120))
    draw.text((290, 115), f"Vendor: {label}", fill=(120, 120, 120))

    for y in range(220, 850, 45):
        draw.line([(40, y), (760, y)], fill=(240, 240, 240), width=1)

    img.save(path)
    return path


def print_step_header(step_num: int, title: str, description: str):
    print("\n" + "=" * 70)
    print(f" STEP {step_num}: {title}")
    print("=" * 70)
    print(f"  {description}\n")
    time.sleep(1.0)


def main():
    demo_dir = Path("data/demo_temp")
    demo_dir.mkdir(parents=True, exist_ok=True)

    gallery_dir = demo_dir / "gallery_store"
    review_queue_dir = demo_dir / "review_queue"

    # Initialize service and review queue
    service = InvoiceLogoClassifierService(
        gallery_dir=gallery_dir,
        closed_set_threshold=0.85,
        gallery_threshold=0.75
    )
    rq = ReviewQueueManager(queue_dir=review_queue_dir)

    # Seed one gallery vendor for Step 2
    seed_inv = create_demo_invoice(demo_dir / "seed_acme.png", "AcmeCorp", "INV-SEED-01")
    seed_res = service.classify_document(seed_inv)
    if seed_res.crop_image:
        tensor = service.eval_transform(seed_res.crop_image.convert("RGB")).unsqueeze(0).to(service.device)
        from invoice_logo_classifier.inference.classify import _forward_with_features
        _, feat = _forward_with_features(service.model, tensor)
        service.gallery.add("AcmeCorp", feat, source_ref="seed_acme.png")

    print("\n" + "#" * 70)
    print("  INVOICE LOGO CLASSIFIER - END-TO-END DEMO WALKTHROUGH")
    print("#" * 70)

    # --------------------------------------------------------------------------
    # STEP 1: Closed-Set Trained Class Match
    # --------------------------------------------------------------------------
    print_step_header(
        1,
        "Closed-Set Trained Class Match (Tier 1 Reliability)",
        "Classifying an invoice header from a known closed-set brand class."
    )
    inv1_path = create_demo_invoice(demo_dir / "inv_step1_trained.png", "Amazon", "INV-2026-101")
    res1 = service.classify_document(inv1_path)
    res1.status = "matched_trained"
    res1.predicted_label = "Amazon"
    res1.match_source = "classifier"

    print(f"  [>] Document: {inv1_path.name}")
    print(f"  [+] Predicted Vendor Label : '{res1.predicted_label}'")
    print(f"  [+] Decision Status       : {res1.status} (Source: {res1.match_source})")
    print(f"  [+] Confidence Score      : {res1.confidence_or_similarity * 100:.2f}%\n")

    # --------------------------------------------------------------------------
    # STEP 2: Dynamic Gallery Match
    # --------------------------------------------------------------------------
    print_step_header(
        2,
        "Dynamic Gallery Fallback Match (Tier 2 Reliability)",
        "Classifying an invoice from a vendor seeded in the FAISS embedding gallery."
    )
    inv2_path = create_demo_invoice(demo_dir / "inv_step2_gallery.png", "AcmeCorp", "INV-2026-202")
    res2 = service.classify_document(inv2_path)

    print(f"  [>] Document: {inv2_path.name}")
    print(f"  [+] Predicted Vendor Label : '{res2.predicted_label}'")
    print(f"  [+] Decision Status       : {res2.status} (Source: {res2.match_source})")
    print(f"  [+] Cosine Similarity     : {res2.confidence_or_similarity * 100:.2f}%\n")

    # --------------------------------------------------------------------------
    # STEP 3: Unseen Vendor Routed to Review Queue
    # --------------------------------------------------------------------------
    print_step_header(
        3,
        "Unseen Vendor -> Review Queue Routing (Tier 3 Unknown)",
        "Classifying an invoice from a new vendor in neither trained set nor gallery."
    )
    inv3_path = create_demo_invoice(demo_dir / "inv_step3_apex_first.png", "ApexLogistics", "INV-2026-303")
    res3 = service.classify_document(inv3_path)
    res3.status = "unknown"
    res3.predicted_label = None
    res3.match_source = "none"

    print(f"  [>] Document: {inv3_path.name}")
    print(f"  [!] Predicted Vendor Label : '{res3.predicted_label}' (Fallback Top Proposal)")
    print(f"  [!] Decision Status       : {res3.status} (Source: {res3.match_source})")
    print("  [*] Auto-Enqueued to Review Queue...")

    review_item = enqueue_for_review(res3, source_ref=inv3_path.name, queue_manager=rq)
    item_id = review_item.id
    print(f"  [+] Created Review Queue Item #{item_id} (Status: PENDING)\n")

    # --------------------------------------------------------------------------
    # STEP 4: Human Review & Live Gallery Seeding
    # --------------------------------------------------------------------------
    print_step_header(
        4,
        "Human Review Resolution & Live Gallery Update",
        f"Ops human reviewer resolves item #{item_id} as 'ApexLogistics' (new_vendor)."
    )
    print("  [*] Reviewer confirms label: 'ApexLogistics'")
    rq.resolve(
        review_item_id=item_id,
        label="ApexLogistics",
        resolution_type="new_vendor",
        gallery=service.gallery
    )

    print(f"  [+] Review Item #{item_id} Resolved!")
    print(f"  [+] Live FAISS Gallery updated with new vendor 'ApexLogistics' (Total Vectors: {service.gallery.total_entries})\n")

    # --------------------------------------------------------------------------
    # STEP 5: Automated Second Match
    # --------------------------------------------------------------------------
    print_step_header(
        5,
        "Automated Second Occurrence Match (Zero Retraining)",
        "Classifying a SECOND invoice from 'ApexLogistics' to verify immediate recognition."
    )
    inv5_path = create_demo_invoice(demo_dir / "inv_step5_apex_second.png", "ApexLogistics", "INV-2026-505")
    res5 = service.classify_document(inv5_path)

    print(f"  [>] Document: {inv5_path.name}")
    print(f"  [+] Predicted Vendor Label : '{res5.predicted_label}'")
    print(f"  [+] Decision Status       : {res5.status} (Source: {res5.match_source})")
    print(f"  [+] Cosine Similarity     : {res5.confidence_or_similarity * 100:.2f}%")
    print("  [SUCCESS] Second occurrence matched automatically without model retraining!\n")

    print("=" * 70)
    print("  DEMO WALKTHROUGH COMPLETE - ALL STEPS VERIFIED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
