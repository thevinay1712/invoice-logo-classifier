#!/usr/bin/env python3
"""
Offline Model Retraining & Versioning Pipeline Script.

Executes a documented, repeatable offline retraining pass to improve the closed-set classifier:
1. Merges classifier misses from Phase 11 (`review_queue/classifier_misses.csv`) into existing class folders.
2. Promotes approved gallery vendors into new closed-set trained classes under `data/processed/{train,val,test}`.
3. Re-trains the classifier, outputting a NEW versioned checkpoint directory (`models/v2/`, `models/v3/`).
4. Runs evaluation and regression checks against the previous version (alerting if any class F1 drops >= 5%).
5. Cleans up graduated vendor entries from `VendorGallery` to prevent dual-system overlap.
6. Updates `models/ACTIVE_VERSION.json` pointer to activate the new version (with full `--rollback-to` support).
"""

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invoice_logo_classifier.config import settings
from invoice_logo_classifier.gallery import VendorGallery
from invoice_logo_classifier.training.train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrain_pipeline")


def rollback_version(version_name: str, model_dir: Path) -> None:
    """Switches the ACTIVE_VERSION.json pointer back to a previous model version."""
    target_dir = model_dir / version_name
    pointer_file = model_dir / "ACTIVE_VERSION.json"

    if not target_dir.exists() and version_name != "root":
        logger.error(f"Cannot rollback: version directory '{target_dir}' does not exist.")
        sys.exit(1)

    pointer_data = {
        "active_version": version_name,
        "activated_at": Path(__file__).stat().st_mtime,
        "comment": f"Rolled back to {version_name}"
    }

    with open(pointer_file, "w", encoding="utf-8") as f:
        json.dump(pointer_data, f, indent=2)

    logger.info(f"[✓] Active model version successfully set to '{version_name}'.")


def run_retraining_pipeline(
    misses_csv: Path,
    graduate_vendors: list[str],
    epochs: int = 10,
    warmup_epochs: int = 2,
    lr: float = 0.001,
    models_dir: Path | None = None,
    gallery_dir: Path | None = None,
    data_dir: Path | None = None,
) -> Path:
    """Runs end-to-end retraining, comparative evaluation, regression check, and gallery cleanup."""
    target_models_dir = models_dir or settings.model_dir
    target_gallery_dir = gallery_dir or settings.gallery_dir
    target_data_dir = data_dir or settings.processed_data_dir

    logger.info("Starting model retraining and versioning pipeline...")

    # 1. Determine next model version string (v1, v2, v3, ...)
    version_dirs = [d for d in target_models_dir.glob("v*") if d.is_dir() and d.name[1:].isdigit()]
    if version_dirs:
        max_v = max(int(d.name[1:]) for d in version_dirs)
        next_v_str = f"v{max_v + 1}"
    else:
        next_v_str = "v2"

    new_version_dir = target_models_dir / next_v_str
    new_version_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Targeting new version directory: '{new_version_dir}'")

    # 2. Integrate classifier misses into dataset structure
    miss_count = 0
    if misses_csv.exists():
        with open(misses_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                crop_path = Path(row.get("crop_image_path", ""))
                label = row.get("resolved_label")
                if crop_path.exists() and label:
                    dest_dir = target_data_dir / "train" / label
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / f"miss_{crop_path.name}"
                    shutil.copy2(crop_path, dest_file)
                    miss_count += 1
        logger.info(f"Integrated {miss_count} classifier misses into dataset structure.")

    # 3. Integrate graduated gallery vendors as new classes
    graduated_count = 0
    if graduate_vendors:
        for vendor in graduate_vendors:
            vendor_clean = vendor.strip()
            if not vendor_clean:
                continue

            # Create split directories
            for split in ["train", "val", "test"]:
                (target_data_dir / split / vendor_clean).mkdir(parents=True, exist_ok=True)

            # Copy crops from review queue / gallery source refs if available
            crops_src = settings.review_queue_dir / "crops"
            if crops_src.exists():
                for crop_img in crops_src.glob("*.png"):
                    shutil.copy2(crop_img, target_data_dir / "train" / vendor_clean / crop_img.name)

            graduated_count += 1
        logger.info(f"Incorporated {graduated_count} graduated vendors as new classes: {graduate_vendors}")

    # 4. Update selected_classes.json single source of truth
    selected_classes_path = settings.data_dir / "selected_classes.json"
    existing_classes = []
    if selected_classes_path.exists():
        with open(selected_classes_path, "r", encoding="utf-8") as f:
            existing_classes = json.load(f)

    updated_classes = list(dict.fromkeys(existing_classes + [v.strip() for v in graduate_vendors if v.strip()]))
    with open(selected_classes_path, "w", encoding="utf-8") as f:
        json.dump(updated_classes, f, indent=2)

    # 5. Execute Phase 5 Training into versioned directory
    logger.info(f"Training version {next_v_str} model for {epochs} epochs...")
    train_model(
        data_dir=target_data_dir,
        output_dir=new_version_dir,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        lr=lr,
        class_list=updated_classes,
    )

    # 6. Regression Check & Comparative Evaluation
    logger.info("Executing regression check against previous active version...")
    previous_f1s = {}
    active_pointer = target_models_dir / "ACTIVE_VERSION.json"
    if active_pointer.exists():
        try:
            with open(active_pointer, "r", encoding="utf-8") as f:
                prev_info = json.load(f)
            prev_v = prev_info.get("active_version")
            prev_meta_path = target_models_dir / prev_v / "model_metadata.json"
            if prev_meta_path.exists():
                with open(prev_meta_path, "r", encoding="utf-8") as f:
                    prev_meta = json.load(f)
                previous_f1s = prev_meta.get("per_class_f1", {})
        except Exception as e:  # noqa: BLE001 – previous metadata read is best-effort; failure is non-fatal
            logger.warning(f"Could not load previous version metadata: {e}")

    # Inspect new version metadata
    new_meta_path = new_version_dir / "model_metadata.json"
    new_f1s = {}
    if new_meta_path.exists():
        with open(new_meta_path, "r", encoding="utf-8") as f:
            new_meta = json.load(f)
        new_f1s = new_meta.get("per_class_f1", {})

    regressions = []
    for cls_name, prev_score in previous_f1s.items():
        new_score = new_f1s.get(cls_name, 0.0)
        drop = prev_score - new_score
        if drop >= 0.05:  # 5% F1 degradation threshold
            regressions.append((cls_name, prev_score, new_score, drop))

    print("\n" + "=" * 65)
    print(f"  MODEL RETRAINING COMPARISON ({next_v_str} vs Previous)")
    print("=" * 65)
    if regressions:
        print(f"  [!] ALERT: {len(regressions)} class(es) experienced F1 regression >= 5%:")
        for cls_name, p_s, n_s, d in regressions:
            print(f"      - Class '{cls_name}': Previous F1={p_s:.3f} -> New F1={n_s:.3f} (Drop: -{d*100:.1f}%)")
    else:
        print("  [✓] No catastrophic forgetting regressions detected!")
    print("=" * 65 + "\n")

    # 7. Gallery De-duplication: Remove graduated vendors from gallery
    if target_gallery_dir.exists() and (target_gallery_dir / "gallery_index.faiss").exists():
        gallery = VendorGallery.load(target_gallery_dir)
        for vendor in graduate_vendors:
            v_clean = vendor.strip()
            removed = gallery.remove_vendor(v_clean)
            logger.info(f"Removed {removed} entries for graduated vendor '{v_clean}' from gallery.")
        gallery.save(target_gallery_dir)

    # 8. Activate new model version
    rollback_version(next_v_str, target_models_dir)

    logger.info(f"[✓] Retraining pipeline complete. Version '{next_v_str}' is now ACTIVE.")
    return new_version_dir


def main():
    parser = argparse.ArgumentParser(
        description="Offline model retraining pipeline incorporating misses and graduating gallery vendors.",
        epilog="This is a deliberate, reviewed action triggered manually, NOT an automatic online loop."
    )
    parser.add_argument(
        "--misses-csv",
        type=Path,
        default=settings.review_queue_dir / "classifier_misses.csv",
        help="Path to classifier misses CSV exported from review queue"
    )
    parser.add_argument(
        "--graduate-vendors",
        type=str,
        default="",
        help="Comma-separated list of approved gallery vendor labels to promote into trained classes"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of fine-tuning training epochs (default: 10)"
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Warmup epochs training classification head only (default: 2)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for optimization (default: 0.001)"
    )
    parser.add_argument(
        "--rollback-to",
        type=str,
        default=None,
        help="Rollback active model pointer to specified version directory (e.g. v1, v2)"
    )

    args = parser.parse_args()

    if args.rollback_to:
        rollback_version(args.rollback_to, settings.model_dir)
        sys.exit(0)

    grad_list = [v.strip() for v in args.graduate_vendors.split(",") if v.strip()]
    run_retraining_pipeline(
        misses_csv=args.misses_csv,
        graduate_vendors=grad_list,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
