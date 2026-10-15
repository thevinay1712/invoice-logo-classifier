"""CLI script to run test split evaluation and confidence threshold calibration."""

import argparse
import csv
import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from invoice_logo_classifier.augmentation import get_eval_transforms
from invoice_logo_classifier.config import settings
from invoice_logo_classifier.evaluation import (
    evaluate_model,
    generate_evaluation_summary_md,
    plot_confusion_matrix,
    sweep_confidence_thresholds,
)
from invoice_logo_classifier.training.models import build_model


def main():
    parser = argparse.ArgumentParser(description="Run test set evaluation and threshold calibration sweep.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=settings.model_dir / "best_model.pt",
        help="Path to trained PyTorch model checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.processed_data_dir / "test",
        help="Path to processed test ImageFolder dataset split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.base_dir / "evaluation_results",
        help="Destination directory for evaluation reports and plots.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" RUNNING CLASSIFIER EVALUATION & THRESHOLD CALIBRATION")
    print("=" * 70)
    print(f" Model Checkpoint Path : '{args.model_path}'")
    print(f" Test Dataset Path     : '{args.data_dir}'")
    print(f" Results Output Path   : '{args.output_dir}'")
    print("=" * 70)

    if not args.model_path.exists():
        print(f"\n[ERROR] Model checkpoint not found at '{args.model_path}'. Run training first.", file=sys.stderr)
        sys.exit(1)

    if not args.data_dir.exists():
        print(f"\n[ERROR] Test dataset split not found at '{args.data_dir}'. Build dataset first.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint metadata and weights
    checkpoint = torch.load(args.model_path, weights_only=False, map_location=device)
    class_list = checkpoint.get("class_list", [])
    backbone_name = checkpoint.get("backbone_name", "resnet50")

    model = build_model(backbone_name=backbone_name, num_classes=len(class_list), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Load test dataset
    eval_transform = get_eval_transforms(image_size=settings.image_size)
    test_ds = ImageFolder(root=str(args.data_dir), transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"\nLoaded {len(test_ds):,} test images across {len(class_list)} target classes.")

    # Execute evaluation and threshold sweep
    report, y_true, probs = evaluate_model(model, test_loader, class_list, device=device)
    sweep_results = sweep_confidence_thresholds(y_true, probs)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-class metrics CSV
    metrics_csv_path = args.output_dir / "metrics.csv"
    with open(metrics_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "precision", "recall", "f1_score", "support"])
        for cls_name, m in report.per_class_metrics.items():
            writer.writerow([cls_name, m["precision"], m["recall"], m["f1_score"], m["support"]])

    # Save threshold sweep CSV
    sweep_csv_path = args.output_dir / "threshold_sweep.csv"
    with open(sweep_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "auto_coverage_pct", "auto_accuracy_pct", "fallback_rate_pct", "confident_count", "fallback_count"])
        for s in sweep_results:
            writer.writerow([s["threshold"], s["auto_coverage_pct"], s["auto_accuracy_pct"], s["fallback_rate_pct"], s["confident_count"], s["fallback_count"]])

    # Save raw confusion matrix CSV
    cm_csv_path = args.output_dir / "confusion_matrix.csv"
    np.savetxt(cm_csv_path, np.array(report.confusion_matrix), fmt="%d", delimiter=",")

    # Save plotted confusion matrix PNG
    cm_png_path = args.output_dir / "confusion_matrix.png"
    plot_confusion_matrix(report.confusion_matrix, class_list, output_path=cm_png_path)

    # Save markdown summary report
    summary_md_path = args.output_dir / "summary.md"
    generate_evaluation_summary_md(report, sweep_results, output_path=summary_md_path)

    print("\n" + "-" * 70)
    print(" EVALUATION SUMMARY RESULTS")
    print("-" * 70)
    print(f" Overall Test Accuracy  : {report.overall_accuracy:.2f}%")
    print(f" Macro F1 Score         : {report.macro_f1:.2f}%")
    print(f" Top Confused Pairs     : {len(report.confused_pairs)} pairs detected")
    print(f" Confusion Matrix PNG   : '{cm_png_path}'")
    print(f" Threshold Sweep CSV    : '{sweep_csv_path}'")
    print(f" Evaluation Summary MD  : '{summary_md_path}'")
    print("-" * 70)
    print("=" * 70)
    print(" Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
