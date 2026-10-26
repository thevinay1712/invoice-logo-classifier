"""CLI script to run closed-set logo classifier training end-to-end."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging

from invoice_logo_classifier.config import settings
from invoice_logo_classifier.training import train_model

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run logo classifier fine-tuning pipeline.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet50", "clip_vit_b32"],
        help="Backbone architecture to fine-tune (resnet50 or clip_vit_b32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Total training epochs (default 5 for quick execution, standard 20 for GPU).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=1,
        help="Warmup epochs training only head with frozen backbone.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for classification head.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.processed_data_dir,
        help="Directory containing processed ImageFolder train/val datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.model_dir,
        help="Directory where trained model checkpoints and logs will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.seed,
        help="Random seed for PyTorch operations.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" RUNNING LOGO CLASSIFIER TRAINING")
    print("=" * 70)
    print(f" Selected Backbone Architecture : {args.backbone}")
    print(f" Target Processed Dataset Path  : '{args.data_dir}'")
    print(f" Model Checkpoint Output Path   : '{args.output_dir}'")
    print(f" Training Epochs (Warmup)       : {args.epochs} ({args.warmup_epochs} warmup)")
    print(f" Batch Size / Learning Rate     : {args.batch_size} / {args.lr}")
    print("=" * 70)

    try:
        metadata = train_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            backbone_name=args.backbone,
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            learning_rate=args.lr,
            seed=args.seed,
        )
    except FileNotFoundError as err:
        print(f"\n[ERROR] {err}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "-" * 70)
    print(" TRAINING SUMMARY & METADATA")
    print("-" * 70)
    print(f" Backbone Used              : {metadata['backbone_used']}")
    print(f" Total Target Classes       : {metadata['num_classes']}")
    print(f" Best Validation Accuracy   : {metadata['best_val_acc']}%")
    print(f" Final Validation Accuracy  : {metadata['final_val_acc']}%")
    print(f" Epochs Completed           : {metadata['epochs_trained']}")
    print(f" Checkpoints Saved To       : '{args.output_dir}'")
    print(f" Model Metadata JSON        : '{args.output_dir / 'model_metadata.json'}'")
    print("-" * 70)
    print("=" * 70)
    print(" Training execution complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
