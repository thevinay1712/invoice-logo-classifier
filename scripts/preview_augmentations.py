"""CLI script to generate and save a 3x3 preview grid of augmented logo images."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid

from invoice_logo_classifier.augmentation import get_train_transforms
from invoice_logo_classifier.config import settings


def create_synthetic_test_image(size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Generate a synthetic test logo image with vibrant text and geometric shapes."""
    img = Image.new("RGB", size, color=(240, 242, 245))
    draw = ImageDraw.Draw(img)

    # Draw colorful logo shapes
    draw.rectangle([20, 20, 100, 100], fill=(220, 50, 50), outline=(0, 0, 0), width=2)
    draw.ellipse([110, 20, 190, 100], fill=(50, 150, 220), outline=(0, 0, 0), width=2)
    draw.polygon([(60, 120), (100, 190), (20, 190)], fill=(50, 200, 100), outline=(0, 0, 0))

    # Add stylized text
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001 – PIL font loading has no specific typed exception; fallback to None
        font = None

    draw.text((120, 140), "VENDOR", fill=(30, 30, 30), font=font)
    draw.text((120, 160), "LOGO TEST", fill=(100, 100, 100), font=font)

    return img


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Denormalize a float tensor back to [0, 1] range for image visualization."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return (tensor * std_t + mean_t).clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Preview training data augmentations in a 3x3 grid.")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional path to a sample image file. If omitted, uses a sample from processed dataset or generates a synthetic image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.data_dir / "augmentation_preview.png",
        help="Path where the output preview grid PNG will be saved.",
    )
    args = parser.parse_args()

    # Locate or generate input image
    source_image = None
    if args.image and args.image.exists():
        print(f"Loading provided image: '{args.image}'")
        source_image = Image.open(args.image).convert("RGB")
    else:
        # Try to find a sample image in data/processed/train/
        sample_images = list((settings.processed_data_dir / "train").rglob("*.jpg"))
        if sample_images:
            sample_path = sample_images[0]
            print(f"Loading dataset sample image: '{sample_path}'")
            source_image = Image.open(sample_path).convert("RGB")
        else:
            print("No existing processed dataset image found. Generating synthetic test logo image...")
            source_image = create_synthetic_test_image()

    transform = get_train_transforms(image_size=224)

    # Generate 9 augmented versions
    augmented_tensors = []
    print("Applying training augmentations to generate 9 grid samples...")
    for i in range(9):
        augmented_t = transform(source_image.copy())
        denorm_t = denormalize(augmented_t)
        augmented_tensors.append(denorm_t)

    # Stack into grid tensor (3x3 grid)
    grid_tensor = make_grid(augmented_tensors, nrow=3, padding=4, normalize=False)

    # Convert tensor to PIL Image and save
    grid_np = grid_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    grid_img = Image.fromarray(grid_np)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(output_path)

    print("\n" + "=" * 70)
    print(" AUGMENTATION PREVIEW GRID SAVED")
    print("=" * 70)
    print(f" Preview Output File : '{output_path}'")
    print(" Grid Dimension      : 3x3 (9 augmented variants)")
    print(f" Image Resolution    : {grid_img.size[0]}x{grid_img.size[1]} px")
    print("=" * 70)


if __name__ == "__main__":
    main()
