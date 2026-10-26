"""Seed initial vendor embeddings in the FAISS Vector Gallery store."""

import sys
from pathlib import Path
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from invoice_logo_classifier.augmentation import get_eval_transforms
from invoice_logo_classifier.config import settings
from invoice_logo_classifier.gallery import VendorGallery
from invoice_logo_classifier.inference.classify import _forward_with_features
from invoice_logo_classifier.training.models import build_model


def create_gallery_logo(name: str, color: tuple[int, int, int], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 290, 90], fill=color)
    draw.text((30, 40), f"LOGO: {name.upper()}", fill=(255, 255, 255))
    img.save(path)
    return img


def main():
    print("=" * 70)
    print(" SEEDING DYNAMIC VENDOR GALLERY (FAISS + SQLITE)")
    print("=" * 70)

    model_path = settings.model_dir / "best_model.pt"
    if not model_path.exists():
        print(f"[ERROR] Trained model checkpoint not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cpu")
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    class_list = checkpoint.get("class_list", [])
    backbone_name = checkpoint.get("backbone_name", "resnet50")

    model = build_model(backbone_name=backbone_name, num_classes=len(class_list), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_transform = get_eval_transforms(image_size=settings.image_size)

    gallery_dir = settings.gallery_dir
    gallery_dir.mkdir(parents=True, exist_ok=True)

    gallery = VendorGallery(embedding_dim=2048)

    temp_dir = settings.base_dir / "data" / "gallery_seeds"
    temp_dir.mkdir(parents=True, exist_ok=True)

    seed_vendors = [
        {"name": "AcmeCorp", "color": (30, 60, 120), "samples": 3},
        {"name": "GlobalLogistics", "color": (16, 185, 129), "samples": 2},
        {"name": "FastTrackShipping", "color": (245, 158, 11), "samples": 2},
        {"name": "ApexLogistics", "color": (139, 92, 246), "samples": 2},
    ]

    for v in seed_vendors:
        for s in range(1, v["samples"] + 1):
            seed_img_path = temp_dir / f"{v['name']}_{s}.png"
            img = create_gallery_logo(v["name"], v["color"], seed_img_path)
            tensor = eval_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                _, feat = _forward_with_features(model, tensor)
            gallery.add(vendor_label=v["name"], embedding=feat, source_ref=seed_img_path.name)

    gallery.save(gallery_dir)
    print(f"\nGallery successfully seeded with {gallery.total_entries} vector embeddings in '{gallery_dir}'.")
    summary = gallery.get_vendor_summary()
    for vendor, count in summary.items():
        print(f"  • {vendor:<25}: {count} embeddings")


if __name__ == "__main__":
    main()
