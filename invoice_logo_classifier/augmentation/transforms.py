"""Image augmentation and evaluation transforms for closing the domain gap between real photos and invoice document logos."""

import io
import random
from collections.abc import Sequence

from PIL import Image
from torchvision import transforms

from invoice_logo_classifier.config import settings


class RandomJPEGCompression:
    """Simulate low-quality JPEG compression artifacts commonly found in scanned invoice documents."""

    def __init__(self, quality_range: tuple[int, int] = (30, 85), p: float = 0.4):
        """Initialize JPEG compression transform.

        Args:
            quality_range: Min and max JPEG quality factor (1-100).
            p: Probability of applying JPEG compression.
        """
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply JPEG compression randomly."""
        if random.random() > self.p:
            return img

        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = io.BytesIO()
        # Ensure image is in RGB format before saving to JPEG
        rgb_img = img.convert("RGB") if img.mode != "RGB" else img
        rgb_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).copy()


def get_train_transforms(
    image_size: int | tuple[int, int] | None = None,
    grayscale_prob: float = 0.35,
    rotation_degrees: float = 10.0,
    jpeg_prob: float = 0.4,
    jpeg_quality_range: tuple[int, int] = (30, 85),
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> transforms.Compose:
    """Build training transform pipeline with document domain augmentations.

    Args:
        image_size: Target (height, width) or scalar size (default settings.image_size).
        grayscale_prob: Probability of random grayscale conversion (30-40% range).
        rotation_degrees: Max positive/negative rotation degrees (modest range, e.g. 10 deg).
        jpeg_prob: Probability of applying JPEG artifact simulation.
        jpeg_quality_range: (min, max) quality factors for JPEG compression.
        mean: Normalization mean (default settings.normalize_mean).
        std: Normalization std (default settings.normalize_std).

    Returns:
        torchvision.transforms.Compose pipeline for training images.
    """
    if image_size is None:
        target_size = settings.image_size
    elif isinstance(image_size, int):
        target_size = (image_size, image_size)
    else:
        target_size = image_size

    target_mean = list(mean if mean is not None else settings.normalize_mean)
    target_std = list(std if std is not None else settings.normalize_std)

    return transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.RandomRotation(degrees=rotation_degrees, fill=255),
            transforms.RandomGrayscale(p=grayscale_prob),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            RandomJPEGCompression(quality_range=jpeg_quality_range, p=jpeg_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=target_mean, std=target_std),
        ]
    )


def get_eval_transforms(
    image_size: int | tuple[int, int] | None = None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> transforms.Compose:
    """Build deterministic validation and test transform pipeline.

    Args:
        image_size: Target (height, width) or scalar size (default settings.image_size).
        mean: Normalization mean (default settings.normalize_mean).
        std: Normalization std (default settings.normalize_std).

    Returns:
        torchvision.transforms.Compose pipeline for evaluation images.
    """
    if image_size is None:
        target_size = settings.image_size
    elif isinstance(image_size, int):
        target_size = (image_size, image_size)
    else:
        target_size = image_size

    target_mean = list(mean if mean is not None else settings.normalize_mean)
    target_std = list(std if std is not None else settings.normalize_std)

    return transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=target_mean, std=target_std),
        ]
    )
