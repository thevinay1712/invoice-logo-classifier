"""Image augmentation and evaluation transforms for closing the domain gap between real photos and invoice document logos."""

import io
import random
from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageEnhance
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


class RandomScannerBanding:
    """Simulate flatbed scanner sensor banding and feeder streaks."""

    def __init__(self, num_bands_range: tuple[int, int] = (1, 3), p: float = 0.3):
        """Initialize scanner banding transform.

        Args:
            num_bands_range: Min and max number of sensor lines to simulate.
            p: Probability of applying scanner banding.
        """
        self.num_bands_range = num_bands_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply faint scanner sensor streaks randomly across the image."""
        if random.random() > self.p:
            return img

        result = img.convert("RGB").copy()
        draw = ImageDraw.Draw(result)
        w, h = result.size
        num_bands = random.randint(self.num_bands_range[0], self.num_bands_range[1])

        for _ in range(num_bands):
            is_horizontal = random.random() > 0.3
            shade = random.randint(185, 235)
            line_width = random.randint(1, 2)
            if is_horizontal:
                y = random.randint(0, h - 1)
                draw.line([(0, y), (w, y)], fill=(shade, shade, shade), width=line_width)
            else:
                x = random.randint(0, w - 1)
                draw.line([(x, 0), (x, h)], fill=(shade, shade, shade), width=line_width)

        return result


class RandomLowDPISimulation:
    """Simulate low-resolution document scanning, faxing, and downsampling pixelation."""

    def __init__(self, scale_range: tuple[float, float] = (0.35, 0.75), p: float = 0.3):
        """Initialize low-DPI simulation transform.

        Args:
            scale_range: Min and max downscaling factors to simulate DPI reduction.
            p: Probability of applying low-DPI downsampling.
        """
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Downsample and upsample image to simulate scanner resolution degradation."""
        if random.random() > self.p:
            return img

        w, h = img.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        low_w = max(int(w * scale), 8)
        low_h = max(int(h * scale), 8)
        resample_method = random.choice([Image.Resampling.BILINEAR, Image.Resampling.NEAREST])
        downsampled = img.resize((low_w, low_h), resample=resample_method)
        return downsampled.resize((w, h), resample=Image.Resampling.BILINEAR)


class RandomPhotocopyContrast:
    """Simulate high-contrast photocopy toner ink thresholding."""

    def __init__(self, contrast_factor_range: tuple[float, float] = (1.5, 3.0), p: float = 0.25):
        """Initialize photocopy contrast transform.

        Args:
            contrast_factor_range: Range of contrast boost multipliers.
            p: Probability of applying photocopy contrast.
        """
        self.contrast_factor_range = contrast_factor_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Boost contrast to simulate harsh photocopier toner transfer."""
        if random.random() > self.p:
            return img

        factor = random.uniform(self.contrast_factor_range[0], self.contrast_factor_range[1])
        enhancer = ImageEnhance.Contrast(img.convert("RGB"))
        return enhancer.enhance(factor)


def get_train_transforms(
    image_size: int | tuple[int, int] | None = None,
    grayscale_prob: float = 0.35,
    rotation_degrees: float = 10.0,
    perspective_prob: float = 0.3,
    perspective_distortion: float = 0.15,
    scanner_banding_prob: float = 0.3,
    low_dpi_prob: float = 0.3,
    photocopy_prob: float = 0.25,
    jpeg_prob: float = 0.4,
    jpeg_quality_range: tuple[int, int] = (40, 95),
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> transforms.Compose:
    """Build training transform pipeline with document domain augmentations.

    Args:
        image_size: Target (height, width) or scalar size (default settings.image_size).
        grayscale_prob: Probability of random grayscale conversion (30-40% range).
        rotation_degrees: Max positive/negative rotation degrees (modest range, e.g. 10 deg).
        perspective_prob: Probability of random perspective tilt simulating scanner bed skew.
        perspective_distortion: Distortion scale for perspective transform.
        scanner_banding_prob: Probability of applying scanner sensor banding lines.
        low_dpi_prob: Probability of low-DPI fax/scan downsampling simulation.
        photocopy_prob: Probability of harsh photocopy contrast enhancement.
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

    # Resize to 256 then center-crop to target: better for small logos embedded in headers
    resize_to = (int(target_size[0] * 256 / 224), int(target_size[1] * 256 / 224))

    target_mean = list(mean if mean is not None else settings.normalize_mean)
    target_std = list(std if std is not None else settings.normalize_std)

    return transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.CenterCrop(target_size),
            transforms.RandomPerspective(distortion_scale=perspective_distortion, p=perspective_prob, fill=255),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=rotation_degrees, fill=255),
            RandomScannerBanding(p=scanner_banding_prob),
            RandomLowDPISimulation(p=low_dpi_prob),
            RandomPhotocopyContrast(p=photocopy_prob),
            transforms.RandomGrayscale(p=grayscale_prob),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
            RandomJPEGCompression(quality_range=jpeg_quality_range, p=jpeg_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=target_mean, std=target_std),
            # RandomErasing simulates partial logo occlusion from overlapping header text
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=1.0),
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

    # Resize to 256 then center-crop to target for consistent eval preprocessing
    resize_to = (int(target_size[0] * 256 / 224), int(target_size[1] * 256 / 224))

    target_mean = list(mean if mean is not None else settings.normalize_mean)
    target_std = list(std if std is not None else settings.normalize_std)

    return transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=target_mean, std=target_std),
        ]
    )
