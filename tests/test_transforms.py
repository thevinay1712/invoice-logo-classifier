"""Unit tests for training and evaluation image transform pipelines."""

import pytest
import torch
from PIL import Image

from invoice_logo_classifier.augmentation import (
    RandomJPEGCompression,
    get_eval_transforms,
    get_train_transforms,
)


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Fixture returning a colorful 100x100 synthetic PIL image."""
    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    for x in range(30, 70):
        for y in range(30, 70):
            img.putpixel((x, y), (250, 50, 50))
    return img


def test_eval_transforms_deterministic(sample_pil_image):
    """Verify eval transforms are strictly deterministic across multiple invocations."""
    eval_transform = get_eval_transforms(image_size=224)

    t1 = eval_transform(sample_pil_image.copy())
    t2 = eval_transform(sample_pil_image.copy())
    t3 = eval_transform(sample_pil_image.copy())

    assert isinstance(t1, torch.Tensor)
    assert t1.shape == (3, 224, 224)

    # Output tensors must be bitwise identical
    assert torch.equal(t1, t2)
    assert torch.equal(t2, t3)


def test_train_transforms_randomness(sample_pil_image):
    """Verify training transforms produce different outputs across repeated calls due to stochastic augmentations."""
    train_transform = get_train_transforms(image_size=224)

    outputs = [train_transform(sample_pil_image.copy()) for _ in range(10)]

    for t in outputs:
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3, 224, 224)

    # Confirm not all outputs are identical across 10 random runs
    all_identical = all(torch.equal(outputs[0], outputs[i]) for i in range(1, 10))
    assert not all_identical, "Train transforms failed to produce stochastic variations across 10 runs."


def test_random_jpeg_compression(sample_pil_image):
    """Verify RandomJPEGCompression handles PIL images cleanly and applies compression without crashing."""
    jpeg_transform = RandomJPEGCompression(quality_range=(20, 50), p=1.0)
    compressed = jpeg_transform(sample_pil_image)

    assert isinstance(compressed, Image.Image)
    assert compressed.size == sample_pil_image.size
    assert compressed.mode == "RGB"
