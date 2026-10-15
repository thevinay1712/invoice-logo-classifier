"""Integration tests for end-to-end invoice logo classification decision hierarchy."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from invoice_logo_classifier.inference import (
    CandidateResult,
    ClassificationResult,
    InvoiceLogoClassifierService,
    classify_invoice,
)


@pytest.fixture
def synthetic_invoice_image(tmp_path) -> Path:
    """Fixture generating a synthetic document invoice image with a logo in the header."""
    inv_path = tmp_path / "test_invoice.png"
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw colorful synthetic logo box in top-left header
    draw.rectangle([20, 20, 150, 90], fill=(220, 40, 40), outline=(0, 0, 0))
    draw.text((30, 40), "VENDOR LOGO", fill=(255, 255, 255))

    # Draw invoice text content
    draw.text((50, 300), "INVOICE #9901 - AMOUNT DUE: $1,250.00", fill=(0, 0, 0))
    img.save(inv_path)
    return inv_path


def test_classify_matched_trained_outcome(synthetic_invoice_image):
    """Test Outcome 1: Classifier softmax score clears closed_set_threshold -> status 'matched_trained'."""
    # Set low closed_set_threshold (0.01) to force classifier match
    service = InvoiceLogoClassifierService(
        closed_set_threshold=0.01,
        gallery_threshold=0.99,
    )

    result = service.classify_document(synthetic_invoice_image)

    assert isinstance(result, ClassificationResult)
    assert result.status == "matched_trained"
    assert result.match_source == "classifier"
    assert result.predicted_label is not None
    assert result.crop_image is not None
    assert isinstance(result.candidate_results[0], CandidateResult)


def test_classify_matched_gallery_outcome(synthetic_invoice_image):
    """Test Outcome 2: Classifier passes, but gallery match clears gallery_threshold -> status 'matched_gallery'."""
    service = InvoiceLogoClassifierService(
        closed_set_threshold=0.999,  # Force classifier to pass
        gallery_threshold=0.10,     # Force low gallery threshold to hit
    )

    # Pre-seed gallery with a synthetic vector matching the candidate image
    crop_img = Image.new("RGB", (100, 100), color=(220, 40, 40))
    crop_tensor = service.eval_transform(crop_img).unsqueeze(0).to(service.device)
    with service.device:
        pass
    import torch
    with torch.no_grad():
        from invoice_logo_classifier.inference.classify import _forward_with_features
        _, feature_vec = _forward_with_features(service.model, crop_tensor)

    from invoice_logo_classifier.gallery import VendorGallery
    service.gallery = VendorGallery(embedding_dim=len(feature_vec))
    service.gallery.add(
        vendor_label="Preseeded Vendor Inc",
        embedding=feature_vec,
        source_ref="manual_review_001.pdf",
    )

    result = service.classify_document(synthetic_invoice_image)

    assert result.status == "matched_gallery"
    assert result.match_source == "gallery"
    assert result.predicted_label == "Preseeded Vendor Inc"


def test_classify_unknown_outcome(synthetic_invoice_image):
    """Test Outcome 3: Neither classifier nor gallery clears their respective threshold -> status 'unknown'."""
    service = InvoiceLogoClassifierService(
        closed_set_threshold=0.999,  # Impossible classifier threshold
        gallery_threshold=0.999,     # Impossible gallery threshold
    )

    result = service.classify_document(synthetic_invoice_image)

    assert result.status == "unknown"
    assert result.match_source == "none"
    assert result.predicted_label is None


def test_module_level_classify_invoice_shortcut(synthetic_invoice_image):
    """Verify module-level classify_invoice helper function executes end-to-end."""
    service = InvoiceLogoClassifierService(closed_set_threshold=0.01)
    result = classify_invoice(synthetic_invoice_image, service=service)

    assert isinstance(result, ClassificationResult)
    assert result.source_path == synthetic_invoice_image
