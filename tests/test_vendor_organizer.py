"""Unit tests for BulkInvoiceVendorClassifier and organize_bulk_invoices."""

import json
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

from invoice_logo_classifier.inference.classify import InvoiceLogoClassifierService
from invoice_logo_classifier.inference.vendor_organizer import (
    BulkInvoiceVendorClassifier,
    organize_bulk_invoices,
)


@pytest.fixture
def sample_bulk_invoice_dir(tmp_path) -> tuple[Path, Path]:
    """Create sample input directory with synthetic invoice files and target output directory."""
    input_dir = tmp_path / "bulk_input"
    output_dir = tmp_path / "classified_output"
    input_dir.mkdir()

    # Create 3 synthetic invoice images
    for i in range(1, 4):
        img_path = input_dir / f"invoice_doc_{i:03d}.png"
        img = Image.new("RGB", (600, 800), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 150, 90], fill=(50 * i, 100, 200))
        draw.text((30, 40), f"VENDOR LOGO {i}", fill=(255, 255, 255))
        draw.text((50, 300), f"INVOICE #{i} - AMOUNT DUE: ${100 * i}.00", fill=(0, 0, 0))
        img.save(img_path)

    return input_dir, output_dir


def test_bulk_invoice_vendor_classifier_run(sample_bulk_invoice_dir):
    """Test bulk vendor classification and file sorting into vendor folders."""
    input_dir, output_dir = sample_bulk_invoice_dir

    service = InvoiceLogoClassifierService(
        closed_set_threshold=0.01,  # force match to trained class
        gallery_threshold=0.99,
    )

    classifier = BulkInvoiceVendorClassifier(classifier_service=service, copy_mode=True)
    summary = classifier.process_directory(input_dir=input_dir, output_dir=output_dir)

    assert summary.total_processed == 3
    assert output_dir.exists()
    assert (output_dir / "vendor_classification_report.json").exists()

    # Check report content
    with open(output_dir / "vendor_classification_report.json", encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_processed"] == 3
        assert len(data["processed_files"]) == 3


def test_organize_bulk_invoices_shortcut(sample_bulk_invoice_dir):
    """Test module-level organize_bulk_invoices shortcut function."""
    input_dir, output_dir = sample_bulk_invoice_dir

    service = InvoiceLogoClassifierService(closed_set_threshold=0.01)
    summary = organize_bulk_invoices(
        input_dir=input_dir,
        output_dir=output_dir,
        copy_mode=True,
        service=service,
    )

    assert summary.total_processed == 3
    assert len(summary.processed_files) == 3
