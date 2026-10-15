"""Unit tests for multi-format document ingestion engine (PDF, DOCX, PNG, corrupt files)."""

from pathlib import Path

import docx
import pytest
from PIL import Image
from reportlab.pdfgen import canvas

from invoice_logo_classifier.ingestion import (
    IngestedDocument,
    IngestionError,
    ingest,
)


@pytest.fixture
def synthetic_png_file(tmp_path) -> Path:
    """Fixture generating a valid synthetic 100x100 PNG image file."""
    img_path = tmp_path / "test_invoice.png"
    img = Image.new("RGB", (100, 100), color=(200, 50, 50))
    img.save(img_path)
    return img_path


@pytest.fixture
def synthetic_multipage_pdf(tmp_path) -> Path:
    """Fixture generating a 3-page synthetic PDF document using reportlab."""
    pdf_path = tmp_path / "test_multipage.pdf"
    c = canvas.Canvas(str(pdf_path))

    # Page 1 header
    c.drawString(100, 750, "INVOICE #001 - PAGE 1 HEADER")
    c.showPage()

    # Page 2 content
    c.drawString(100, 750, "INVOICE #001 - PAGE 2 DETAILS")
    c.showPage()

    # Page 3 footer
    c.drawString(100, 750, "INVOICE #001 - PAGE 3 SUMMARY")
    c.showPage()

    c.save()
    return pdf_path


@pytest.fixture
def synthetic_docx_file(tmp_path) -> Path:
    """Fixture generating a synthetic .docx invoice document using python-docx."""
    docx_path = tmp_path / "test_invoice.docx"
    doc = docx.Document()
    doc.add_heading("INVOICE VENDOR LOGO TEST", level=1)
    doc.add_paragraph("Invoice Number: INV-99882")
    doc.save(docx_path)
    return docx_path


@pytest.fixture
def corrupted_image_file(tmp_path) -> Path:
    """Fixture generating a corrupted fake image file containing garbage bytes."""
    corrupt_path = tmp_path / "corrupt_invoice.png"
    corrupt_path.write_bytes(b"NOT_A_REAL_IMAGE_HEADER_GARBAGE_BYTES_12345")
    return corrupt_path


def test_ingest_png_image(synthetic_png_file):
    """Verify PNG image ingestion outputs normalized RGB page image with page_count=1."""
    doc = ingest(synthetic_png_file)

    assert isinstance(doc, IngestedDocument)
    assert doc.source_format == ".png"
    assert doc.page_count == 1
    assert len(doc.pages) == 1
    assert isinstance(doc.pages[0], Image.Image)
    assert doc.pages[0].mode == "RGB"


def test_ingest_multipage_pdf_and_max_pages_limit(synthetic_multipage_pdf):
    """Verify multi-page PDF ingestion reports true total page count while respecting max_pages limit."""
    # Test max_pages=1 (default)
    doc1 = ingest(synthetic_multipage_pdf, max_pages=1)
    assert doc1.source_format == ".pdf"
    assert doc1.page_count == 3  # True total page count
    assert len(doc1.pages) == 1  # Only 1 page rendered into images list

    # Test max_pages=2
    doc2 = ingest(synthetic_multipage_pdf, max_pages=2)
    assert doc2.page_count == 3
    assert len(doc2.pages) == 2


def test_ingest_docx_handling_or_missing_libreoffice_error(synthetic_docx_file):
    """Verify DOCX ingestion raises actionable IngestionError if LibreOffice is missing, or converts cleanly."""
    try:
        doc = ingest(synthetic_docx_file)
        assert doc.source_format == ".docx"
        assert doc.page_count >= 1
    except IngestionError as err:
        # LibreOffice missing on PATH is a valid expected actionable error
        assert "LibreOffice" in str(err) or "soffice" in str(err)


def test_ingest_corrupted_file_raises_ingestion_error(corrupted_image_file):
    """Verify corrupted files raise human-readable IngestionError instead of bare unhandled exceptions."""
    with pytest.raises(IngestionError) as exc_info:
        ingest(corrupted_image_file)

    assert "Corrupt or unreadable" in str(exc_info.value)


def test_ingest_unsupported_file_extension(tmp_path):
    """Verify unsupported file formats raise explicit IngestionError."""
    unsupported = tmp_path / "test.xyz"
    unsupported.write_text("dummy text")

    with pytest.raises(IngestionError) as exc_info:
        ingest(unsupported)

    assert "Unsupported file format" in str(exc_info.value)
