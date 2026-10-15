"""Multi-format document ingestion engine for PDF, DOCX, and image formats."""

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pypdf
from PIL import Image, ImageDraw, ImageOps

from invoice_logo_classifier.config import settings

logger = logging.getLogger(__name__)


class IngestionError(ValueError):
    """Custom exception raised when document ingestion fails due to unreadable, corrupt, or unsupported files."""



@dataclass
class IngestedDocument:
    """Normalized document container storing RGB page images and metadata."""

    source_path: Path
    source_format: str
    pages: list[Image.Image]
    page_count: int
    warnings: list[str] = field(default_factory=list)


def _ingest_image(path: Path) -> tuple[list[Image.Image], int, list[str]]:
    """Load and auto-orient single image document."""
    warnings = []
    try:
        with Image.open(path) as raw_img:
            # Auto-orient using EXIF tag
            oriented = ImageOps.exif_transpose(raw_img)
            rgb_img = oriented.convert("RGB")
            # Force load image data into memory before returning
            rgb_img.load()
            return [rgb_img], 1, warnings
    except Exception as err:  # noqa: BLE001 – any PIL/IO failure on corrupt image should be an IngestionError
        raise IngestionError(f"Corrupt or unreadable image file '{path}': {err}")


def _ingest_pdf(
    path: Path, max_pages: int = 1, dpi: int = 200
) -> tuple[list[Image.Image], int, list[str]]:
    """Rasterize PDF page documents into RGB images."""
    warnings = []

    # Verify PDF readability and count total pages using pypdf
    try:
        reader = pypdf.PdfReader(str(path))
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise IngestionError(f"PDF file '{path}' contains zero pages.")
    except IngestionError:
        raise
    except Exception as err:  # noqa: BLE001 – pypdf raises many exception types on corrupt PDFs
        raise IngestionError(f"Corrupt or unreadable PDF document '{path}': {err}")

    rendered_pages = []

    # Try pdf2image (poppler backend)
    try:
        import pdf2image

        images = pdf2image.convert_from_path(
            str(path),
            dpi=dpi,
            first_page=1,
            last_page=min(max_pages, total_pages),
        )
        for img in images:
            rgb_img = img.convert("RGB")
            rendered_pages.append(rgb_img)
    except Exception as poppler_err:  # noqa: BLE001 – poppler not installed or conversion fail; fall back
        warning_msg = (
            f"pdf2image/poppler conversion failed or poppler is not installed ({poppler_err}). "
            "Falling back to pypdf image extraction / canvas renderer."
        )
        warnings.append(warning_msg)
        logger.warning(warning_msg)

        # Fallback rendering via pypdf page inspection
        pages_to_render = min(max_pages, total_pages)
        for page_idx in range(pages_to_render):
            page = reader.pages[page_idx]
            page_extracted = False
            # Check for embedded images on the page
            for img_obj in page.images:
                try:
                    pil_img = Image.open(img_obj.data).convert("RGB")
                    rendered_pages.append(pil_img)
                    page_extracted = True
                    break
                except Exception:  # noqa: BLE001, S112 – skip individual embedded image decode failures
                    continue

            if not page_extracted:
                # Synthetic canvas fallback if no embedded image found
                canvas = Image.new("RGB", (600, 800), color=(255, 255, 255))
                draw = ImageDraw.Draw(canvas)
                page_text = page.extract_text() or f"PDF Page {page_idx + 1}"
                draw.text((30, 30), page_text[:200], fill=(0, 0, 0))
                rendered_pages.append(canvas)

    return rendered_pages, total_pages, warnings


def _ingest_docx(
    path: Path, max_pages: int = 1, dpi: int = 200
) -> tuple[list[Image.Image], int, list[str]]:
    """Convert DOCX/DOC documents to PDF via LibreOffice headless, then ingest PDF pages."""
    soffice_cmd = shutil.which("soffice") or shutil.which("libreoffice")

    if not soffice_cmd:
        raise IngestionError(
            f"LibreOffice executable ('soffice') not found on system PATH. "
            f"Cannot convert DOCX file '{path}'. Please install LibreOffice or convert document to PDF/Image."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cmd = [
            soffice_cmd,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(tmp_path),
            str(path),
        ]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if res.returncode != 0:
                raise IngestionError(
                    f"LibreOffice conversion failed for '{path}': {res.stderr or res.stdout}"
                )
        except Exception as err:  # noqa: BLE001 – OSError/subprocess errors wrapped as IngestionError
            raise IngestionError(f"Failed to execute LibreOffice conversion for '{path}': {err}")

        converted_pdf = tmp_path / f"{path.stem}.pdf"
        if not converted_pdf.exists():
            raise IngestionError(
                f"LibreOffice conversion completed but expected PDF output '{converted_pdf}' was not created."
            )

        return _ingest_pdf(converted_pdf, max_pages=max_pages, dpi=dpi)


def ingest(
    file_path: Path | str,
    max_pages: int = 1,
    dpi: int | None = None,
) -> IngestedDocument:
    """Ingest multi-format invoice document (.pdf, .docx, .doc, .png, .jpg, .jpeg, .tiff, .bmp).

    Args:
        file_path: Path to target document file.
        max_pages: Maximum number of pages to render into images (default 1 for headers).
        dpi: DPI resolution for PDF rendering (default settings.pdf_dpi or 200).

    Returns:
        IngestedDocument dataclass holding normalized RGB page images and page counts.

    Raises:
        IngestionError: If file is missing, corrupt, unsupported, or missing conversion tools.
    """
    target_path = Path(file_path).resolve()
    target_dpi = dpi if dpi is not None else settings.pdf_dpi

    if not target_path.exists() or not target_path.is_file():
        raise IngestionError(f"Document file not found: '{target_path}'.")

    source_ext = target_path.suffix.lower()
    supported_exts = {ext.lower() for ext in settings.supported_file_extensions}

    if source_ext not in supported_exts:
        raise IngestionError(
            f"Unsupported file format '{source_ext}' for file '{target_path}'. "
            f"Supported formats: {sorted(supported_exts)}"
        )

    # Route based on extension
    if source_ext in (".pdf",):
        pages, total_pages, warnings = _ingest_pdf(
            target_path, max_pages=max_pages, dpi=target_dpi
        )
    elif source_ext in (".docx", ".doc"):
        pages, total_pages, warnings = _ingest_docx(
            target_path, max_pages=max_pages, dpi=target_dpi
        )
    else:
        # Standard image formats (.png, .jpg, .jpeg, .tiff, .bmp)
        pages, total_pages, warnings = _ingest_image(target_path)

    return IngestedDocument(
        source_path=target_path,
        source_format=source_ext,
        pages=pages,
        page_count=total_pages,
        warnings=warnings,
    )
