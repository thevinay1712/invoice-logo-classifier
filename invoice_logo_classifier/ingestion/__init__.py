"""Ingestion module: converts PDF, DOCX, and image files to standard page images."""

from invoice_logo_classifier.ingestion.ingest import (
    IngestedDocument,
    IngestionError,
    ingest,
)

__all__ = [
    "IngestedDocument",
    "IngestionError",
    "ingest",
]
