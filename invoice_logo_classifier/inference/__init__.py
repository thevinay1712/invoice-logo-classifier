"""Inference module: end-to-end classify() workflow combining ingestion, detection, classifier, and gallery fallback."""

from invoice_logo_classifier.inference.classify import (
    CandidateResult,
    ClassificationResult,
    InvoiceLogoClassifierService,
    classify_invoice,
    get_classifier_service,
)
from invoice_logo_classifier.inference.vendor_organizer import (
    BulkInvoiceVendorClassifier,
    VendorBatchSummary,
    organize_bulk_invoices,
)

__all__ = [
    "BulkInvoiceVendorClassifier",
    "CandidateResult",
    "ClassificationResult",
    "InvoiceLogoClassifierService",
    "VendorBatchSummary",
    "classify_invoice",
    "get_classifier_service",
    "organize_bulk_invoices",
]

