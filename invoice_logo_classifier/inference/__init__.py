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
    "CandidateResult",
    "ClassificationResult",
    "InvoiceLogoClassifierService",
    "classify_invoice",
    "get_classifier_service",
    "BulkInvoiceVendorClassifier",
    "VendorBatchSummary",
    "organize_bulk_invoices",
]

