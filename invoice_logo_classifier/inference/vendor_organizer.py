"""Bulk invoice vendor classification and document organization module.

Ingests bulk unorganized invoices, classifies each invoice by vendor using logo ML & gallery matching,
and sorts physical documents into vendor-specific directories with a unified summary report.
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from invoice_logo_classifier.inference.classify import (
    ClassificationResult,
    InvoiceLogoClassifierService,
    get_classifier_service,
)

logger = logging.getLogger(__name__)


@dataclass
class VendorBatchSummary:
    """Consolidated summary of bulk invoice vendor classification pass."""

    total_processed: int = 0
    matched_trained_count: int = 0
    matched_gallery_count: int = 0
    unassigned_count: int = 0
    vendors_found: list[str] = field(default_factory=list)
    vendor_breakdown: dict[str, int] = field(default_factory=dict)
    processed_files: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert summary container to standard dictionary."""
        return asdict(self)


class BulkInvoiceVendorClassifier:
    """Engine for classifying bulk invoices by vendor and sorting them into vendor directories."""

    def __init__(
        self,
        classifier_service: InvoiceLogoClassifierService | None = None,
        copy_mode: bool = True,
    ):
        """Initialize bulk vendor classifier.

        Args:
            classifier_service: Invoice logo classification service instance.
            copy_mode: If True, copies invoices into vendor folders. If False, moves them.
        """
        self.service = classifier_service or get_classifier_service()
        self.copy_mode = copy_mode

    def process_directory(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        supported_extensions: tuple[str, ...] = (".pdf", ".png", ".jpg", ".jpeg", ".docx"),
    ) -> VendorBatchSummary:
        """Process all invoice files in input_dir, classify vendor for each, and sort into output_dir.

        Directory Structure Created:
            output_dir/
            ├── <Vendor_A>/
            │   ├── invoice_001.pdf
            ├── <Vendor_B>/
            │   ├── invoice_002.png
            ├── Unassigned_ReviewQueue/
            │   ├── invoice_003.pdf
            └── vendor_classification_report.json

        Args:
            input_dir: Directory containing bulk unclassified invoice documents.
            output_dir: Target root directory for sorted vendor folders.
            supported_extensions: File extensions to process.

        Returns:
            VendorBatchSummary detailing classification results across all files.
        """
        in_path = Path(input_dir)
        out_path = Path(output_dir)

        if not in_path.exists() or not in_path.is_dir():
            raise ValueError(f"Input directory '{input_dir}' does not exist or is not a directory.")

        out_path.mkdir(parents=True, exist_ok=True)

        summary = VendorBatchSummary()
        vendor_counts: dict[str, int] = {}

        # Collect target invoice files
        files_to_process = [
            f for f in sorted(in_path.glob("*"))
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        logger.info(f"Starting bulk vendor classification for {len(files_to_process)} invoices in '{in_path}'...")

        for file_path in files_to_process:
            try:
                result: ClassificationResult = self.service.classify_document(file_path)
                vendor_label = result.predicted_label or "Unassigned_ReviewQueue"

                # Sanitize vendor folder name
                safe_folder_name = "".join(
                    c if c.isalnum() or c in (" ", "_", "-") else "_" for c in vendor_label
                ).strip().replace(" ", "_")

                target_folder = out_path / safe_folder_name
                target_folder.mkdir(parents=True, exist_ok=True)

                dest_file_path = target_folder / file_path.name

                if self.copy_mode:
                    shutil.copy2(file_path, dest_file_path)
                else:
                    shutil.move(file_path, dest_file_path)

                # Update stats
                summary.total_processed += 1
                if result.status == "matched_trained":
                    summary.matched_trained_count += 1
                elif result.status == "matched_gallery":
                    summary.matched_gallery_count += 1
                else:
                    summary.unassigned_count += 1

                vendor_counts[vendor_label] = vendor_counts.get(vendor_label, 0) + 1

                summary.processed_files.append(
                    {
                        "original_filename": file_path.name,
                        "classified_vendor": vendor_label,
                        "status": result.status,
                        "confidence_score": result.confidence_or_similarity,
                        "match_source": result.match_source,
                        "destination_path": str(dest_file_path),
                    }
                )

            except Exception as err:  # noqa: BLE001
                logger.error(f"Failed to classify invoice file '{file_path.name}': {err}")
                summary.processed_files.append(
                    {
                        "original_filename": file_path.name,
                        "classified_vendor": "Error_FailedProcessing",
                        "status": "error",
                        "confidence_score": 0.0,
                        "match_source": "error",
                        "error_message": str(err),
                    }
                )

        summary.vendor_breakdown = vendor_counts
        summary.vendors_found = sorted([v for v in vendor_counts if v != "Unassigned_ReviewQueue"])

        # Write unified report JSON
        report_path = out_path / "vendor_classification_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

        logger.info(
            f"Bulk vendor classification complete. Processed {summary.total_processed} files. "
            f"Report saved to '{report_path}'."
        )

        return summary


def organize_bulk_invoices(
    input_dir: Path | str,
    output_dir: Path | str,
    copy_mode: bool = True,
    service: InvoiceLogoClassifierService | None = None,
) -> VendorBatchSummary:
    """Module-level entry point function for bulk invoice vendor classification.

    Args:
        input_dir: Input directory containing bulk invoice files.
        output_dir: Output directory for vendor-sorted folders.
        copy_mode: If True, copies files. If False, moves files.
        service: Optional InvoiceLogoClassifierService instance.

    Returns:
        VendorBatchSummary container.
    """
    classifier = BulkInvoiceVendorClassifier(classifier_service=service, copy_mode=copy_mode)
    return classifier.process_directory(input_dir=input_dir, output_dir=output_dir)
