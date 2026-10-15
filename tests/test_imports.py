"""Tests verifying clean imports across all invoice-logo-classifier subpackages."""

import importlib

import pytest


@pytest.mark.parametrize(
    "package_name",
    [
        "invoice_logo_classifier",
        "invoice_logo_classifier.config",
        "invoice_logo_classifier.data",
        "invoice_logo_classifier.augmentation",
        "invoice_logo_classifier.training",
        "invoice_logo_classifier.evaluation",
        "invoice_logo_classifier.ingestion",
        "invoice_logo_classifier.detection",
        "invoice_logo_classifier.gallery",
        "invoice_logo_classifier.inference",
        "invoice_logo_classifier.review",
        "invoice_logo_classifier.api",
    ],
)
def test_subpackage_imports(package_name: str):
    """Verify that every module/subpackage can be imported without error."""
    mod = importlib.import_module(package_name)
    assert mod is not None
