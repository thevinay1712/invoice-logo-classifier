"""Pytest configuration and shared fixtures for invoice-logo-classifier."""

import pytest

from invoice_logo_classifier.config import Settings, settings


@pytest.fixture
def default_settings() -> Settings:
    """Fixture providing default configuration settings instance."""
    return settings
