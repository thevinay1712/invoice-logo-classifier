"""Tests for Settings and configuration management."""

from pathlib import Path

from invoice_logo_classifier.config import Settings, settings


def test_default_settings():
    """Verify default setting values and types."""
    assert isinstance(settings, Settings)
    assert settings.pdf_dpi == 300
    assert settings.image_size == (224, 224)
    assert settings.detection_confidence_threshold == 0.50
    assert settings.closed_set_confidence_threshold == 0.45
    assert settings.label_column == "company_name"
    assert settings.image_column == "image_path"
    assert settings.bbox_column == "bbox"
    assert settings.category_column == "industry_name"
    assert settings.batch_size == 32
    assert settings.learning_rate == 1e-3


def test_settings_paths_exist_or_derivable():
    """Verify base directories are Path instances."""
    assert isinstance(settings.base_dir, Path)
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.model_dir, Path)
    assert isinstance(settings.gallery_dir, Path)
    assert isinstance(settings.review_queue_dir, Path)


def test_settings_env_override(monkeypatch):
    """Verify settings can be overridden via environment variables."""
    monkeypatch.setenv("ILC_PDF_DPI", "150")
    monkeypatch.setenv("ILC_BATCH_SIZE", "64")
    custom_settings = Settings()
    assert custom_settings.pdf_dpi == 150
    assert custom_settings.batch_size == 64
