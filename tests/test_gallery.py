"""Unit tests for gallery store nearest-neighbor search, threshold gating, and persistence."""

import numpy as np
import pytest
from PIL import Image

from invoice_logo_classifier.gallery import (
    GalleryMatch,
    LogoEmbeddingExtractor,
    VendorGallery,
)


@pytest.fixture
def synthetic_gallery() -> VendorGallery:
    """Fixture providing a VendorGallery instance loaded with synthetic orthogonal vectors."""
    gallery = VendorGallery(embedding_dim=4)

    # Vendor 1: vector [1, 0, 0, 0]
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    gallery.add("Acme Corp", v1, source_ref="invoice_001.pdf")

    # Vendor 1 (alternate template): vector [0.9, 0.1, 0.0, 0.0] normalized
    v1_alt = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    gallery.add("Acme Corp", v1_alt, source_ref="invoice_002.pdf")

    # Vendor 2: vector [0, 1, 0, 0]
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    gallery.add("Beta Logistics", v2, source_ref="invoice_003.pdf")

    return gallery


def test_gallery_add_and_search_nearest_neighbor(synthetic_gallery):
    """Verify search returns the correct nearest neighbor vendor for a query vector."""
    query = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
    matches = synthetic_gallery.search(query, k=2)

    assert len(matches) == 2
    assert isinstance(matches[0], GalleryMatch)
    assert matches[0].vendor_label == "Acme Corp"
    assert matches[0].similarity_score > 0.90

    summary = synthetic_gallery.get_vendor_summary()
    assert summary["Acme Corp"] == 2
    assert summary["Beta Logistics"] == 1


def test_gallery_best_match_threshold_gating(synthetic_gallery):
    """Verify best_match returns top match above threshold and None below threshold."""
    # Query vector close to Acme Corp
    query_close = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    match = synthetic_gallery.best_match(query_close, threshold=0.80)
    assert match is not None
    assert match.vendor_label == "Acme Corp"

    # Query vector orthogonal to all (e.g. [0, 0, 0, 1]) -> similarity ~0.0
    query_unseen = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    no_match = synthetic_gallery.best_match(query_unseen, threshold=0.75)
    assert no_match is None


def test_gallery_save_and_load_roundtrip(tmp_path, synthetic_gallery):
    """Verify VendorGallery save and load methods accurately persist vectors and SQLite metadata."""
    save_dir = tmp_path / "test_gallery_save"
    synthetic_gallery.save(save_dir)

    loaded_gallery = VendorGallery.load(save_dir)

    assert loaded_gallery.total_entries == synthetic_gallery.total_entries
    assert loaded_gallery.embedding_dim == synthetic_gallery.embedding_dim

    # Verify search on loaded gallery
    query = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    match = loaded_gallery.best_match(query, threshold=0.80)

    assert match is not None
    assert match.vendor_label == "Beta Logistics"


def test_logo_embedding_extractor_output():
    """Verify LogoEmbeddingExtractor extracts 1D L2-normalized float32 vectors."""
    extractor = LogoEmbeddingExtractor(use_clip_embedding=False)
    img = Image.new("RGB", (100, 100), color=(150, 100, 200))

    vec = extractor.extract(img)

    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.ndim == 1
    # Check L2 normalization (norm == 1.0)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)
