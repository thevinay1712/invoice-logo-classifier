"""Unit tests for offline review queue storage, resolution workflows, and live gallery feedback loop."""

import numpy as np
import pytest
from PIL import Image

from invoice_logo_classifier.gallery import VendorGallery
from invoice_logo_classifier.review import (
    ReviewItem,
    ReviewQueueManager,
)


class DummyExtractor:
    """Mock extractor returning a fixed unit vector of specified dimension."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def extract(self, img: Image.Image) -> np.ndarray:
        v = np.ones((self.dim,), dtype=np.float32)
        return v / np.linalg.norm(v)


@pytest.fixture
def review_mgr(tmp_path) -> ReviewQueueManager:
    """Fixture providing a fresh ReviewQueueManager instance operating in a temporary directory."""
    queue_dir = tmp_path / "review_queue_test"
    return ReviewQueueManager(queue_dir=queue_dir)


@pytest.fixture
def test_gallery(tmp_path) -> VendorGallery:
    """Fixture providing a fresh VendorGallery vector store instance."""
    gallery = VendorGallery(embedding_dim=4)
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    gallery.add("Existing Vendor", v1)
    return gallery


@pytest.fixture
def synthetic_crop_image() -> Image.Image:
    """Fixture providing a 100x100 RGB synthetic logo crop image."""
    return Image.new("RGB", (100, 100), color=(50, 150, 220))


def test_enqueue_creates_pending_review_item(review_mgr, synthetic_crop_image):
    """Verify enqueuing an unknown invoice crop creates SQLite row, saves PNG file, and lists as pending."""
    item = review_mgr.enqueue(
        crop_image=synthetic_crop_image,
        confidence_or_similarity=0.45,
        invoice_source_ref="invoice_unseen_001.pdf",
        predicted_label="class_A",
    )

    assert isinstance(item, ReviewItem)
    assert item.status == "pending"
    assert item.invoice_source_ref == "invoice_unseen_001.pdf"
    assert item.crop_image_path.exists()

    pending_items = review_mgr.list_pending()
    assert len(pending_items) == 1
    assert pending_items[0].id == item.id


def test_resolve_new_vendor_path_updates_live_gallery(review_mgr, test_gallery, synthetic_crop_image):
    """Verify resolving item via 'new_vendor' path immediately expands the live gallery vector index."""
    item = review_mgr.enqueue(
        crop_image=synthetic_crop_image,
        confidence_or_similarity=0.40,
        invoice_source_ref="invoice_new_vendor.pdf",
    )

    initial_gallery_count = test_gallery.total_entries

    resolved_item = review_mgr.resolve(
        review_item_id=item.id,
        label="Brand Delta Inc",
        resolution_type="new_vendor",
        gallery=test_gallery,
        extractor=DummyExtractor(dim=4),
    )

    assert resolved_item.status == "resolved"
    assert resolved_item.resolved_label == "Brand Delta Inc"
    assert resolved_item.resolution_type == "new_vendor"

    # Live gallery vector count must increase by 1
    assert test_gallery.total_entries == initial_gallery_count + 1

    # Verify vendor search in live gallery
    summary = test_gallery.get_vendor_summary()
    assert "Brand Delta Inc" in summary
    assert summary["Brand Delta Inc"] == 1

    # Confirms pending list no longer includes resolved item
    assert len(review_mgr.list_pending()) == 0


def test_resolve_classifier_missed_trained_class_logs_csv_and_keeps_gallery_unchanged(
    review_mgr, test_gallery, synthetic_crop_image
):
    """Verify resolving via 'classifier_missed_trained_class' logs to CSV and DOES NOT touch the live gallery."""
    item = review_mgr.enqueue(
        crop_image=synthetic_crop_image,
        confidence_or_similarity=0.60,
        invoice_source_ref="invoice_missed_class.pdf",
        predicted_label="class_B",
    )

    initial_gallery_count = test_gallery.total_entries

    resolved_item = review_mgr.resolve(
        review_item_id=item.id,
        label="class_A",
        resolution_type="classifier_missed_trained_class",
        gallery=test_gallery,
        extractor=DummyExtractor(dim=4),
    )

    assert resolved_item.status == "resolved"
    assert resolved_item.resolved_label == "class_A"
    assert resolved_item.resolution_type == "classifier_missed_trained_class"

    # Gallery count MUST NOT change
    assert test_gallery.total_entries == initial_gallery_count
    assert "class_A" not in test_gallery.get_vendor_summary()

    # CSV file must exist and contain the recorded miss
    csv_file = review_mgr.misses_csv_path
    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "class_A" in content
    assert "invoice_missed_class.pdf" in content
