"""SQLite-backed offline review queue and live gallery feedback loop module."""

import csv
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from invoice_logo_classifier.config import settings
from invoice_logo_classifier.gallery import LogoEmbeddingExtractor, VendorGallery

logger = logging.getLogger(__name__)


@dataclass
class ReviewItem:
    """Review item container representing an unassigned/unknown invoice logo."""

    id: int
    invoice_source_ref: str
    predicted_label: str | None
    confidence_or_similarity: float
    crop_image_path: Path
    status: str  # "pending" | "resolved"
    resolved_label: str | None
    resolution_type: str | None
    created_at: str
    resolved_at: str | None


class ReviewQueueManager:
    """Manages SQLite review queue storage, crop persistence, and resolution workflows."""

    def __init__(self, queue_dir: Path | str | None = None):
        """Initialize review queue manager.

        Args:
            queue_dir: Directory path for database, crops, and logs.
        """
        self.queue_dir = Path(queue_dir if queue_dir is not None else settings.review_queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir = self.queue_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.queue_dir / "review_queue.db"
        self.misses_csv_path = self.queue_dir / "classifier_misses.csv"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database table schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_source_ref TEXT NOT NULL,
                predicted_label TEXT,
                confidence_or_similarity REAL NOT NULL,
                crop_image_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                resolved_label TEXT,
                resolution_type TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def enqueue(
        self,
        crop_image: Image.Image,
        confidence_or_similarity: float,
        invoice_source_ref: str = "",
        predicted_label: str | None = None,
    ) -> ReviewItem:
        """Enqueue an unknown invoice logo crop into the review queue.

        Args:
            crop_image: Cropped logo PIL Image.
            confidence_or_similarity: Score from inference pass.
            invoice_source_ref: Path or reference to source invoice file.
            predicted_label: Low-confidence predicted label (or None).

        Returns:
            Created ReviewItem dataclass object.
        """
        created_at = datetime.now(tz=UTC).isoformat()
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO review_queue (invoice_source_ref, predicted_label, confidence_or_similarity, crop_image_path, status, created_at)
            VALUES (?, ?, ?, ?, 'pending', ?)
            """,
            (invoice_source_ref, predicted_label, float(confidence_or_similarity), "", created_at),
        )
        item_id = cursor.lastrowid
        conn.commit()

        # Save crop image PNG file
        crop_filename = f"crop_{item_id:06d}.png"
        crop_save_path = self.crops_dir / crop_filename
        crop_image.convert("RGB").save(crop_save_path, format="PNG")

        cursor.execute(
            "UPDATE review_queue SET crop_image_path = ? WHERE id = ?",
            (str(crop_save_path), item_id),
        )
        conn.commit()
        conn.close()

        logger.info(f"Enqueued unknown invoice logo for review (Item #{item_id}, ref='{invoice_source_ref}').")

        return ReviewItem(
            id=item_id,
            invoice_source_ref=invoice_source_ref,
            predicted_label=predicted_label,
            confidence_or_similarity=float(confidence_or_similarity),
            crop_image_path=crop_save_path,
            status="pending",
            resolved_label=None,
            resolution_type=None,
            created_at=created_at,
            resolved_at=None,
        )

    def resolve(
        self,
        review_item_id: int,
        label: str,
        resolution_type: str = "new_vendor",
        gallery: VendorGallery | None = None,
        extractor: LogoEmbeddingExtractor | None = None,
    ) -> ReviewItem:
        """Resolve a pending review queue item by human feedback.

        Resolution Types:
            - 'new_vendor': Adds item's embedding directly to the live VendorGallery so future occurrences are matched.
            - 'classifier_missed_trained_class': Logs miss to classifier_misses.csv for Phase 16 retraining (does NOT add to gallery).

        Args:
            review_item_id: SQLite primary key ID of review item.
            label: Human-verified correct vendor label string.
            resolution_type: 'new_vendor' or 'classifier_missed_trained_class'.
            gallery: Optional VendorGallery instance (loads settings.gallery_dir if None).
            extractor: Optional LogoEmbeddingExtractor instance.

        Returns:
            Updated ReviewItem object.
        """
        valid_types = ("new_vendor", "classifier_missed_trained_class")
        if resolution_type not in valid_types:
            raise ValueError(
                f"Invalid resolution_type '{resolution_type}'. Must be one of {valid_types}."
            )

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, invoice_source_ref, predicted_label, confidence_or_similarity, crop_image_path, status, created_at FROM review_queue WHERE id = ?",
            (review_item_id,),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise KeyError(f"Review item #{review_item_id} not found in database.")

        item_id, source_ref, pred_label, conf, crop_path_str, _, created_at = row
        resolved_at = datetime.now(tz=UTC).isoformat()
        crop_path = Path(crop_path_str)

        if not crop_path.exists():
            conn.close()
            raise FileNotFoundError(f"Crop image file not found at '{crop_path}'.")

        crop_img = Image.open(crop_path).convert("RGB")

        if resolution_type == "new_vendor":
            # Immediately update the live gallery store
            if gallery is not None:
                target_gallery = gallery
            else:
                g_dir = settings.gallery_dir
                if g_dir.exists() and (g_dir / "gallery_index.faiss").exists():
                    target_gallery = VendorGallery.load(g_dir)
                else:
                    target_gallery = VendorGallery(embedding_dim=settings.embedding_dim)

            target_extractor = extractor or LogoEmbeddingExtractor()
            vec = target_extractor.extract(crop_img)
            target_gallery.add(vendor_label=label, embedding=vec, source_ref=source_ref)
            target_gallery.save(settings.gallery_dir)
            logger.info(f"Updated live gallery with new vendor '{label}' from review item #{item_id}.")

        elif resolution_type == "classifier_missed_trained_class":
            # Log classifier miss to CSV for Phase 16 retraining without modifying the gallery
            csv_exists = self.misses_csv_path.exists()
            with open(self.misses_csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not csv_exists:
                    writer.writerow(
                        ["item_id", "invoice_source_ref", "predicted_label", "resolved_label", "confidence", "crop_image_path", "timestamp"]
                    )
                writer.writerow(
                    [item_id, source_ref, pred_label, label, conf, str(crop_path), resolved_at]
                )
            logger.info(f"Logged classifier miss for trained class '{label}' to '{self.misses_csv_path}'.")

        cursor.execute(
            """
            UPDATE review_queue
            SET status = 'resolved', resolved_label = ?, resolution_type = ?, resolved_at = ?
            WHERE id = ?
            """,
            (label, resolution_type, resolved_at, item_id),
        )
        conn.commit()
        conn.close()

        return ReviewItem(
            id=item_id,
            invoice_source_ref=source_ref,
            predicted_label=pred_label,
            confidence_or_similarity=conf,
            crop_image_path=crop_path,
            status="resolved",
            resolved_label=label,
            resolution_type=resolution_type,
            created_at=created_at,
            resolved_at=resolved_at,
        )

    def list_pending(self, limit: int = 50) -> list[ReviewItem]:
        """List pending review queue items up to limit for dashboard/API display.

        Args:
            limit: Maximum items to return.

        Returns:
            List of pending ReviewItem objects ordered by creation date ascending.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, invoice_source_ref, predicted_label, confidence_or_similarity, crop_image_path, status, resolved_label, resolution_type, created_at, resolved_at
            FROM review_queue
            WHERE status = 'pending'
            ORDER BY id ASC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        items = []
        for r in rows:
            items.append(
                ReviewItem(
                    id=r[0],
                    invoice_source_ref=r[1],
                    predicted_label=r[2],
                    confidence_or_similarity=float(r[3]),
                    crop_image_path=Path(r[4]),
                    status=r[5],
                    resolved_label=r[6],
                    resolution_type=r[7],
                    created_at=r[8],
                    resolved_at=r[9],
                )
            )
        return items

    def delete_item(self, review_item_id: int) -> bool:
        """Delete an item from the review queue database.

        Args:
            review_item_id: ID of the review item to remove.

        Returns:
            True if an item was deleted, False if not found.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT crop_image_path FROM review_queue WHERE id = ?", (review_item_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        crop_path_str = row[0]
        cursor.execute("DELETE FROM review_queue WHERE id = ?", (review_item_id,))
        conn.commit()
        conn.close()

        if crop_path_str:
            crop_path = Path(crop_path_str)
            if crop_path.exists():
                try:
                    crop_path.unlink()
                except Exception as err:  # noqa: BLE001
                    logger.warning(f"Failed to remove crop image '{crop_path}': {err}")

        logger.info(f"Deleted review item #{review_item_id}.")
        return True


# Singleton manager instance
_default_review_manager: ReviewQueueManager | None = None


def get_review_queue_manager() -> ReviewQueueManager:
    """Return or initialize global singleton ReviewQueueManager instance."""
    global _default_review_manager
    if _default_review_manager is None:
        _default_review_manager = ReviewQueueManager()
    return _default_review_manager


def enqueue_for_review(
    classification_result: Any,
    source_ref: str = "",
    queue_manager: ReviewQueueManager | None = None,
) -> ReviewItem:
    """Enqueue an unknown classification result for human review.

    Args:
        classification_result: ClassificationResult instance from Phase 10.
        source_ref: Optional source invoice file reference.
        queue_manager: Optional ReviewQueueManager instance.

    Returns:
        Created ReviewItem dataclass.
    """
    mgr = queue_manager or get_review_queue_manager()
    ref = source_ref or str(getattr(classification_result, "source_path", ""))
    return mgr.enqueue(
        crop_image=classification_result.crop_image,
        confidence_or_similarity=getattr(classification_result, "confidence_or_similarity", 0.0),
        invoice_source_ref=ref,
        predicted_label=getattr(classification_result, "predicted_label", None),
    )
