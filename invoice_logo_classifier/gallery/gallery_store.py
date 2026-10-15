"""Persistent FAISS and SQLite embedding gallery store for unseen vendor logo matching."""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from invoice_logo_classifier.config import settings

logger = logging.getLogger(__name__)


# Note: The default similarity threshold is a baseline configuration placeholder; Phase 15 empirically calibrates the optimal operating threshold.


@dataclass
class GalleryMatch:
    """Dataclass storing nearest-neighbor search match details."""

    vendor_label: str
    similarity_score: float
    source_invoice_ref: str
    added_at: str
    vector_index: int


class VendorGallery:
    """Persistent vector database combining FAISS index search with SQLite metadata storage."""

    def __init__(self, embedding_dim: int = 512):
        """Initialize VendorGallery vector index and metadata store.

        Args:
            embedding_dim: Dimension of L2-normalized feature vectors (default 512).
        """
        self.embedding_dim = embedding_dim
        # IndexFlatIP measures Inner Product (exact Cosine Similarity on L2-normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata: list[dict[str, Any]] = []

    @property
    def total_entries(self) -> int:
        """Return total count of vectors stored in the gallery index."""
        return self.index.ntotal

    def add(
        self,
        vendor_label: str,
        embedding: np.ndarray,
        source_ref: str = "",
    ) -> int:
        """Add a new vendor logo embedding entry to the gallery.

        Args:
            vendor_label: Vendor / brand name string (human provided).
            embedding: 1D or 2D float32 numpy vector array.
            source_ref: Optional source invoice file reference string.

        Returns:
            Vector index integer assigned to the new entry.
        """
        vec = np.ascontiguousarray(embedding, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        if vec.shape[1] != self.embedding_dim:
            # Dynamically adapt index dimension on first insert if needed
            if self.total_entries == 0:
                self.embedding_dim = vec.shape[1]
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, got {vec.shape[1]}"
                )

        # Ensure L2 normalization
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        norm_vec = vec / norm

        vector_index = self.total_entries
        self.index.add(norm_vec)

        entry_meta = {
            "vector_index": vector_index,
            "vendor_label": str(vendor_label).strip(),
            "source_invoice_ref": str(source_ref),
            "added_at": datetime.now(tz=UTC).isoformat(),
        }
        self.metadata.append(entry_meta)

        logger.info(
            f"Added vendor logo embedding for '{vendor_label}' to gallery (index #{vector_index})."
        )
        return vector_index

    def remove_vendor(self, vendor_label: str) -> int:
        """Remove all embedding entries for a given vendor label from the gallery.

        Args:
            vendor_label: Vendor / brand label string to remove.

        Returns:
            Number of removed embedding entries.
        """
        target = str(vendor_label).strip()
        keep_indices = [i for i, m in enumerate(self.metadata) if m["vendor_label"] != target]
        removed_count = len(self.metadata) - len(keep_indices)

        if removed_count == 0:
            logger.info(f"No gallery entries found for vendor '{target}'.")
            return 0

        # Rebuild FAISS index with remaining vectors
        new_index = faiss.IndexFlatIP(self.embedding_dim)
        new_metadata = []

        if keep_indices and self.index.ntotal > 0:
            remaining_vecs = []
            for old_idx in keep_indices:
                vec = self.index.reconstruct(old_idx)
                remaining_vecs.append(vec)
                meta = self.metadata[old_idx].copy()
                meta["vector_index"] = len(remaining_vecs) - 1
                new_metadata.append(meta)

            arr = np.ascontiguousarray(np.vstack(remaining_vecs), dtype=np.float32)
            new_index.add(arr)

        self.index = new_index
        self.metadata = new_metadata

        logger.info(
            f"Removed {removed_count} gallery entries for vendor '{target}'. Remaining vectors: {self.total_entries}."
        )
        return removed_count

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[GalleryMatch]:
        """Search top-k nearest neighbor matches by cosine similarity.

        Args:
            query_embedding: 1D or 2D float32 query vector.
            k: Top-k nearest neighbors to return.

        Returns:
            List of GalleryMatch objects ordered by descending similarity score.
        """
        if self.total_entries == 0:
            return []

        q_vec = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        if q_vec.shape[1] != self.embedding_dim:
            logger.warning(
                f"Query embedding dimension {q_vec.shape[1]} does not match gallery index dimension {self.embedding_dim}. Returning empty matches."
            )
            return []

        norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        q_norm = q_vec / norm

        top_k = min(k, self.total_entries)
        scores, indices = self.index.search(q_norm, top_k)

        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            matches.append(
                GalleryMatch(
                    vendor_label=meta["vendor_label"],
                    similarity_score=float(score),
                    source_invoice_ref=meta["source_invoice_ref"],
                    added_at=meta["added_at"],
                    vector_index=int(idx),
                )
            )

        return matches

    def best_match(
        self,
        query_embedding: np.ndarray,
        threshold: float | None = None,
    ) -> GalleryMatch | None:
        """Return single top match if its cosine similarity score meets or exceeds threshold.

        Args:
            query_embedding: Query embedding vector.
            threshold: Cosine similarity threshold (default settings.gallery_similarity_threshold).

        Returns:
            GalleryMatch object if top match clears threshold, else None.
        """
        target_thresh = (
            threshold if threshold is not None else settings.gallery_similarity_threshold
        )
        matches = self.search(query_embedding, k=1)
        if matches and matches[0].similarity_score >= target_thresh:
            return matches[0]
        return None

    def get_vendor_summary(self) -> dict[str, int]:
        """Return dict mapping each vendor_label to count of stored embeddings."""
        summary: dict[str, int] = {}
        for meta in self.metadata:
            lbl = meta["vendor_label"]
            summary[lbl] = summary.get(lbl, 0) + 1
        return summary

    def save(self, directory: Path | str) -> Path:
        """Save FAISS index and parallel SQLite metadata database to disk.

        Args:
            directory: Path to gallery destination folder.

        Returns:
            Path to saved gallery directory.
        """
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        faiss_file = target_dir / "gallery_index.faiss"
        sqlite_file = target_dir / "gallery_metadata.db"

        # Save FAISS index
        faiss.write_index(self.index, str(faiss_file))

        # Save SQLite metadata database
        conn = sqlite3.connect(str(sqlite_file))
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS gallery_entries")
        cursor.execute(
            """
            CREATE TABLE gallery_entries (
                vector_index INTEGER PRIMARY KEY,
                vendor_label TEXT NOT NULL,
                source_invoice_ref TEXT,
                added_at TEXT NOT NULL
            )
            """
        )

        for meta in self.metadata:
            cursor.execute(
                """
                INSERT INTO gallery_entries (vector_index, vendor_label, source_invoice_ref, added_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    meta["vector_index"],
                    meta["vendor_label"],
                    meta["source_invoice_ref"],
                    meta["added_at"],
                ),
            )

        conn.commit()
        conn.close()

        logger.info(
            f"Saved gallery with {self.total_entries} entries to '{target_dir}'."
        )
        return target_dir

    @classmethod
    def load(cls, directory: Path | str) -> "VendorGallery":
        """Load VendorGallery instance from saved FAISS index and SQLite metadata database.

        Args:
            directory: Directory containing saved gallery index and db files.

        Returns:
            Loaded VendorGallery instance.
        """
        target_dir = Path(directory)
        faiss_file = target_dir / "gallery_index.faiss"
        sqlite_file = target_dir / "gallery_metadata.db"

        if not faiss_file.exists() or not sqlite_file.exists():
            raise FileNotFoundError(
                f"Gallery index or database file not found in '{target_dir}'."
            )

        faiss_index = faiss.read_index(str(faiss_file))
        gallery = cls(embedding_dim=faiss_index.d)
        gallery.index = faiss_index

        conn = sqlite3.connect(str(sqlite_file))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT vector_index, vendor_label, source_invoice_ref, added_at FROM gallery_entries ORDER BY vector_index ASC"
        )
        rows = cursor.fetchall()
        conn.close()

        gallery.metadata = [
            {
                "vector_index": r[0],
                "vendor_label": r[1],
                "source_invoice_ref": r[2],
                "added_at": r[3],
            }
            for r in rows
        ]

        logger.info(f"Loaded gallery with {gallery.total_entries} entries from '{target_dir}'.")
        return gallery
