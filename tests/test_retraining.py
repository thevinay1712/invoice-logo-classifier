import json

from invoice_logo_classifier.gallery import VendorGallery
from scripts.find_graduation_candidates import find_graduation_candidates
from scripts.retrain_with_new_labels import rollback_version


def test_find_graduation_candidates(tmp_path):
    gallery_dir = tmp_path / "gallery_store"
    gallery_dir.mkdir()

    gallery = VendorGallery(embedding_dim=4)
    # Add 45 entries for 'AcmeCorp' (clears 40 threshold) and 10 entries for 'BetaCorp'
    vec = [0.1, 0.2, 0.3, 0.4]
    for i in range(45):
        gallery.add("AcmeCorp", vec, source_ref=f"inv_{i}.png")
    for i in range(10):
        gallery.add("BetaCorp", vec, source_ref=f"inv_b_{i}.png")

    gallery.save(gallery_dir)

    output_json = tmp_path / "candidates.json"
    candidates = find_graduation_candidates(
        gallery_dir=gallery_dir,
        min_embeddings=40,
        output_json=output_json
    )

    assert len(candidates) == 1
    assert candidates[0]["vendor_label"] == "AcmeCorp"
    assert candidates[0]["embedding_count"] == 45

    assert output_json.exists()
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["total_candidates"] == 1


def test_vendor_gallery_remove_vendor():
    gallery = VendorGallery(embedding_dim=4)
    vec = [0.1, 0.2, 0.3, 0.4]
    gallery.add("VendorA", vec)
    gallery.add("VendorA", vec)
    gallery.add("VendorB", vec)

    assert gallery.total_entries == 3

    removed = gallery.remove_vendor("VendorA")
    assert removed == 2
    assert gallery.total_entries == 1
    assert gallery.metadata[0]["vendor_label"] == "VendorB"


def test_rollback_version(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "v1").mkdir()

    rollback_version("v1", model_dir)

    pointer_file = model_dir / "ACTIVE_VERSION.json"
    assert pointer_file.exists()
    data = json.loads(pointer_file.read_text(encoding="utf-8"))
    assert data["active_version"] == "v1"


def test_export_classifier_misses(tmp_path):
    import sqlite3
    from scripts.export_review_labels_for_retraining import export_classifier_misses

    db_path = tmp_path / "test_review_queue.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE review_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_source_ref TEXT,
            predicted_label TEXT,
            resolved_label TEXT,
            confidence_or_similarity REAL,
            crop_image_path TEXT,
            status TEXT,
            resolution_type TEXT,
            resolved_at TEXT,
            created_at TEXT
        )
        """
    )
    cursor.execute(
        """
        INSERT INTO review_queue (invoice_source_ref, predicted_label, resolved_label, confidence_or_similarity, crop_image_path, status, resolution_type, resolved_at, created_at)
        VALUES ('inv_1.png', 'Nike', 'Adidas', 0.42, 'crop_1.png', 'resolved', 'classifier_missed_trained_class', '2026-10-22T00:00:00Z', '2026-10-22T00:00:00Z')
        """
    )
    conn.commit()
    conn.close()

    out_json = tmp_path / "export.json"
    misses = export_classifier_misses(db_path=db_path, output_path=out_json)

    assert len(misses) == 1
    assert misses[0]["resolved_label"] == "Adidas"
    assert out_json.exists()
    export_content = json.loads(out_json.read_text(encoding="utf-8"))
    assert export_content["total_classifier_misses"] == 1

