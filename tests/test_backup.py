from scripts.backup_system_data import create_backup, restore_backup


def test_backup_and_restore(tmp_path):
    # Setup dummy directory structure
    base_dir = tmp_path / "app"
    models_dir = base_dir / "models"
    gallery_dir = base_dir / "gallery_store"
    review_dir = base_dir / "review_queue"

    models_dir.mkdir(parents=True)
    gallery_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)

    (models_dir / "best_model.pth").write_text("dummy model weights")
    (gallery_dir / "faiss_index.bin").write_text("dummy faiss index")
    (review_dir / "review_queue.db").write_text("dummy sqlite db")

    output_dir = tmp_path / "backups"

    # Test create_backup
    archive_path = create_backup(output_dir=output_dir, base_dir=base_dir)
    assert archive_path.exists()
    assert archive_path.stat().st_size > 0

    # Test restore_backup
    restore_target = tmp_path / "restore_app"
    restore_target.mkdir()
    restore_backup(archive_path=archive_path, dest_dir=restore_target)

    assert (restore_target / "models" / "best_model.pth").read_text() == "dummy model weights"
    assert (restore_target / "gallery_store" / "faiss_index.bin").read_text() == "dummy faiss index"
    assert (restore_target / "review_queue" / "review_queue.db").read_text() == "dummy sqlite db"
