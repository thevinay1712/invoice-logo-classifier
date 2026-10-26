#!/usr/bin/env python3
"""
System State Backup & Restore Tool for Invoice Logo Classifier.

Backs up critical persistent application state:
1. Trained model weights & metadata (`models/`)
2. FAISS embedding index & SQLite metadata (`gallery_store/`)
3. Review queue SQLite database & crop images (`review_queue/`)

Losing any of these components means losing real accumulated training work or human review
feedback. This script creates timestamped .tar.gz archives suitable for automated cron jobs
or disaster recovery.
"""

import argparse
import datetime
import tarfile
from pathlib import Path

DEFAULT_TARGETS = [
    Path("models"),
    Path("gallery_store"),
    Path("review_queue"),
]


def create_backup(output_dir: Path, base_dir: Path) -> Path:
    """Creates a compressed .tar.gz backup of all system state directories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    archive_name = f"backup_invoice_logo_classifier_{timestamp}.tar.gz"
    archive_path = output_dir / archive_name

    print(f"[*] Starting system state backup to: {archive_path}")
    count = 0
    with tarfile.open(archive_path, "w:gz") as tar:
        for target_rel in DEFAULT_TARGETS:
            target_full = base_dir / target_rel
            if target_full.exists():
                print(f"  [+] Adding target: {target_rel} (Path: {target_full})")
                tar.add(target_full, arcname=target_rel.as_posix())
                count += 1
            else:
                print(f"  [-] Target skipped (not found): {target_rel}")

    if count == 0:
        print("[!] Warning: No persistent state targets were found to backup!")
    else:
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Backup complete! Total size: {size_mb:.2f} MB")

    return archive_path


def restore_backup(archive_path: Path, dest_dir: Path) -> None:
    """Restores system state from a compressed .tar.gz backup archive."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Backup archive does not exist: {archive_path}")

    print(f"[*] Restoring system state from: {archive_path} to {dest_dir}")
    with tarfile.open(archive_path, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=dest_dir, filter="data")
        else:
            tar.extractall(path=dest_dir)
    print("[OK] Restore complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Backup and restore tool for invoice-logo-classifier state."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backups"),
        help="Directory where backup archives will be stored (default: backups/)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base project directory containing target state folders (default: .)"
    )
    parser.add_argument(
        "--restore",
        type=Path,
        default=None,
        help="Path to backup .tar.gz file to restore"
    )

    args = parser.parse_args()

    if args.restore:
        restore_backup(args.restore, args.base_dir)
    else:
        create_backup(args.output_dir, args.base_dir)


if __name__ == "__main__":
    main()
