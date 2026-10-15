"""Script to clean and re-populate the Review Queue with realistic pending vendor logo crops."""

import sqlite3
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

REVIEW_DIR = Path("review_queue")
CROPS_DIR = REVIEW_DIR / "crops"
DB_PATH = REVIEW_DIR / "review_queue.db"

# Sample unknown vendors for review queue simulation
SAMPLES = [
    {"name": "TechLogisticsCorp", "color": (41, 128, 185), "score": 0.42, "top": "DHL"},
    {"name": "ApexLogistics", "color": (39, 174, 96), "score": 0.38, "top": "FedEx"},
    {"name": "GlobalServicesInc", "color": (142, 68, 173), "score": 0.51, "top": "UPS"},
    {"name": "VertexSystems", "color": (230, 126, 34), "score": 0.29, "top": "Cisco"},
]

def generate_crop_image(name: str, color: tuple[int, int, int], path: Path, size=(300, 100)):
    """Generate a clean synthetic logo crop image for review items."""
    img = Image.new("RGB", size, color=(248, 249, 250))
    draw = ImageDraw.Draw(img)
    
    # Outer border & logo badge
    draw.rectangle([10, 10, size[0] - 10, size[1] - 10], outline=(220, 224, 230), width=2)
    draw.rectangle([25, 20, 85, size[1] - 20], fill=color)
    draw.text((35, size[1] // 2 - 8), name[:3].upper(), fill=(255, 255, 255))
    
    # Text label
    draw.text((105, size[1] // 2 - 10), name, fill=(40, 40, 40))
    img.save(path)

def main():
    print("Resetting and re-seeding Review Queue with clean vendor logo crops...")
    
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Clear existing table
    cursor.execute("DROP TABLE IF EXISTS review_queue")
    cursor.execute(
        """
        CREATE TABLE review_queue (
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
    
    for i, s in enumerate(SAMPLES, start=1):
        crop_path = CROPS_DIR / f"crop_{i:03d}_{s['name']}.png"
        generate_crop_image(s["name"], s["color"], crop_path)
        
        cursor.execute(
            """
            INSERT INTO review_queue (invoice_source_ref, predicted_label, confidence_or_similarity, crop_image_path, status, created_at)
            VALUES (?, ?, ?, ?, 'pending', datetime('now'))
            """,
            (f"sample_invoice_{i:02d}.png", s["top"], s["score"], str(crop_path))
        )
        
    conn.commit()
    conn.close()
    print("Review Queue successfully re-seeded with realistic vendor crops.")

if __name__ == "__main__":
    main()
