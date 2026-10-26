"""Hard negative generator for invoice non-logo document clutter.

Generates realistic invoice document elements that are not vendor logos:
- Address & Tax ID header text blocks
- 1D/2D Barcodes and tracking strips
- Table border grids & invoice financial summary lines
- Blank scanned paper textures with scanner banding and fold shadows
- Document processing rubber stamps (PAID, RECEIVED, APPROVED)
"""

import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def _draw_address_block(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    """Draw multi-line address and company contact text."""
    lines = [
        "ACCOUNTS PAYABLE DEPT",
        "100 CORPORATE PARKWAY, SUITE 400",
        "AUSTIN, TX 78701 USA",
        "PHONE: (512) 555-0149",
        "TAX ID / EIN: 74-2910382",
        "BILLING INQUIRIES: BILLING@CORP.NET",
    ]
    sample_lines = random.sample(lines, k=random.randint(3, len(lines)))
    y = random.randint(10, 25)
    for line in sample_lines:
        x = random.randint(10, 25)
        draw.text((x, y), line, fill=(random.randint(20, 60), random.randint(20, 60), random.randint(20, 60)))
        y += random.randint(16, 24)


def _draw_barcode_strip(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    """Draw realistic 1D barcode stripes with tracking numbers."""
    start_x = random.randint(15, 30)
    end_x = w - random.randint(15, 30)
    bar_y_start = random.randint(20, 35)
    bar_y_end = h - random.randint(35, 50)

    cur_x = start_x
    while cur_x < end_x:
        bar_w = random.choice([1, 2, 3, 4])
        color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
        draw.rectangle([cur_x, bar_y_start, cur_x + bar_w, bar_y_end], fill=color)
        gap = random.choice([2, 3, 4, 6])
        cur_x += bar_w + gap

    # Numbers under barcode
    barcode_digits = f"*{random.randint(100000, 999999)}-{random.randint(1000, 9999)}*"
    draw.text((start_x + 10, bar_y_end + 4), barcode_digits, fill=(40, 40, 40))


def _draw_table_grid(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    """Draw table gridlines, column headers, and itemized rows."""
    draw.rectangle([10, 15, w - 10, h - 15], outline=(160, 160, 160), width=1)
    # Header row
    draw.line([(10, 40), (w - 10, 40)], fill=(120, 120, 120), width=2)
    draw.text((15, 20), "QTY", fill=(50, 50, 50))
    draw.text((60, 20), "DESCRIPTION", fill=(50, 50, 50))
    draw.text((w - 75, 20), "AMOUNT", fill=(50, 50, 50))

    # Grid rows
    y = 65
    while y < h - 25:
        draw.line([(10, y), (w - 10, y)], fill=(200, 200, 200), width=1)
        draw.text((18, y + 4), str(random.randint(1, 10)), fill=(80, 80, 80))
        draw.text((60, y + 4), f"Professional Services Tier {random.randint(1, 4)}", fill=(80, 80, 80))
        draw.text((w - 75, y + 4), f"${random.randint(100, 950)}.00", fill=(80, 80, 80))
        y += 28


def _draw_rubber_stamp(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    """Draw an ink rubber stamp (PAID, RECEIVED, APPROVED)."""
    stamp_text = random.choice(["PAID", "RECEIVED", "APPROVED", "VERIFIED", "VOID"])
    color = random.choice([
        (190, 40, 40),   # Red
        (30, 70, 180),   # Blue
        (40, 140, 60),   # Green
    ])

    cx, cy = w // 2, h // 2
    sw, sh = int(w * 0.75), int(h * 0.45)
    box = [cx - sw // 2, cy - sh // 2, cx + sw // 2, cy + sh // 2]
    draw.rectangle(box, outline=color, width=3)
    draw.text((cx - sw // 2 + 20, cy - 10), stamp_text, fill=color)
    draw.text((cx - sw // 2 + 15, cy + 12), f"DATE: 2026-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}", fill=color)


def _draw_textured_blank_paper(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    """Draw blank scanned paper with fold lines, staple shadow, and scanner grain."""
    # Faint fold line
    fold_y = random.randint(h // 4, 3 * h // 4)
    draw.line([(0, fold_y), (w, fold_y)], fill=(225, 225, 220), width=2)
    # Staple mark in upper left corner
    staple_x = random.randint(15, 30)
    staple_y = random.randint(15, 30)
    draw.rectangle([staple_x, staple_y, staple_x + 12, staple_y + 3], fill=(100, 100, 105))


def generate_clutter_crop(size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Generate a single realistic non-logo document clutter crop.

    Args:
        size: Target (width, height) in pixels.

    Returns:
        PIL Image containing non-logo invoice document elements.
    """
    w, h = size
    # Off-white / light cream document paper background
    bg_color = (
        random.randint(245, 255),
        random.randint(245, 255),
        random.randint(240, 252),
    )
    img = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)

    generator_type = random.choice([
        "address",
        "barcode",
        "table",
        "stamp",
        "blank_paper",
    ])

    if generator_type == "address":
        _draw_address_block(draw, w, h)
    elif generator_type == "barcode":
        _draw_barcode_strip(draw, w, h)
    elif generator_type == "table":
        _draw_table_grid(draw, w, h)
    elif generator_type == "stamp":
        _draw_rubber_stamp(draw, w, h)
    else:
        _draw_textured_blank_paper(draw, w, h)

    return img


def generate_clutter_dataset(
    output_base_dir: Path | str,
    train_count: int = 150,
    val_count: int = 30,
    test_count: int = 20,
    class_name: str = "_Non_Logo_Clutter",
) -> dict[str, int]:
    """Generate hard negative clutter crops for train, val, and test splits.

    Args:
        output_base_dir: Root directory containing train/, val/, test/ folders.
        train_count: Number of training clutter crops.
        val_count: Number of validation clutter crops.
        test_count: Number of test clutter crops.
        class_name: Subdirectory class name (default '_Non_Logo_Clutter').

    Returns:
        Dictionary mapping split name to generated image count.
    """
    base_path = Path(output_base_dir)
    counts = {}

    splits = [
        ("train", train_count),
        ("val", val_count),
        ("test", test_count),
    ]

    for split_name, count in splits:
        split_dir = base_path / split_name / class_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            crop_img = generate_clutter_crop((224, 224))
            file_path = split_dir / f"clutter_{i:04d}.png"
            crop_img.save(file_path, format="PNG")

        counts[split_name] = count

    return counts
