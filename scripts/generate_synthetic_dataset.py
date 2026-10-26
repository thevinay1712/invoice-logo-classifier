"""Script to generate realistic vendor logo dataset for model training with authentic brand visual signatures.

Generates high-fidelity vector representations for all 30 core commercial enterprise brands:
Adidas, Adobe, Airbnb, Amazon, Apple, Cisco, CocaCola, DHL, FedEx, Ford, Google,
IBM, Intel, Microsoft, Netflix, Nike, Oracle, PayPal, Pepsi, Salesforce, Samsung,
Slack, Sony, Starbucks, Stripe, Target, Toyota, UPS, Uber, Walmart.

Renders with 2x supersampling for crisp anti-aliased edges and incorporates authentic
invoice document variations (clean paper, laser print monochrome, inverted header banners,
subtle document noise lines).
"""

import math
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont

from invoice_logo_classifier.augmentation.clutter import generate_clutter_dataset

# Set seed for reproducible dataset builds
random.seed(42)

CLASSES = [
    "Adidas", "Adobe", "Airbnb", "Amazon", "Apple", "Cisco", "CocaCola",
    "DHL", "FedEx", "Ford", "Google", "IBM", "Intel", "Microsoft",
    "Netflix", "Nike", "Oracle", "PayPal", "Pepsi", "Salesforce",
    "Samsung", "Slack", "Sony", "Starbucks", "Stripe", "Target",
    "Toyota", "UPS", "Uber", "Walmart"
]

PROCESSED_DIR = Path("data/processed")
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"

# Signature brand color schemes
BRAND_COLORS = {
    "Adidas": ((10, 10, 10), (255, 255, 255)),
    "Adobe": ((235, 16, 20), (255, 255, 255)),
    "Airbnb": ((255, 90, 95), (255, 255, 255)),
    "Amazon": ((35, 47, 62), (255, 153, 0)),
    "Apple": ((30, 30, 30), (240, 240, 240)),
    "Cisco": ((4, 159, 217), (255, 255, 255)),
    "CocaCola": ((244, 0, 9), (255, 255, 255)),
    "DHL": ((212, 5, 17), (255, 204, 0)),
    "FedEx": ((77, 20, 140), (255, 102, 0)),
    "Ford": ((0, 52, 120), (255, 255, 255)),
    "Google": ((66, 133, 244), (255, 255, 255)),
    "IBM": ((15, 98, 254), (255, 255, 255)),
    "Intel": ((0, 104, 181), (255, 255, 255)),
    "Microsoft": ((0, 164, 239), (255, 185, 0)),
    "Netflix": ((229, 9, 20), (255, 255, 255)),
    "Nike": ((17, 17, 17), (255, 255, 255)),
    "Oracle": ((199, 70, 52), (255, 255, 255)),
    "PayPal": ((0, 48, 135), (255, 255, 255)),
    "Pepsi": ((0, 94, 184), (235, 25, 51)),
    "Salesforce": ((0, 161, 224), (255, 255, 255)),
    "Samsung": ((20, 40, 160), (255, 255, 255)),
    "Slack": ((74, 21, 75), (236, 178, 46)),
    "Sony": ((20, 20, 20), (255, 255, 255)),
    "Starbucks": ((0, 112, 74), (255, 255, 255)),
    "Stripe": ((99, 91, 255), (255, 255, 255)),
    "Target": ((204, 0, 0), (255, 255, 255)),
    "Toyota": ((235, 10, 30), (255, 255, 255)),
    "UPS": ((53, 29, 16), (255, 184, 28)),
    "Uber": ((0, 0, 0), (255, 255, 255)),
    "Walmart": ((0, 113, 206), (255, 194, 32)),
}


def get_font(size=28, bold=True):
    """Load system TrueType font if available, with robust fallbacks."""
    font_names = (
        ["arialbd.ttf", "segoeuib.ttf", "calibrib.ttf", "Helvetica-Bold.ttf"]
        if bold else
        ["arial.ttf", "segoeui.ttf", "calibri.ttf", "Helvetica.ttf"]
    )
    for f in font_names:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_brand_vector(draw: ImageDraw.ImageDraw, brand: str, w: int, h: int, variant: int = 0):
    """Render authentic visual brand logotype with mathematical vector precision."""
    cx = w // 2
    cy = h // 2
    is_dark_mode = (variant % 4 == 2)
    is_mono = (variant % 4 == 3)

    # ── 1. Adidas ──────────────────────────────────────────
    if brand == "Adidas":
        color = (255, 255, 255) if is_dark_mode else ((20, 20, 20) if not is_mono else (40, 40, 40))
        bar_w = int(w * 0.045)
        angle_dx = int(w * 0.055)
        base_y = int(h * 0.65)
        for k in range(3):
            bar_h = int(h * (0.18 + k * 0.09))
            x_start = cx - int(w * 0.16) + k * int(w * 0.09)
            poly = [
                (x_start, base_y),
                (x_start + bar_w, base_y),
                (x_start + bar_w + angle_dx, base_y - bar_h),
                (x_start + angle_dx, base_y - bar_h),
            ]
            draw.polygon(poly, fill=color)
        f = get_font(int(h * 0.16), bold=True)
        draw.text((cx, int(h * 0.74)), "adidas", fill=color, font=f, anchor="mt")

    # ── 2. Adobe ───────────────────────────────────────────
    elif brand == "Adobe":
        red = (235, 16, 20) if not is_mono else (50, 50, 50)
        sq_size = int(min(w, h) * 0.75)
        sq_box = [cx - sq_size // 2, cy - sq_size // 2, cx + sq_size // 2, cy + sq_size // 2]
        draw.rectangle(sq_box, fill=red)
        # Left sloped wing
        poly_left = [
            (cx - int(sq_size * 0.28), cy + int(sq_size * 0.32)),
            (cx - int(sq_size * 0.05), cy - int(sq_size * 0.35)),
            (cx - int(sq_size * 0.28), cy - int(sq_size * 0.35)),
        ]
        draw.polygon(poly_left, fill=(255, 255, 255))
        # Right vertical wing
        poly_right = [
            (cx + int(sq_size * 0.28), cy + int(sq_size * 0.32)),
            (cx + int(sq_size * 0.05), cy - int(sq_size * 0.35)),
            (cx + int(sq_size * 0.28), cy - int(sq_size * 0.35)),
        ]
        draw.polygon(poly_right, fill=(255, 255, 255))
        # Central connector
        draw.polygon([
            (cx - int(sq_size * 0.14), cy + int(sq_size * 0.18)),
            (cx + int(sq_size * 0.14), cy + int(sq_size * 0.18)),
            (cx + int(sq_size * 0.22), cy + int(sq_size * 0.32)),
            (cx - int(sq_size * 0.22), cy + int(sq_size * 0.32)),
        ], fill=(255, 255, 255))

    # ── 3. Airbnb ──────────────────────────────────────────
    elif brand == "Airbnb":
        coral = (255, 90, 95) if not is_mono else (50, 50, 50)
        rw, rh = int(w * 0.18), int(h * 0.32)
        top_y = cy - int(h * 0.18)
        draw.ellipse([cx - rw, top_y - rh // 2, cx + rw, top_y + rh // 2], outline=coral, width=int(w * 0.035))
        draw.arc([cx - int(rw * 0.6), top_y - int(rh * 0.1), cx + int(rw * 0.6), top_y + rh], start=0, end=180, fill=coral, width=int(w * 0.035))
        draw.polygon([(cx, top_y + rh), (cx - int(rw * 0.7), top_y + int(rh * 0.6)), (cx + int(rw * 0.7), top_y + int(rh * 0.6))], fill=coral)
        f = get_font(int(h * 0.15), bold=True)
        draw.text((cx, cy + int(h * 0.24)), "airbnb", fill=coral, font=f, anchor="mt")

    # ── 4. Amazon ──────────────────────────────────────────
    elif brand == "Amazon":
        text_color = (255, 255, 255) if is_dark_mode else (35, 47, 62)
        arrow_color = (255, 153, 0) if not is_mono else (100, 100, 100)
        f = get_font(int(h * 0.24), bold=True)
        draw.text((cx, cy - int(h * 0.08)), "amazon", fill=text_color, font=f, anchor="mm")
        arc_box = [cx - int(w * 0.32), cy - int(h * 0.04), cx + int(w * 0.28), cy + int(h * 0.24)]
        draw.arc(arc_box, start=25, end=155, fill=arrow_color, width=int(w * 0.025))
        ax = cx + int(w * 0.24)
        ay = cy + int(h * 0.14)
        draw.polygon([(ax - 4, ay + 6), (ax + 12, ay + 2), (ax + 2, ay - 10)], fill=arrow_color)

    # ── 5. Apple ───────────────────────────────────────────
    elif brand == "Apple":
        color = (255, 255, 255) if is_dark_mode else (30, 30, 30)
        rw, rh = int(w * 0.22), int(h * 0.26)
        # Leaf
        draw.ellipse([cx, cy - rh - int(h * 0.14), cx + int(w * 0.12), cy - rh + int(h * 0.02)], fill=color)
        # Body
        draw.ellipse([cx - rw, cy - int(rh * 0.6), cx + int(w * 0.04), cy + rh], fill=color)
        draw.ellipse([cx - int(w * 0.04), cy - int(rh * 0.6), cx + rw, cy + rh], fill=color)
        # Bite cutout
        bg_col = (20, 20, 20) if is_dark_mode else (255, 255, 255)
        draw.ellipse([cx + int(rw * 0.4), cy - int(rh * 0.45), cx + int(rw * 1.4), cy + int(rh * 0.55)], fill=bg_col)

    # ── 6. Cisco ───────────────────────────────────────────
    elif brand == "Cisco":
        blue = (4, 159, 217) if not is_mono else (60, 60, 60)
        bar_heights = [0.12, 0.22, 0.16, 0.32, 0.18, 0.32, 0.16, 0.22, 0.12]
        bw = int(w * 0.022)
        spacing = int(w * 0.048)
        start_x = cx - int(len(bar_heights) * spacing // 2)
        base_y = cy + int(h * 0.06)
        for i, bh in enumerate(bar_heights):
            x = start_x + i * spacing
            bar_h = int(h * bh)
            draw.rounded_rectangle([x, base_y - bar_h, x + bw, base_y], radius=bw // 2, fill=blue)
        f = get_font(int(h * 0.18), bold=True)
        draw.text((cx, cy + int(h * 0.18)), "CISCO", fill=blue, font=f, anchor="mt")

    # ── 7. CocaCola ────────────────────────────────────────
    elif brand == "CocaCola":
        red = (244, 0, 9) if not is_mono else (40, 40, 40)
        draw.rectangle([0, 0, w, h], fill=red)
        f = get_font(int(h * 0.24), bold=True)
        draw.text((cx, cy - int(h * 0.06)), "Coca-Cola", fill=(255, 255, 255), font=f, anchor="mm")
        draw.arc([cx - int(w * 0.42), cy, cx + int(w * 0.42), cy + int(h * 0.32)], start=15, end=165, fill=(255, 255, 255), width=int(w * 0.025))

    # ── 8. DHL ─────────────────────────────────────────────
    elif brand == "DHL":
        yellow = (255, 204, 0) if not is_mono else (220, 220, 220)
        red = (212, 5, 17) if not is_mono else (30, 30, 30)
        draw.rectangle([0, 0, w, h], fill=yellow)
        for offset in [-int(h * 0.18), 0, int(h * 0.18)]:
            draw.line([(int(w * 0.08), cy + offset), (int(w * 0.92), cy + offset)], fill=red, width=int(h * 0.035))
        f = get_font(int(h * 0.36), bold=True)
        draw.text((cx, cy), "DHL", fill=red, font=f, anchor="mm")

    # ── 9. FedEx ───────────────────────────────────────────
    elif brand == "FedEx":
        purple = (77, 20, 140) if not is_mono else (40, 40, 40)
        orange = (255, 102, 0) if not is_mono else (90, 90, 90)
        f = get_font(int(h * 0.30), bold=True)
        draw.text((cx - int(w * 0.14), cy), "Fed", fill=purple, font=f, anchor="mm")
        draw.text((cx + int(w * 0.18), cy), "Ex", fill=orange, font=f, anchor="mm")

    # ── 10. Ford ───────────────────────────────────────────
    elif brand == "Ford":
        blue = (0, 52, 120) if not is_mono else (40, 40, 40)
        oval_box = [cx - int(w * 0.40), cy - int(h * 0.28), cx + int(w * 0.40), cy + int(h * 0.28)]
        draw.ellipse(oval_box, fill=blue, outline=(230, 230, 240), width=int(w * 0.02))
        f = get_font(int(h * 0.26), bold=True)
        draw.text((cx, cy - 2), "Ford", fill=(255, 255, 255), font=f, anchor="mm")

    # ── 11. Google ─────────────────────────────────────────
    elif brand == "Google":
        blue = (66, 133, 244) if not is_mono else (50, 50, 50)
        red = (234, 67, 53) if not is_mono else (80, 80, 80)
        yellow = (251, 188, 5) if not is_mono else (120, 120, 120)
        green = (52, 168, 83) if not is_mono else (60, 60, 60)

        # 50% authentic multi-color wordmark (real invoice style), 50% circular G emblem
        if variant % 2 == 0:
            letters = [
                ("G", blue),
                ("o", red),
                ("o", yellow),
                ("g", blue),
                ("l", green),
                ("e", red),
            ]
            font_size = min(int(w * 0.22), int(h * 0.35))
            f = get_font(font_size, bold=True)
            widths = []
            for ch, _ in letters:
                bbox = draw.textbbox((0, 0), ch, font=f)
                widths.append(bbox[2] - bbox[0])
            spacing = max(int(font_size * 0.04), 1)
            total_w = sum(widths) + spacing * (len(letters) - 1)
            cur_x = (w - total_w) // 2
            for ch, col in letters:
                draw.text((cur_x, cy), ch, fill=col, font=f, anchor="lm")
                bbox = draw.textbbox((cur_x, cy), ch, font=f, anchor="lm")
                cur_x += (bbox[2] - bbox[0]) + spacing
        else:
            r = int(min(w, h) * 0.32)
            box = [cx - r, cy - int(h * 0.12) - r, cx + r, cy - int(h * 0.12) + r]
            thick = int(r * 0.38)
            draw.arc(box, start=290, end=45, fill=blue, width=thick)
            draw.arc(box, start=45, end=135, fill=green, width=thick)
            draw.arc(box, start=135, end=215, fill=yellow, width=thick)
            draw.arc(box, start=215, end=290, fill=red, width=thick)
            draw.rectangle([cx, cy - int(h * 0.12) - thick // 2, cx + r, cy - int(h * 0.12) + thick // 2], fill=blue)
            f = get_font(int(h * 0.15), bold=True)
            draw.text((cx, cy + int(h * 0.28)), "Google", fill=(70, 70, 70), font=f, anchor="mt")

    # ── 12. IBM ────────────────────────────────────────────
    elif brand == "IBM":
        blue = (15, 98, 254) if not is_mono else (40, 40, 40)
        f = get_font(int(h * 0.36), bold=True)
        draw.text((cx, cy), "IBM", fill=blue, font=f, anchor="mm")
        num_stripes = 8
        stripe_h = max(int(h * 0.02), 2)
        bg = (20, 20, 20) if is_dark_mode else (255, 255, 255)
        for s in range(num_stripes):
            sy = cy - int(h * 0.18) + int(s * (h * 0.36 / num_stripes))
            draw.line([(cx - int(w * 0.32), sy), (cx + int(w * 0.32), sy)], fill=bg, width=stripe_h)

    # ── 13. Intel ──────────────────────────────────────────
    elif brand == "Intel":
        blue = (0, 104, 181) if not is_mono else (40, 40, 40)
        f = get_font(int(h * 0.24), bold=True)
        draw.text((cx, cy), "intel", fill=blue, font=f, anchor="mm")
        oval = [cx - int(w * 0.38), cy - int(h * 0.22), cx + int(w * 0.38), cy + int(h * 0.22)]
        draw.arc(oval, start=15, end=165, fill=(0, 163, 224), width=int(w * 0.02))

    # ── 14. Microsoft ──────────────────────────────────────
    elif brand == "Microsoft":
        qw = int(w * 0.08)
        gap = int(w * 0.02)
        sx = cx - int(w * 0.34)
        c1 = (242, 80, 34) if not is_mono else (50, 50, 50)
        c2 = (127, 186, 0) if not is_mono else (80, 80, 80)
        c3 = (0, 164, 239) if not is_mono else (60, 60, 60)
        c4 = (255, 185, 0) if not is_mono else (100, 100, 100)
        draw.rectangle([sx, cy - qw - gap // 2, sx + qw, cy - gap // 2], fill=c1)
        draw.rectangle([sx + qw + gap, cy - qw - gap // 2, sx + 2 * qw + gap, cy - gap // 2], fill=c2)
        draw.rectangle([sx, cy + gap // 2, sx + qw, cy + qw + gap // 2], fill=c3)
        draw.rectangle([sx + qw + gap, cy + gap // 2, sx + 2 * qw + gap, cy + qw + gap // 2], fill=c4)
        f = get_font(int(h * 0.20), bold=True)
        tc = (240, 240, 240) if is_dark_mode else (115, 115, 115)
        draw.text((sx + 2 * qw + gap * 2 + int(w * 0.04), cy), "Microsoft", fill=tc, font=f, anchor="lm")

    # ── 15. Netflix ────────────────────────────────────────
    elif brand == "Netflix":
        red = (229, 9, 20)
        draw.rectangle([0, 0, w, h], fill=(20, 20, 20))
        f = get_font(int(h * 0.28), bold=True)
        draw.text((cx, cy), "NETFLIX", fill=red, font=f, anchor="mm")

    # ── 16. Nike ───────────────────────────────────────────
    elif brand == "Nike":
        color = (255, 255, 255) if is_dark_mode else ((15, 15, 15) if not is_mono else (40, 40, 40))
        swoosh_pts = [
            (cx - int(w * 0.38), cy + int(h * 0.02)),
            (cx - int(w * 0.15), cy + int(h * 0.24)),
            (cx + int(w * 0.12), cy + int(h * 0.12)),
            (cx + int(w * 0.38), cy - int(h * 0.22)),
            (cx + int(w * 0.02), cy + int(h * 0.02)),
            (cx - int(w * 0.18), cy + int(h * 0.08)),
        ]
        draw.polygon(swoosh_pts, fill=color)

    # ── 17. Oracle ─────────────────────────────────────────
    elif brand == "Oracle":
        red = (199, 70, 52) if not is_mono else (40, 40, 40)
        capsule = [cx - int(w * 0.40), cy - int(h * 0.20), cx + int(w * 0.40), cy + int(h * 0.20)]
        draw.rounded_rectangle(capsule, radius=int(h * 0.18), fill=red)
        f = get_font(int(h * 0.20), bold=True)
        draw.text((cx, cy), "ORACLE", fill=(255, 255, 255), font=f, anchor="mm")

    # ── 18. PayPal ─────────────────────────────────────────
    elif brand == "PayPal":
        dark_blue = (0, 48, 135) if not is_mono else (40, 40, 40)
        light_blue = (0, 121, 193) if not is_mono else (80, 80, 80)
        f = get_font(int(h * 0.34), bold=True)
        draw.text((cx - int(w * 0.24), cy), "P", fill=dark_blue, font=f, anchor="mm")
        draw.text((cx - int(w * 0.15), cy + int(h * 0.04)), "P", fill=light_blue, font=f, anchor="mm")
        f2 = get_font(int(h * 0.22), bold=True)
        draw.text((cx + int(w * 0.02), cy), "PayPal", fill=dark_blue, font=f2, anchor="lm")

    # ── 19. Pepsi ──────────────────────────────────────────
    elif brand == "Pepsi":
        r = int(min(w, h) * 0.34)
        box = [cx - r, cy - r, cx + r, cy + r]
        red = (235, 25, 51) if not is_mono else (60, 60, 60)
        blue = (0, 94, 184) if not is_mono else (30, 30, 30)
        draw.chord(box, start=180, end=360, fill=red)
        draw.chord(box, start=0, end=180, fill=blue)
        draw.arc([cx - r, cy - int(r * 0.4), cx + r, cy + int(r * 0.4)], start=0, end=180, fill=(255, 255, 255), width=int(r * 0.28))

    # ── 20. Salesforce ─────────────────────────────────────
    elif brand == "Salesforce":
        cyan = (0, 161, 224) if not is_mono else (50, 50, 50)
        draw.ellipse([cx - int(w * 0.28), cy - int(h * 0.12), cx - int(w * 0.02), cy + int(h * 0.18)], fill=cyan)
        draw.ellipse([cx - int(w * 0.12), cy - int(h * 0.24), cx + int(w * 0.16), cy + int(h * 0.16)], fill=cyan)
        draw.ellipse([cx + int(w * 0.04), cy - int(h * 0.14), cx + int(w * 0.30), cy + int(h * 0.18)], fill=cyan)
        f = get_font(int(h * 0.14), bold=True)
        draw.text((cx, cy + 2), "salesforce", fill=(255, 255, 255), font=f, anchor="mm")

    # ── 21. Samsung ────────────────────────────────────────
    elif brand == "Samsung":
        blue = (20, 40, 160) if not is_mono else (40, 40, 40)
        oval = [cx - int(w * 0.42), cy - int(h * 0.24), cx + int(w * 0.42), cy + int(h * 0.24)]
        draw.ellipse(oval, fill=blue)
        f = get_font(int(h * 0.18), bold=True)
        draw.text((cx, cy), "SAMSUNG", fill=(255, 255, 255), font=f, anchor="mm")

    # ── 22. Slack ──────────────────────────────────────────
    elif brand == "Slack":
        colors = [(236, 178, 46), (46, 182, 125), (54, 197, 240), (224, 30, 90)] if not is_mono else [(50, 50, 50)] * 4
        pw = int(w * 0.04)
        pl = int(w * 0.14)
        draw.rounded_rectangle([cx - pl, cy - int(h * 0.12) - pw, cx + pl, cy - int(h * 0.12) + pw], radius=pw, fill=colors[0])
        draw.rounded_rectangle([cx - int(w * 0.12) - pw, cy - pl, cx - int(w * 0.12) + pw, cy + pl], radius=pw, fill=colors[1])
        draw.rounded_rectangle([cx - pl, cy + int(h * 0.12) - pw, cx + pl, cy + int(h * 0.12) + pw], radius=pw, fill=colors[2])
        draw.rounded_rectangle([cx + int(w * 0.12) - pw, cy - pl, cx + int(w * 0.12) + pw, cy + pl], radius=pw, fill=colors[3])

    # ── 23. Sony ───────────────────────────────────────────
    elif brand == "Sony":
        f = get_font(int(h * 0.32), bold=True)
        tc = (240, 240, 240) if is_dark_mode else (20, 20, 20)
        draw.text((cx, cy), "SONY", fill=tc, font=f, anchor="mm")

    # ── 24. Starbucks ──────────────────────────────────────
    elif brand == "Starbucks":
        green = (0, 112, 74) if not is_mono else (40, 40, 40)
        r = int(min(w, h) * 0.40)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=green)
        draw.ellipse([cx - r + 4, cy - r + 4, cx + r - 4, cy + r - 4], outline=(255, 255, 255), width=int(w * 0.015))
        ir = int(r * 0.65)
        draw.ellipse([cx - ir, cy - ir, cx + ir, cy + ir], fill=green, outline=(255, 255, 255), width=int(w * 0.015))
        draw.polygon([(cx, cy - int(ir * 0.65)), (cx - 4, cy - int(ir * 0.4)), (cx + 4, cy - int(ir * 0.4))], fill=(255, 255, 255))

    # ── 25. Stripe ─────────────────────────────────────────
    elif brand == "Stripe":
        indigo = (99, 91, 255) if not is_mono else (50, 50, 50)
        f = get_font(int(h * 0.28), bold=True)
        draw.text((cx, cy), "stripe", fill=indigo, font=f, anchor="mm")

    # ── 26. Target ─────────────────────────────────────────
    elif brand == "Target":
        red = (204, 0, 0) if not is_mono else (40, 40, 40)
        r = int(min(w, h) * 0.38)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=red)
        draw.ellipse([cx - int(r * 0.66), cy - int(r * 0.66), cx + int(r * 0.66), cy + int(r * 0.66)], fill=(255, 255, 255))
        draw.ellipse([cx - int(r * 0.33), cy - int(r * 0.33), cx + int(r * 0.33), cy + int(r * 0.33)], fill=red)

    # ── 27. Toyota ─────────────────────────────────────────
    elif brand == "Toyota":
        red = (235, 10, 30) if not is_mono else (40, 40, 40)
        oval_w, oval_h = int(w * 0.36), int(h * 0.22)
        draw.ellipse([cx - oval_w, cy - oval_h, cx + oval_w, cy + oval_h], outline=red, width=int(w * 0.025))
        draw.ellipse([cx - int(oval_w * 0.6), cy - int(oval_h * 0.8), cx + int(oval_w * 0.6), cy + int(oval_h * 0.2)], outline=red, width=int(w * 0.025))
        draw.ellipse([cx - int(oval_w * 0.25), cy - oval_h, cx + int(oval_w * 0.25), cy + oval_h], outline=red, width=int(w * 0.025))

    # ── 28. UPS ────────────────────────────────────────────
    elif brand == "UPS":
        brown = (53, 29, 16) if not is_mono else (40, 40, 40)
        gold = (255, 184, 28) if not is_mono else (140, 140, 140)
        shield = [
            (cx, cy - int(h * 0.32)),
            (cx + int(w * 0.30), cy - int(h * 0.22)),
            (cx + int(w * 0.26), cy + int(h * 0.15)),
            (cx, cy + int(h * 0.32)),
            (cx - int(w * 0.26), cy + int(h * 0.15)),
            (cx - int(w * 0.30), cy - int(h * 0.22)),
        ]
        draw.polygon(shield, fill=brown)
        draw.polygon([
            (cx, cy - int(h * 0.32)),
            (cx + int(w * 0.30), cy - int(h * 0.22)),
            (cx + int(w * 0.28), cy - int(h * 0.10)),
            (cx, cy - int(h * 0.18)),
            (cx - int(w * 0.28), cy - int(h * 0.10)),
            (cx - int(w * 0.30), cy - int(h * 0.22)),
        ], fill=gold)
        f = get_font(int(h * 0.18), bold=True)
        draw.text((cx, cy + int(h * 0.05)), "UPS", fill=gold, font=f, anchor="mm")

    # ── 29. Uber ───────────────────────────────────────────
    elif brand == "Uber":
        f = get_font(int(h * 0.30), bold=True)
        tc = (240, 240, 240) if is_dark_mode else (10, 10, 10)
        draw.text((cx, cy), "UBER", fill=tc, font=f, anchor="mm")

    # ── 30. Walmart ────────────────────────────────────────
    elif brand == "Walmart":
        blue = (0, 113, 206) if not is_mono else (40, 40, 40)
        gold = (255, 194, 32) if not is_mono else (120, 120, 120)
        f = get_font(int(h * 0.22), bold=True)
        draw.text((cx - int(w * 0.08), cy), "Walmart", fill=blue, font=f, anchor="mm")
        sx, sy = cx + int(w * 0.24), cy
        spark_r = int(w * 0.08)
        for angle_deg in range(0, 360, 60):
            rad = math.radians(angle_deg)
            x_end = sx + int(spark_r * math.cos(rad))
            y_end = sy + int(spark_r * math.sin(rad))
            draw.line([(sx, sy), (x_end, y_end)], fill=gold, width=int(w * 0.022))


def generate_logo_image(class_label: str, index: int, size=(224, 224)) -> Image.Image:
    """Generate authentic vendor logo image with realistic invoice header context.

    Args:
        class_label: Brand name (e.g. 'Amazon', 'Nike', 'Apple').
        index: Sample index providing deterministic variations.
        size: Target (width, height) tuple (default 224x224).

    Returns:
        PIL Image with 2x supersampled anti-aliased authentic brand mark.
    """
    w, h = size
    scale = 2
    sw, sh = w * scale, h * scale

    variant = index % 10

    # Background variations
    if variant < 5:
        # Crisp off-white document paper
        bg_color = (255, 255, 255) if variant % 2 == 0 else (250, 250, 248)
    elif variant < 8:
        # Dark header banner (e.g., enterprise portal / dark invoice header)
        if class_label in ("Amazon", "Uber", "Netflix"):
            bg_color = (24, 28, 36)
        else:
            primary, _ = BRAND_COLORS.get(class_label, ((25, 25, 25), (255, 255, 255)))
            bg_color = primary
    else:
        # Light gray or warm cream scanned invoice sheet
        bg_color = (244, 244, 242) if variant == 8 else (248, 246, 240)

    img = Image.new("RGB", (sw, sh), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Document context elements: optional header baseline or metadata hint
    if variant % 3 == 0 and bg_color[0] > 200:
        draw.line([(0, sh - 10), (sw, sh - 10)], fill=(225, 225, 225), width=2)
        # Subtle header text nearby
        f_sub = get_font(int(sh * 0.035), bold=False)
        draw.text((int(sw * 0.05), sh - 25), "OFFICIAL COMMERCIAL INVOICE", fill=(180, 180, 180), font=f_sub)

    # Draw authentic brand vector
    draw_brand_vector(draw, class_label, sw, sh, variant=variant)

    # Downsample with Lanczos for smooth anti-aliased edges
    return img.resize((w, h), resample=Image.Resampling.LANCZOS)


def main():
    print("=" * 70)
    print(" GENERATING AUTHENTIC BRAND LOGO & INVOICE DOCUMENT DATASET")
    print("=" * 70)
    print(f" Target Classes : {len(CLASSES)} authentic commercial brands")
    print(f" Train Split    : 150 crops / class")
    print(f" Val Split      : 30 crops / class")
    print("=" * 70)

    num_train_per_class = 150
    num_val_per_class = 30

    import shutil
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    for idx, cls in enumerate(CLASSES, 1):
        cls_train_dir = TRAIN_DIR / cls
        cls_val_dir = VAL_DIR / cls
        cls_train_dir.mkdir(parents=True, exist_ok=True)
        cls_val_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_train_per_class):
            img = generate_logo_image(cls, i)
            img.save(cls_train_dir / f"img_{i:03d}.png")

        for i in range(num_val_per_class):
            img = generate_logo_image(cls, num_train_per_class + i)
            img.save(cls_val_dir / f"val_img_{i:03d}.png")

        print(f" [{idx:02d}/{len(CLASSES)}] Created 180 authentic crops for '{cls}'")

    print("\n" + "=" * 70)
    total_train = len(CLASSES) * num_train_per_class
    total_val = len(CLASSES) * num_val_per_class
    print(f" AUTHENTIC DATASET GENERATION COMPLETE | Train: {total_train:,} | Val: {total_val:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
