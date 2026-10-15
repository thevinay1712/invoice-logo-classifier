"""Script to generate realistic vendor logo dataset for model training with real company class names."""

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Set seed for reproducibility
random.seed(42)

CLASSES = [
    "Amazon", "Apple", "Google", "Microsoft", "DHL", "FedEx", "UPS",
    "Nike", "Adidas", "Walmart", "Target", "Starbucks", "CocaCola",
    "Pepsi", "Ford", "Toyota", "IBM", "Intel", "Samsung", "Sony",
    "Uber", "Airbnb", "Netflix", "Slack", "Salesforce", "Oracle",
    "Cisco", "Adobe", "PayPal", "Stripe"
]

PROCESSED_DIR = Path("data/processed")
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"

# Signature brand color schemes
BRAND_COLORS = {
    "Amazon": ((35, 47, 62), (255, 153, 0)),
    "Apple": ((40, 40, 40), (240, 240, 240)),
    "Google": ((66, 133, 244), (255, 255, 255)),
    "Microsoft": ((0, 164, 239), (255, 185, 0)),
    "DHL": ((212, 5, 17), (255, 204, 0)),
    "FedEx": ((77, 20, 140), (255, 102, 0)),
    "UPS": ((53, 29, 16), (255, 184, 28)),
    "Nike": ((17, 17, 17), (255, 255, 255)),
    "Adidas": ((10, 10, 10), (255, 255, 255)),
    "Walmart": ((0, 113, 206), (255, 194, 32)),
    "Target": ((204, 0, 0), (255, 255, 255)),
    "Starbucks": ((0, 112, 74), (255, 255, 255)),
    "CocaCola": ((244, 0, 9), (255, 255, 255)),
    "Pepsi": ((0, 94, 184), (235, 25, 51)),
    "Ford": ((0, 52, 120), (255, 255, 255)),
    "Toyota": ((235, 10, 30), (255, 255, 255)),
    "IBM": ((15, 98, 254), (255, 255, 255)),
    "Intel": ((0, 104, 181), (255, 255, 255)),
    "Samsung": ((20, 40, 160), (255, 255, 255)),
    "Sony": ((5, 5, 5), (255, 255, 255)),
    "Uber": ((0, 0, 0), (255, 255, 255)),
    "Airbnb": ((255, 90, 95), (255, 255, 255)),
    "Netflix": ((229, 9, 20), (255, 255, 255)),
    "Slack": ((74, 21, 75), (236, 178, 46)),
    "Salesforce": ((0, 161, 224), (255, 255, 255)),
    "Oracle": ((199, 70, 52), (255, 255, 255)),
    "Cisco": ((4, 159, 217), (255, 255, 255)),
    "Adobe": ((255, 0, 0), (255, 255, 255)),
    "PayPal": ((0, 48, 135), (255, 255, 255)),
    "Stripe": ((99, 91, 255), (255, 255, 255)),
}

def get_font(size=24):
    """Load TrueType font if available, fallback to default font."""
    font_names = ["arial.ttf", "arialbd.ttf", "segoeui.ttf", "DejaVuSans.ttf"]
    for f_name in font_names:
        try:
            return ImageFont.truetype(f_name, size)
        except Exception:
            continue
    return ImageFont.load_default()

def draw_brand_icon(draw: ImageDraw.ImageDraw, brand: str, icon_color: tuple, bounds: list[int]):
    """Draw a recognizable brand logo symbol."""
    x1, y1, x2, y2 = bounds
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    if brand == "Adidas":
        # 3 classic diagonal bars / stripes (angled right)
        for k in range(3):
            sx = x1 + 10 + k * 18
            bar_w = 12
            poly = [
                (sx, y2 - 5),
                (sx + bar_w, y2 - 5),
                (sx + bar_w + 14, y1 + 5 + (2 - k) * 8),
                (sx + 14, y1 + 5 + (2 - k) * 8)
            ]
            draw.polygon(poly, fill=icon_color)
            
    elif brand == "Amazon":
        # Draw authentic Amazon logo: 'amazon' or 'amazon.com' text + orange smile arrow underneath
        arrow_color = (255, 153, 0) if icon_color == (15, 15, 15) else icon_color
        # Smile curve arc under text
        draw.arc([x1 + 5, cy + 2, x2 - 15, y2], start=20, end=160, fill=arrow_color, width=4)
        # Arrowhead at right pointing up-right
        draw.polygon([(x2 - 18, y2 - 8), (x2 - 6, y2 - 4), (x2 - 12, y2 - 16)], fill=arrow_color)
        
    elif brand == "Apple":
        # Apple silhouette circle with bite
        r = min(w, h) // 3
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=icon_color)
        draw.ellipse([cx - 4, cy - r - 8, cx + 8, cy - r + 2], fill=icon_color)
        
    elif brand == "Nike":
        # Swoosh curve polygon
        swoosh = [
            (x1 + 5, cy + 5),
            (cx - 10, y2 - 5),
            (x2 - 5, y1 + 5),
            (cx, cy + 5)
        ]
        draw.polygon(swoosh, fill=icon_color)
        
    elif brand == "Microsoft":
        # 4-square grid
        mid_x, mid_y = cx, cy
        draw.rectangle([x1 + 10, y1 + 5, mid_x - 3, mid_y - 3], fill=(242, 80, 34))
        draw.rectangle([mid_x + 3, y1 + 5, x2 - 10, mid_y - 3], fill=(127, 186, 0))
        draw.rectangle([x1 + 10, mid_y + 3, mid_x - 3, y2 - 5], fill=(0, 164, 239))
        draw.rectangle([mid_x + 3, mid_y + 3, x2 - 10, y2 - 5], fill=(255, 185, 0))
        
    elif brand == "Google":
        # 4-color horizontal ribbon
        draw.rectangle([x1 + 5, cy - 6, x1 + 5 + w // 4, cy + 6], fill=(66, 133, 244))
        draw.rectangle([x1 + 5 + w // 4, cy - 6, x1 + 5 + w // 2, cy + 6], fill=(234, 67, 53))
        draw.rectangle([x1 + 5 + w // 2, cy - 6, x1 + 5 + 3 * w // 4, cy + 6], fill=(251, 188, 5))
        draw.rectangle([x1 + 5 + 3 * w // 4, cy - 6, x2 - 5, cy + 6], fill=(52, 168, 83))
        
    elif brand == "Target":
        # Bullseye circles
        r = min(w, h) // 3
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=icon_color, width=4)
        draw.ellipse([cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2], fill=icon_color)
        
    elif brand == "Starbucks":
        # Siren circle frame
        r = min(w, h) // 3
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=icon_color)
        draw.ellipse([cx - r + 4, cy - r + 4, cx + r - 4, cy + r - 4], outline=(255, 255, 255), width=3)
        
    elif brand == "FedEx":
        # Bold block bar
        draw.rectangle([x1 + 5, cy - 10, cx, cy + 10], fill=(77, 20, 140))
        draw.rectangle([cx, cy - 10, x2 - 5, cy + 10], fill=(255, 102, 0))
        
    elif brand == "DHL":
        # Yellow card with red horizontal slash stripes
        draw.rectangle([x1 + 5, cy - 12, x2 - 5, cy + 12], fill=(255, 204, 0))
        for s in range(3):
            draw.line([(x1 + 15 + s * 25, cy - 8), (x1 + 35 + s * 25, cy + 8)], fill=(212, 5, 17), width=4)
            
    elif brand == "UPS":
        # Shield shape outline
        shield = [
            (cx, y1 + 5),
            (x2 - 10, y1 + 15),
            (x2 - 15, cy + 15),
            (cx, y2 - 5),
            (x1 + 15, cy + 15),
            (x1 + 10, y1 + 15)
        ]
        draw.polygon(shield, fill=icon_color)
        
    else:
        # Default brand accent rectangle
        draw.rectangle([x1 + 10, cy - 6, x2 - 10, cy + 6], fill=icon_color)

def generate_logo_image(class_label: str, index: int, size=(224, 224)) -> Image.Image:
    """Generate realistic invoice & thermal receipt logo images on paper & dark banner backgrounds."""
    primary, secondary = BRAND_COLORS.get(class_label, ((40, 80, 160), (255, 255, 255)))
    
    variant = index % 10
    
    # 50% white paper background (Black/Dark logo)
    # 30% dark banner background (Dark navy / black background with white/colored text & logo)
    # 20% solid brand badge / header box
    if variant < 5:
        bg_color = random.choice([(255, 255, 255), (250, 250, 250), (245, 245, 245)])
        icon_color = (15, 15, 15) if class_label != "Amazon" else (255, 153, 0)
        text_color = (15, 15, 15)
        draw_badge = False
    elif variant < 8:
        # Dark mode / dark banner background (e.g. Amazon dark navy #232F3E or dark grey/black)
        bg_color = random.choice([(35, 47, 62), (18, 18, 18), (25, 25, 25)])
        icon_color = (255, 153, 0) if class_label == "Amazon" else (240, 240, 240)
        text_color = (255, 255, 255)
        draw_badge = False
    else:
        bg_color = (255, 255, 255)
        badge_fill = primary
        icon_color = secondary
        text_color = secondary
        draw_badge = True
        
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    font = get_font(size=22)
    
    pad_x = 10 + (index % 4) * 4
    pad_y = 15 + (index % 5) * 4
    
    if draw_badge:
        draw.rectangle([pad_x, pad_y, size[0] - pad_x, size[1] - pad_y], fill=badge_fill)
        icon_bounds = [pad_x + 10, pad_y + 10, size[0] - pad_x - 10, pad_y + 70]
    else:
        icon_bounds = [pad_x + 20, pad_y + 10, size[0] - pad_x - 20, pad_y + 80]
        
    # Draw brand icon mark
    draw_brand_icon(draw, class_label, icon_color, icon_bounds)
    
    # Draw text label below or centered
    text_str = class_label.lower() if (class_label == "Amazon" or (class_label == "Adidas" and index % 2 == 0)) else class_label.upper()
    bbox = font.getbbox(text_str) if hasattr(font, "getbbox") else (0, 0, len(text_str) * 11, 18)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    x_pos = max((size[0] - text_w) // 2, 5)
    y_pos = icon_bounds[3] + 10
    if y_pos + text_h > size[1] - 10:
        y_pos = size[1] - text_h - 10
        
    draw.text((x_pos, y_pos), text_str, fill=text_color, font=font)
    
    # Optional thermal receipt noise/lines at bottom for paper realism
    if variant in (1, 3) and not draw_badge:
        y_line = y_pos + text_h + 15
        if y_line < size[1] - 15:
            draw.line([(pad_x, y_line), (size[0] - pad_x, y_line)], fill=(180, 180, 180), width=1)
            draw.line([(pad_x, y_line + 4), (size[0] - pad_x, y_line + 4)], fill=(180, 180, 180), width=1)
            
    return img

def main():
    print(f"Generating realistic vendor logo dataset for {len(CLASSES)} brand classes...")
    num_train_per_class = 50
    num_val_per_class = 12
    
    import shutil
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
        
    for cls in CLASSES:
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
            
    print(f"Dataset generated successfully in '{PROCESSED_DIR}'. Total classes: {len(CLASSES)}")

if __name__ == "__main__":
    main()
