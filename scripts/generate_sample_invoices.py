"""Generate realistic sample invoice images for 1-click live testing in the web dashboard."""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_synthetic_dataset import generate_logo_image

OUTPUT_DIR = Path("invoice_logo_classifier/api/static/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_styled_invoice(path: Path, brand: str, invoice_num: str, bill_to_name: str, items: list[tuple], subtotal: str, tax: str, total: str):
    img = Image.new("RGB", (800, 1050), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header container
    draw.rectangle([40, 30, 760, 180], fill=(248, 249, 250), outline=(226, 232, 240), width=1)

    # Paste real brand logo image matching trained neural network features
    logo_crop = generate_logo_image(brand, index=0, size=(160, 160))
    img.paste(logo_crop, (50, 30))

    # Header details
    draw.text((280, 55), f"{brand.upper()} ENTERPRISE SERVICES", fill=(15, 23, 42))
    draw.text((280, 78), "OFFICIAL COMMERCIAL INVOICE", fill=(71, 85, 105))
    draw.text((280, 100), f"Invoice Number: {invoice_num}", fill=(100, 116, 139))
    draw.text((280, 120), "Date: October 22, 2026 | Status: Authorized", fill=(100, 116, 139))

    # Bill To
    draw.rectangle([40, 195, 760, 275], fill=(241, 245, 249))
    draw.text((60, 210), f"BILL TO: {bill_to_name}", fill=(30, 41, 59))
    draw.text((60, 230), "100 Innovation Way, Suite 400, San Francisco, CA 94105", fill=(71, 85, 105))
    draw.text((60, 250), "Payment Terms: Net 30 | PO Number: PO-88301", fill=(100, 116, 139))

    # Line Items Header
    draw.rectangle([40, 295, 760, 330], fill=(30, 41, 59))
    draw.text((60, 305), "ITEM DESCRIPTION", fill=(255, 255, 255))
    draw.text((450, 305), "QTY", fill=(255, 255, 255))
    draw.text((540, 305), "UNIT PRICE", fill=(255, 255, 255))
    draw.text((660, 305), "AMOUNT", fill=(255, 255, 255))

    y = 345
    for desc, qty, price, amt in items:
        draw.text((60, y), desc, fill=(30, 41, 59))
        draw.text((460, y), qty, fill=(30, 41, 59))
        draw.text((550, y), price, fill=(30, 41, 59))
        draw.text((670, y), amt, fill=(30, 41, 59))
        draw.line([(40, y + 25), (760, y + 25)], fill=(226, 232, 240), width=1)
        y += 45

    # Totals
    draw.rectangle([480, y + 20, 760, y + 140], fill=(248, 249, 250), outline=(226, 232, 240))
    draw.text((500, y + 35), "Subtotal:", fill=(71, 85, 105))
    draw.text((680, y + 35), subtotal, fill=(15, 23, 42))
    draw.text((500, y + 65), "Tax (8.25%):", fill=(71, 85, 105))
    draw.text((680, y + 65), tax, fill=(15, 23, 42))
    draw.line([(500, y + 95), (740, y + 95)], fill=(203, 213, 225))
    draw.text((500, y + 105), "TOTAL DUE:", fill=(15, 23, 42))
    draw.text((670, y + 105), total, fill=(16, 185, 129))

    img.save(path)


def main():
    print("Generating demo sample invoice receipt images with brand visual signatures...")

    # 1. Adidas
    draw_styled_invoice(
        OUTPUT_DIR / "adidas_receipt.png",
        brand="Adidas",
        invoice_num="ADI-2026-99214",
        bill_to_name="Enterprise Operations Corp.",
        items=[
            ("Adidas Ultraboost Light Running Shoes - Core Black", "10", "$190.00", "$1,900.00"),
            ("Adidas Adicolor Classics Trefoil Hooded Sweatshirt - M", "15", "$65.00", "$975.00"),
            ("Adidas Tiro 23 League Training Track Pants - Black", "12", "$50.00", "$600.00"),
            ("Bulk Team Apparel Freight & Express Handling", "1", "$85.00", "$85.00"),
        ],
        subtotal="$3,560.00",
        tax="$293.70",
        total="$3,853.70",
    )

    # 2. Amazon
    draw_styled_invoice(
        OUTPUT_DIR / "amazon_invoice.png",
        brand="Amazon",
        invoice_num="AMZ-2026-881920",
        bill_to_name="Cloud Scale Systems, Inc.",
        items=[
            ("Amazon Elastic Compute Cloud (EC2) - Dedicated Host Instances", "720 Hrs", "$1.20/hr", "$864.00"),
            ("Amazon Simple Storage Service (S3) - Standard Storage Pool", "25,000 GB", "$0.023/GB", "$575.00"),
            ("Amazon Relational Database Service (RDS) Multi-AZ Cluster", "720 Hrs", "$1.45/hr", "$1,044.00"),
            ("AWS Premium Support Business Level Contract", "1 Month", "Flat Rate", "$450.00"),
        ],
        subtotal="$2,933.00",
        tax="$241.97",
        total="$3,174.97",
    )

    # 3. Nike
    draw_styled_invoice(
        OUTPUT_DIR / "nike_receipt.png",
        brand="Nike",
        invoice_num="NKE-2026-44021",
        bill_to_name="Athletic Retailers Consortium",
        items=[
            ("Nike Air Max 90 Classic Edition - White/Black (Size 10.5)", "12", "$140.00", "$1,680.00"),
            ("Nike Dri-FIT Legend Fitness Training Tops - L", "30", "$30.00", "$900.00"),
            ("Nike Sportswear Club Fleece Pullover Hoodies", "15", "$60.00", "$900.00"),
            ("Nike Elite Cushioned Basketball Crew Socks (Bulk)", "40", "$16.00", "$640.00"),
        ],
        subtotal="$4,120.00",
        tax="$339.90",
        total="$4,459.90",
    )

    # 4. Google
    draw_styled_invoice(
        OUTPUT_DIR / "google_invoice.png",
        brand="Google",
        invoice_num="GOOG-2026-77391",
        bill_to_name="Alpha Dynamics Innovations",
        items=[
            ("Google Cloud Platform - Compute Engine Instances", "720 Hrs", "$1.15/hr", "$828.00"),
            ("Google Workspace Enterprise Plus - 50 User Licenses", "50", "$30.00/mo", "$1,500.00"),
            ("Google BigQuery Analysis & High-Throughput Storage", "15,000 GB", "$0.02/GB", "$300.00"),
            ("Google Cloud Premier Support Agreement", "1 Month", "Flat Rate", "$500.00"),
        ],
        subtotal="$3,128.00",
        tax="$258.06",
        total="$3,386.06",
    )

    # 4. Unknown Vendor (Apex Logistics) for Tier 3 review queue demo
    img_apex = Image.new("RGB", (800, 1050), color=(255, 255, 255))
    draw_apex = ImageDraw.Draw(img_apex)
    draw_apex.rectangle([40, 30, 760, 180], fill=(248, 249, 250), outline=(226, 232, 240), width=1)
    draw_apex.rectangle([60, 55, 260, 155], fill=(79, 70, 229))
    draw_apex.text((80, 95), "APEX LOGISTICS", fill=(255, 255, 255))
    draw_apex.text((280, 55), "APEX LOGISTICS FREIGHT CORP.", fill=(15, 23, 42))
    draw_apex.text((280, 78), "FREIGHT SHIPPING STATEMENT", fill=(71, 85, 105))
    draw_apex.text((280, 100), "Invoice #: APX-2026-00412", fill=(100, 116, 139))
    draw_apex.rectangle([40, 195, 760, 275], fill=(241, 245, 249))
    draw_apex.text((60, 210), "BILL TO: Global Logistics Hub LLC", fill=(30, 41, 59))
    draw_apex.rectangle([40, 295, 760, 330], fill=(30, 41, 59))
    draw_apex.text((60, 305), "CARRIER SERVICE", fill=(255, 255, 255))
    draw_apex.text((660, 305), "AMOUNT", fill=(255, 255, 255))
    draw_apex.text((60, 350), "Interstate Freight Transport & Delivery", fill=(30, 41, 59))
    draw_apex.text((660, 350), "$850.00", fill=(30, 41, 59))
    draw_apex.text((500, 450), "TOTAL DUE: $850.00", fill=(16, 185, 129))
    img_apex.save(OUTPUT_DIR / "apex_logistics_invoice.png")

    # 5. Non-Logo Document (Address block and barcode only, no vendor logo)
    img_generic = Image.new("RGB", (800, 1050), color=(255, 255, 255))
    draw_gen = ImageDraw.Draw(img_generic)
    draw_gen.rectangle([40, 30, 760, 180], fill=(250, 250, 250), outline=(226, 232, 240), width=1)
    draw_gen.text((60, 55), "COMMERCIAL BILL OF LADING / FREIGHT RECEIPT", fill=(15, 23, 42))
    draw_gen.text((60, 80), "ISSUED BY LOCAL WAREHOUSE DEPOT #4", fill=(71, 85, 105))
    draw_gen.text((60, 105), "Tracking Number: *TRK-9901-2026-X*", fill=(100, 116, 139))
    draw_gen.text((60, 130), "Date: October 22, 2026 | No Corporate Logo Attached", fill=(100, 116, 139))
    for x in range(520, 720, 6):
        w = 3 if (x % 12 == 0) else 1
        draw_gen.line([(x, 55), (x, 150)], fill=(20, 20, 20), width=w)
    draw_gen.rectangle([40, 195, 760, 275], fill=(245, 245, 245))
    draw_gen.text((60, 210), "CONSIGNEE: General Supply Distribution Co.", fill=(30, 41, 59))
    draw_gen.text((500, 450), "TOTAL DUE: $340.00", fill=(16, 185, 129))
    img_generic.save(OUTPUT_DIR / "generic_freight_receipt.png")

    print(f"Sample invoices generated successfully in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
