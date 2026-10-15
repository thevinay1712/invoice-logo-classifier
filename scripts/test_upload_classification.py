"""Diagnostic script testing accurate invoice classification with brand-matched logos."""

import sys
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from invoice_logo_classifier.inference import InvoiceLogoClassifierService
from scripts.generate_synthetic_dataset import BRAND_COLORS, generate_logo_image

def create_brand_invoice(path: Path, label: str, doc_id: str) -> Path:
    """Helper to render a clean invoice with the brand's actual visual logo block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (800, 1000), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header border & logo box
    draw.rectangle([40, 40, 760, 180], outline=(210, 210, 210), width=2)
    
    # Generate actual brand logo crop matching training distribution
    logo_crop = generate_logo_image(label, index=0, size=(200, 100))
    img.paste(logo_crop, (60, 60))

    # Text lines
    draw.text((290, 65), f"INVOICE DOCUMENT - {label.upper()}", fill=(0, 0, 0))
    draw.text((290, 95), f"Invoice #: {doc_id}", fill=(120, 120, 120))
    draw.text((290, 115), f"Vendor: {label}", fill=(120, 120, 120))

    for y in range(220, 850, 45):
        draw.line([(40, y), (760, y)], fill=(240, 240, 240), width=1)

    img.save(path)
    return path

def main():
    test_dir = Path("data/test_invoices")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test invoice images for various brands
    test_brands = ["Amazon", "Adidas", "Apple", "DHL", "FedEx", "Google", "Microsoft", "Nike", "Starbucks", "Walmart"]
    
    service = InvoiceLogoClassifierService(closed_set_threshold=0.50)
    
    print("=" * 70)
    print(" BRAND INVOICE CLASSIFICATION VERIFICATION")
    print("=" * 70)
    print(f" Loaded Class List ({len(service.class_list)} classes): {service.class_list[:10]}...")
    print(f" Closed-Set Threshold: {service.closed_set_threshold}\n")
    
    correct_count = 0
    for brand in test_brands:
        inv_path = create_brand_invoice(test_dir / f"{brand.lower()}_inv.png", brand, f"INV-{brand.upper()}-101")
        res = service.classify_document(inv_path)
        
        is_correct = (res.predicted_label == brand and res.status == "matched_trained")
        if is_correct:
            correct_count += 1
            status_str = "SUCCESS"
        else:
            status_str = "FAILED"
            
        print(f"[{status_str}] Vendor: {brand:<12} | Predicted: {str(res.predicted_label):<12} | Status: {res.status:<15} | Score: {res.confidence_or_similarity * 100:.2f}%")
        
    print("=" * 70)
    print(f" Classification Accuracy: {correct_count}/{len(test_brands)} ({correct_count / len(test_brands) * 100:.1f}%)")
    print("=" * 70)

    # Test actual uploaded Amazon invoice
    uploaded_amazon_path = Path(r"C:\Users\Vinay Sinnur\.gemini\antigravity-ide\brain\1e4b5873-5c9d-45fb-b67e-9b88fe97feeb\.user_uploaded\media_1790098594665.png")
    if uploaded_amazon_path.exists():
        res_upload = service.classify_document(uploaded_amazon_path)
        print("\n" + "=" * 70)
        print(" USER UPLOADED AMAZON INVOICE TEST RESULT:")
        print(f" File Path : {uploaded_amazon_path.name}")
        print(f" Predicted : {res_upload.predicted_label}")
        print(f" Status    : {res_upload.status}")
        print(f" Confidence: {res_upload.confidence_or_similarity * 100:.2f}%")
        print("=" * 70)

if __name__ == "__main__":
    main()

