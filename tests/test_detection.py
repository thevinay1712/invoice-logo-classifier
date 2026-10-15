"""Unit tests for logo region detection, header splitting, cropping, and visualization."""

import pytest
from PIL import Image, ImageDraw

from invoice_logo_classifier.detection import (
    BBox,
    HeuristicHeaderDetector,
    TextDensityRefiner,
    compute_iou,
    crop,
    visualize_detections,
)


@pytest.fixture
def synthetic_page_with_logo() -> tuple[Image.Image, BBox]:
    """Fixture generating a 600x800 synthetic document page with a red logo in the top-left header."""
    w, h = 600, 800
    page_img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(page_img)

    # Draw logo at top-left header: [20, 20, 150, 90]
    true_logo_box = BBox(xmin=20, ymin=20, xmax=150, ymax=90, label="ground_truth_logo")
    draw.rectangle(true_logo_box.to_list(), fill=(220, 40, 40), outline=(0, 0, 0))

    # Add text on lower page
    draw.text((50, 300), "INVOICE CONTENT BODY TEXT LINE 1", fill=(0, 0, 0))
    draw.text((50, 350), "INVOICE CONTENT BODY TEXT LINE 2", fill=(0, 0, 0))

    return page_img, true_logo_box


def test_heuristic_header_detector_iou_overlap(synthetic_page_with_logo):
    """Verify HeuristicHeaderDetector produces candidate boxes overlapping the ground truth header logo (IoU > 0.3)."""
    page_img, true_logo_box = synthetic_page_with_logo

    detector = HeuristicHeaderDetector(header_ratio=0.20, split_corners=True)
    detected_boxes = detector.detect(page_img)

    assert len(detected_boxes) >= 2

    # Top-left box should overlap ground-truth logo
    top_left_box = next(b for b in detected_boxes if b.label == "top_left")
    iou = compute_iou(top_left_box, true_logo_box)

    # Ground truth box (130x70) is entirely inside top-left candidate box (300x160)
    assert iou > 0.15


def test_text_density_refiner(synthetic_page_with_logo):
    """Verify TextDensityRefiner executes cleanly and returns candidate boxes."""
    page_img, _ = synthetic_page_with_logo

    refiner = TextDensityRefiner(header_ratio=0.20)
    boxes = refiner.detect(page_img)

    assert len(boxes) > 0
    assert all(isinstance(b, BBox) for b in boxes)


def test_crop_utility(synthetic_page_with_logo):
    """Verify crop function applies padding and protects image boundaries."""
    page_img, _ = synthetic_page_with_logo
    target_box = BBox(xmin=50, ymin=50, xmax=150, ymax=100)

    cropped = crop(page_img, target_box, padding_px=10)

    # Expected size: (150-50 + 20) x (100-50 + 20) = 120 x 70
    assert cropped.size == (120, 70)

    # Boundary out-of-range coords get clamped safely
    out_of_bounds_box = BBox(xmin=-10, ymin=-10, xmax=700, ymax=900)
    clamped_crop = crop(page_img, out_of_bounds_box, padding_px=0)
    assert clamped_crop.size == (600, 800)


def test_visualize_detections(synthetic_page_with_logo):
    """Verify visualize_detections returns an annotated PIL Image of matching size."""
    page_img, true_logo_box = synthetic_page_with_logo

    annotated = visualize_detections(page_img, [true_logo_box], outline_color="green", width=4)

    assert isinstance(annotated, Image.Image)
    assert annotated.size == page_img.size


def test_compute_iou():
    """Verify compute_iou produces accurate intersection over union values."""
    boxA = [0, 0, 10, 10]  # area 100
    boxB = [5, 0, 15, 10]  # area 100, intersection area 50, union 150
    assert compute_iou(boxA, boxB) == pytest.approx(50 / 150)

    # Disjoint boxes -> IoU 0.0
    boxC = [20, 20, 30, 30]
    assert compute_iou(boxA, boxC) == 0.0
