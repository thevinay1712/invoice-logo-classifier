"""Logo detection and header refinement module."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Bounding box container with confidence score and label."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float = 1.0
    label: str = "logo"

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Return (xmin, ymin, xmax, ymax) coordinate tuple."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def to_list(self) -> list[int]:
        """Return [xmin, ymin, xmax, ymax] coordinate list."""
        return [self.xmin, self.ymin, self.xmax, self.ymax]


def compute_iou(box1: BBox | Sequence[int], box2: BBox | Sequence[int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    coords1 = box1.to_tuple() if isinstance(box1, BBox) else tuple(box1[:4])
    coords2 = box2.to_tuple() if isinstance(box2, BBox) else tuple(box2[:4])

    x1_inter = max(coords1[0], coords2[0])
    y1_inter = max(coords1[1], coords2[1])
    x2_inter = min(coords1[2], coords2[2])
    y2_inter = min(coords1[3], coords2[3])

    inter_w = max(0, x2_inter - x1_inter)
    inter_h = max(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h

    area1 = (coords1[2] - coords1[0]) * (coords1[3] - coords1[1])
    area2 = (coords2[2] - coords2[0]) * (coords2[3] - coords2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


class LogoDetector(ABC):
    """Abstract base class establishing common logo detection interface."""

    @abstractmethod
    def detect(self, page_image: Image.Image) -> list[BBox]:
        """Detect candidate logo bounding boxes on a page image.

        Args:
            page_image: Input PIL document page image (RGB).

        Returns:
            List of candidate BBox bounding box objects.
        """


class HeuristicHeaderDetector(LogoDetector):
    """Heuristic detector segmenting the top header region of an invoice document."""

    def __init__(self, header_ratio: float = 0.20, split_corners: bool = False):
        """Initialize heuristic header detector.

        Args:
            header_ratio: Fraction of document height reserved for header (default 0.20).
            split_corners: If True, returns separate top-left and top-right corner sub-regions.
        """
        self.header_ratio = header_ratio
        self.split_corners = split_corners

    def detect(self, page_image: Image.Image) -> list[BBox]:
        """Return header candidate bounding boxes."""
        w, h = page_image.size
        header_h = max(10, int(h * self.header_ratio))

        boxes = []

        if self.split_corners:
            mid_w = int(w * 0.5)
            # Standard top-left header logo box (for wider banner brand marks)
            boxes.append(
                BBox(
                    xmin=int(w * 0.04),
                    ymin=int(h * 0.02),
                    xmax=int(w * 0.40),
                    ymax=int(header_h * 0.95),
                    confidence=0.98,
                    label="header_logo_left",
                )
            )
            # Focused compact top-left logo box
            boxes.append(
                BBox(
                    xmin=int(w * 0.04),
                    ymin=int(h * 0.02),
                    xmax=int(w * 0.32),
                    ymax=int(header_h * 0.90),
                    confidence=0.95,
                    label="header_logo_left_compact",
                )
            )
            # Top-left corner box (most common logo location)
            boxes.append(
                BBox(xmin=0, ymin=0, xmax=mid_w, ymax=header_h, confidence=0.85, label="top_left")
            )
            # Top-right corner box (secondary text / metadata area)
            boxes.append(
                BBox(xmin=mid_w, ymin=0, xmax=w, ymax=header_h, confidence=0.65, label="top_right")
            )
            # Full header strip
            boxes.append(
                BBox(xmin=0, ymin=0, xmax=w, ymax=header_h, confidence=0.55, label="full_header")
            )
        else:
            boxes.append(
                BBox(xmin=0, ymin=0, xmax=w, ymax=header_h, confidence=0.8, label="full_header")
            )

        return boxes


def extract_visual_header_boxes(page_image: Image.Image, header_ratio: float = 0.25) -> list[BBox]:
    """Dynamically locate visual logo mark / wordmark clusters in document header using computer vision.

    Uses adaptive thresholding, horizontal line suppression, and morphological bridging to find
    true visual bounding boxes of vendor logos without cutting off text or symbols.
    """
    try:
        import cv2
        import numpy as np

        w, h = page_image.size
        header_h = max(20, int(h * header_ratio))
        gray = np.array(page_image.convert("L"))[:header_h, :]
        margin_y = max(4, int(header_h * 0.03))
        margin_x = max(4, int(w * 0.02))

        sub = gray[margin_y:header_h - margin_y, margin_x:int(w * 0.70)]
        sub_h, sub_w = sub.shape
        if sub_h < 15 or sub_w < 15:
            return []

        # Detect dark or light banner
        is_dark = bool(np.median(sub) < 128)
        if is_dark:
            _, thresh = cv2.threshold(sub, 120, 255, cv2.THRESH_BINARY)
        else:
            _, thresh = cv2.threshold(sub, 235, 255, cv2.THRESH_BINARY_INV)

        # Remove thin full-width horizontal divider lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.35)), 1))
        lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        thresh = cv2.subtract(thresh, lines)

        # Morphological close to bridge adjacent logo letters, emblems & icons
        k_w = max(16, int(w * 0.035))
        k_h = max(8, int(header_h * 0.08))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        visual_boxes = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            # Logos are in the left 70% of header, reasonable dimensions
            if bw > 35 and bh > 15 and bw < w * 0.75 and bh < header_h * 0.90:
                if bw * bh > 500:
                    bx = BBox(
                        xmin=x + margin_x,
                        ymin=y + margin_y,
                        xmax=x + margin_x + bw,
                        ymax=y + margin_y + bh,
                        confidence=1.15,
                        label="visual_contour_logo",
                    )
                    visual_boxes.append(bx)

        # Sort visual boxes left-to-right, then top-to-bottom
        visual_boxes.sort(key=lambda b: (b.ymin, b.xmin))
        return visual_boxes
    except Exception as cv_err:
        logger.debug(f"Visual contour logo detection skipped ({cv_err}).")
        return []


class TextDensityRefiner(LogoDetector):
    """Refines candidate header regions using visual contours and optional OCR text density."""

    def __init__(
        self,
        base_detector: LogoDetector | None = None,
        header_ratio: float = 0.20,
    ):
        """Initialize text density refiner.

        Args:
            base_detector: Base detector generating initial candidate boxes.
            header_ratio: Default header ratio if base_detector is not provided.
        """
        self.base_detector = base_detector or HeuristicHeaderDetector(
            header_ratio=header_ratio, split_corners=True
        )

    def detect(self, page_image: Image.Image) -> list[BBox]:
        """Detect and refine candidate logo bounding boxes based on visual contours and text boxes."""
        candidate_boxes = self.base_detector.detect(page_image)
        visual_boxes = extract_visual_header_boxes(page_image)

        # Try to run pytesseract OCR to extract text bounding boxes inside top header
        try:
            import pytesseract
            from pytesseract import Output

            w, h = page_image.size
            header_h = int(h * 0.25)
            header_crop = page_image.crop((0, 0, w, header_h))

            ocr_data = pytesseract.image_to_data(header_crop, output_type=Output.DICT)
            text_boxes = []
            n_boxes = len(ocr_data.get("text", []))
            for i in range(n_boxes):
                text_str = str(ocr_data["text"][i]).strip()
                if text_str and int(ocr_data["conf"][i]) > 20:
                    bx = ocr_data["left"][i]
                    by = ocr_data["top"][i]
                    bw = ocr_data["width"][i]
                    bh = ocr_data["height"][i]
                    text_boxes.append((bx, by, bx + bw, by + bh))

            if text_boxes:
                refined_boxes = []
                for box in candidate_boxes:
                    t_in_box = [
                        tb
                        for tb in text_boxes
                        if tb[0] >= box.xmin and tb[1] >= box.ymin and tb[2] <= box.xmax and tb[3] <= box.ymax
                    ]
                    if not t_in_box:
                        refined_boxes.append(
                            BBox(
                                xmin=box.xmin,
                                ymin=box.ymin,
                                xmax=box.xmax,
                                ymax=box.ymax,
                                confidence=min(1.0, box.confidence + 0.1),
                                label=f"{box.label}_graphic",
                            )
                        )
                    else:
                        refined_boxes.append(box)
                candidate_boxes = refined_boxes

        except Exception as ocr_err:  # noqa: BLE001 – Tesseract optional; any failure falls back to base candidates
            logger.debug(f"Pytesseract text density refinement skipped ({ocr_err}). Returning base candidates.")

        return visual_boxes + candidate_boxes


def crop(
    page_image: Image.Image,
    bbox: BBox | Sequence[int],
    padding_px: int = 10,
) -> Image.Image:
    """Crop bounding box region from page image with boundary protection and optional padding.

    Args:
        page_image: Source PIL document image.
        bbox: Target BBox object or [xmin, ymin, xmax, ymax] sequence.
        padding_px: Extra pixel padding around box edges.

    Returns:
        Cropped PIL Image.
    """
    coords = bbox.to_tuple() if isinstance(bbox, BBox) else tuple(bbox[:4])
    xmin, ymin, xmax, ymax = coords
    w, h = page_image.size

    # Apply padding and clamp to image dimensions
    xmin_padded = max(0, int(xmin - padding_px))
    ymin_padded = max(0, int(ymin - padding_px))
    xmax_padded = min(w, int(xmax + padding_px))
    ymax_padded = min(h, int(ymax + padding_px))

    if xmax_padded <= xmin_padded or ymax_padded <= ymin_padded:
        return page_image.copy()

    return page_image.crop((xmin_padded, ymin_padded, xmax_padded, ymax_padded))


def visualize_detections(
    page_image: Image.Image,
    bboxes: Sequence[BBox | Sequence[int]],
    outline_color: str = "red",
    width: int = 3,
) -> Image.Image:
    """Draw candidate bounding boxes onto a copy of the page image for manual QA inspection.

    Args:
        page_image: Input document page image.
        bboxes: List of BBox objects or coordinate lists.
        outline_color: Color of drawn box outlines.
        width: Line width in pixels.

    Returns:
        Annotated PIL Image copy.
    """
    canvas = page_image.copy()
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001 – PIL font loading has no specific typed exception
        font = None

    for idx, box in enumerate(bboxes):
        coords = box.to_list() if isinstance(box, BBox) else list(box[:4])
        xmin, ymin, xmax, ymax = coords

        draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_color, width=width)

        label_text = ""
        if isinstance(box, BBox):
            label_text = f"{box.label} ({box.confidence:.2f})"
        else:
            label_text = f"Box #{idx + 1}"

        if label_text:
            text_pos = (xmin + 4, max(0, ymin + 4))
            draw.text(text_pos, label_text, fill=outline_color, font=font)

    return canvas
