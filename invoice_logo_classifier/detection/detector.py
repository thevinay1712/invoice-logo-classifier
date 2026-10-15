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
            # Top-left corner box (most common logo location)
            boxes.append(
                BBox(xmin=0, ymin=0, xmax=mid_w, ymax=header_h, confidence=0.9, label="top_left")
            )
            # Top-right corner box
            boxes.append(
                BBox(xmin=mid_w, ymin=0, xmax=w, ymax=header_h, confidence=0.85, label="top_right")
            )
            # Full header strip
            boxes.append(
                BBox(xmin=0, ymin=0, xmax=w, ymax=header_h, confidence=0.75, label="full_header")
            )
        else:
            boxes.append(
                BBox(xmin=0, ymin=0, xmax=w, ymax=header_h, confidence=0.8, label="full_header")
            )

        return boxes


class TextDensityRefiner(LogoDetector):
    """Refines candidate header regions using pytesseract OCR bounding box data to find low-text graphic areas."""

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
        """Detect and refine candidate logo bounding boxes based on text box positions."""
        candidate_boxes = self.base_detector.detect(page_image)
        if not candidate_boxes:
            return candidate_boxes

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

            if not text_boxes:
                return candidate_boxes

            refined_boxes = []
            # For top-left sub-region, find non-text / low-text sub-area
            for box in candidate_boxes:
                t_in_box = [
                    tb
                    for tb in text_boxes
                    if tb[0] >= box.xmin and tb[1] >= box.ymin and tb[2] <= box.xmax and tb[3] <= box.ymax
                ]
                if not t_in_box:
                    # Low text density region found -> bump confidence
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

            return refined_boxes

        except Exception as ocr_err:  # noqa: BLE001 – Tesseract optional; any failure falls back to base candidates
            logger.debug(f"Pytesseract text density refinement skipped ({ocr_err}). Returning base candidates.")
            return candidate_boxes


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
