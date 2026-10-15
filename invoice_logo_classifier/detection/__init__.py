"""Detection module: localizes logo region bounding boxes on page images."""

from invoice_logo_classifier.detection.detector import (
    BBox,
    HeuristicHeaderDetector,
    LogoDetector,
    TextDensityRefiner,
    compute_iou,
    crop,
    visualize_detections,
)

__all__ = [
    "BBox",
    "HeuristicHeaderDetector",
    "LogoDetector",
    "TextDensityRefiner",
    "compute_iou",
    "crop",
    "visualize_detections",
]
