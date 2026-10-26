"""Unified end-to-end inference engine tying ingestion, detection, classifier, and gallery fallback."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from invoice_logo_classifier.augmentation import get_eval_transforms
from invoice_logo_classifier.config import settings
from invoice_logo_classifier.detection import (
    BBox,
    LogoDetector,
    TextDensityRefiner,
    crop,
)
from invoice_logo_classifier.gallery import LogoEmbeddingExtractor, VendorGallery
from invoice_logo_classifier.ingestion import ingest
from invoice_logo_classifier.training.models import build_model

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    """Classification candidate evaluation for a specific detected logo bounding box."""

    predicted_label: str | None
    score: float
    status: str  # "matched_trained" | "matched_gallery" | "unknown"
    match_source: str  # "classifier" | "gallery" | "none"
    bbox: BBox | list[int]
    crop_image: Image.Image


@dataclass
class ClassificationResult:
    """End-to-end invoice logo classification result container."""

    predicted_label: str | None
    confidence_or_similarity: float
    status: str  # "matched_trained" | "matched_gallery" | "unknown"
    match_source: str  # "classifier" | "gallery" | "none"
    crop_image: Image.Image
    source_path: Path
    candidate_results: list[CandidateResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _forward_with_features(model: nn.Module, tensor: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    """Execute single forward pass returning both classification logits and L2-normalized feature vector."""
    if hasattr(model, "visual"):
        # CLIPClassifier architecture
        feat = model.visual(tensor)
        if isinstance(feat, tuple):
            feat = feat[0]
        logits = model.fc(feat)
    elif hasattr(model, "fc"):
        # ResNet50 architecture
        x = model.conv1(tensor)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = model.fc(feat)
    else:
        logits = model(tensor)
        feat = torch.zeros((tensor.size(0), 512), device=tensor.device)

    feat_np = feat.squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(feat_np)
    if norm > 0:
        feat_np = feat_np / norm

    return logits, feat_np


class InvoiceLogoClassifierService:
    """Persistent classification service loading model, metadata, and gallery once at init."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        gallery_dir: Path | str | None = None,
        closed_set_threshold: float | None = None,
        gallery_threshold: float | None = None,
        detector: LogoDetector | None = None,
        use_clip_embedding: bool = False,
        device: str | None = None,
    ):
        """Initialize classification service and load model weights & gallery once.

        Args:
            model_path: Path to trained PyTorch model checkpoint.
            gallery_dir: Directory containing FAISS index and SQLite db.
            closed_set_threshold: Confidence threshold for closed-set classifier.
            gallery_threshold: Cosine similarity threshold for gallery fallback.
            detector: Logo detector instance (defaults to TextDensityRefiner).
            use_clip_embedding: If True, uses separate CLIP embedding extractor.
            device: 'cuda' or 'cpu'.
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.closed_set_threshold = (
            closed_set_threshold
            if closed_set_threshold is not None
            else settings.closed_set_confidence_threshold
        )
        self.gallery_threshold = (
            gallery_threshold
            if gallery_threshold is not None
            else settings.gallery_similarity_threshold
        )
        self.detector = detector or TextDensityRefiner(header_ratio=0.20)
        self.eval_transform = get_eval_transforms(image_size=settings.image_size)
        self.use_clip_embedding = use_clip_embedding

        # Load trained classifier model
        if model_path:
            target_model_path = Path(model_path)
        else:
            active_dir = settings.get_active_model_dir()
            target_model_path = active_dir / "best_model.pt"
            if not target_model_path.exists():
                target_model_path = active_dir / "best_model.pth"
        if target_model_path.exists():
            logger.info(f"Loading trained classifier weights from '{target_model_path}'...")
            checkpoint = torch.load(target_model_path, weights_only=False, map_location=self.device)
            self.class_list = checkpoint.get("class_list", ["default"] * settings.num_classes)
            backbone_name = checkpoint.get("backbone_name", "resnet50")
            self.model = build_model(
                backbone_name=backbone_name, num_classes=len(self.class_list), pretrained=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.warning(
                f"Trained model checkpoint not found at '{target_model_path}'. Initializing pretrained ResNet50 baseline."
            )
            self.class_list = [f"class_{i}" for i in range(settings.num_classes)]
            self.model = build_model(
                backbone_name="resnet50", num_classes=settings.num_classes, pretrained=True
            )

        self.model.to(self.device).eval()
        model_embed_dim = self.model.fc.in_features if hasattr(self.model, "fc") else settings.embedding_dim

        # Load gallery store
        target_gallery_dir = Path(gallery_dir if gallery_dir else settings.gallery_dir)
        if target_gallery_dir.exists() and (target_gallery_dir / "gallery_index.faiss").exists():
            logger.info(f"Loading vendor gallery from '{target_gallery_dir}'...")
            self.gallery = VendorGallery.load(target_gallery_dir)
        else:
            logger.info(f"Initializing empty VendorGallery instance (dim={model_embed_dim})...")
            self.gallery = VendorGallery(embedding_dim=model_embed_dim)

        if self.use_clip_embedding:
            self.clip_extractor = LogoEmbeddingExtractor(
                use_clip_embedding=True, device=str(self.device)
            )
        else:
            self.clip_extractor = None

        # Initialize multimodal zero-shot brand verification and document clutter rejection
        try:
            import open_clip

            clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self._clip_model = clip_model.to(self.device).eval()
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            prompts = [f"{b} brand logo mark" for b in self.class_list] + [
                "plain invoice document title text invoice number bill to",
                "monochrome document header table lines and address text",
            ]
            tokens = self._clip_tokenizer(prompts).to(self.device)
            with torch.no_grad():
                text_feats = self._clip_model.encode_text(tokens)
                self._clip_text_features = text_feats / text_feats.norm(dim=-1, keepdim=True)
            logger.info("OpenCLIP multimodal brand and clutter verification initialized.")
        except Exception as clip_err:
            logger.warning(f"OpenCLIP zero-shot verification not loaded: {clip_err}")
            self._clip_model = None
            self._clip_text_features = None

    def _eval_clip_zeroshot(self, crop_img: Image.Image) -> tuple[str, float]:
        """Evaluate crop with OpenCLIP zero-shot across brand classes and clutter prompts."""
        if getattr(self, "_clip_model", None) is None or getattr(self, "_clip_text_features", None) is None:
            return "unknown", 0.0
        try:
            t = self._clip_preprocess(crop_img.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self._clip_model.encode_image(t)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                sim = (100.0 * feat @ self._clip_text_features.T).softmax(dim=-1).squeeze(0)
                top_v, top_i = torch.max(sim, dim=0)
                idx = int(top_i.item())
                val = float(top_v.item())
                if idx >= len(self.class_list):
                    return "clutter", val
                return self.class_list[idx], val
        except Exception as e:
            logger.debug(f"OpenCLIP eval failed: {e}")
            return "unknown", 0.0

    def classify_crop(self, crop_img: Image.Image) -> tuple[str | None, float, str, str]:
        """Classify a single cropped logo image following the 3-tier decision order.

        Decision Hierarchy:
            1. Classifier softmax prob >= closed_set_threshold (or confirmed via multimodal zero-shot) -> matched_trained (classifier)
            2. Gallery best match similarity >= gallery_threshold -> matched_gallery (gallery)
            3. Neither cleared threshold -> unknown (none)

        Returns:
            (predicted_label, score, status, match_source)
        """
        tensor = self.eval_transform(crop_img.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, feature_vec = _forward_with_features(self.model, tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            top_prob, top_idx = torch.max(probs, dim=0)
            top_score = float(top_prob.item())
            top_label = self.class_list[int(top_idx.item())]

        # Multimodal zero-shot brand & clutter verification
        clip_label, clip_score = self._eval_clip_zeroshot(crop_img)

        # Reject document title text / address block clutter
        if clip_label == "clutter" and clip_score >= 0.50:
            return "clutter", clip_score, "unknown", "none"

        # Tier 1: Closed-set classifier decision
        is_clip_brand_match = (
            clip_label in self.class_list
            and clip_score >= self.closed_set_threshold
            and clip_score >= 0.65
        )

        if top_score >= self.closed_set_threshold or is_clip_brand_match:
            if is_clip_brand_match and (top_label != clip_label or clip_score > top_score):
                final_label = clip_label
                final_score = clip_score
            else:
                final_label = top_label
                final_score = top_score

            if final_label and final_label.lower().startswith(("_non_logo", "non_logo", "clutter")):
                return final_label, final_score, "unknown", "none"
            return final_label, final_score, "matched_trained", "classifier"

        # Tier 2: Dynamic Gallery Fallback
        if self.use_clip_embedding and self.clip_extractor:
            gallery_vec = self.clip_extractor.extract(crop_img)
        else:
            gallery_vec = feature_vec

        gallery_match = self.gallery.best_match(gallery_vec, threshold=self.gallery_threshold)
        if gallery_match is not None:
            return (
                gallery_match.vendor_label,
                gallery_match.similarity_score,
                "matched_gallery",
                "gallery",
            )

        # Tier 3: Unknown vendor (route to review queue)
        return top_label, top_score, "unknown", "none"

    def classify_document(self, file_path: Path | str, source_ref: str | None = None) -> ClassificationResult:
        """Run end-to-end ingestion, detection, and decision order classification on a document.

        Args:
            file_path: Document file path (.pdf, .docx, .png, .jpg, etc.).

        Returns:
            ClassificationResult dataclass.
        """
        target_path = Path(file_path)
        ingested = ingest(target_path, max_pages=1)
        page_img = ingested.pages[0]

        # Detect candidate logo regions
        candidate_boxes = self.detector.detect(page_img)
        if not candidate_boxes:
            w, h = page_img.size
            candidate_boxes = [BBox(xmin=0, ymin=0, xmax=w, ymax=int(h * 0.20), label="fallback_header")]

        candidate_results = []
        for box in candidate_boxes:
            b_label = getattr(box, "label", "")
            pad = 8 if "visual" in b_label else (4 if "compact" in b_label else 10)
            crop_img = crop(page_img, box, padding_px=pad)
            label, score, status, source = self.classify_crop(crop_img)
            candidate_results.append(
                CandidateResult(
                    predicted_label=label,
                    score=score,
                    status=status,
                    match_source=source,
                    bbox=box,
                    crop_image=crop_img,
                )
            )

        # Select candidate with highest confidence / similarity score
        # Priority order: matched_trained > matched_gallery > unknown > clutter, then highest score
        def _sort_key(c: CandidateResult):
            is_clutter = bool(c.predicted_label and c.predicted_label.lower().startswith(("_non_logo", "non_logo", "clutter")))
            if is_clutter:
                return (0, c.score)
            status_priority = {"matched_trained": 3, "matched_gallery": 2, "unknown": 1}
            box_conf = getattr(c.bbox, "confidence", 1.0) if hasattr(c, "bbox") else 1.0
            return (status_priority.get(c.status, 0), c.score * box_conf)

        candidate_results.sort(key=_sort_key, reverse=True)
        primary = candidate_results[0]

        result = ClassificationResult(
            predicted_label=primary.predicted_label if primary.status != "unknown" else None,
            confidence_or_similarity=round(primary.score, 4),
            status=primary.status,
            match_source=primary.match_source,
            crop_image=primary.crop_image,
            source_path=target_path,
            candidate_results=candidate_results,
            warnings=ingested.warnings,
        )

        # Auto-enqueue for human review if status is unknown
        if primary.status == "unknown":
            try:
                from invoice_logo_classifier.review import enqueue_for_review
                doc_ref = str(source_ref) if source_ref else target_path.name
                enqueue_for_review(result, source_ref=doc_ref)
            except Exception as err:  # noqa: BLE001 – review queue failure must not block classification result
                logger.warning(f"Failed to auto-enqueue unknown invoice for review: {err}")

        return result


# Global singleton service instance
_default_service: InvoiceLogoClassifierService | None = None


def get_classifier_service() -> InvoiceLogoClassifierService:
    """Return or initialize global singleton InvoiceLogoClassifierService instance."""
    global _default_service
    if _default_service is None:
        _default_service = InvoiceLogoClassifierService()
    return _default_service


def reset_classifier_service() -> None:
    """Reset global singleton InvoiceLogoClassifierService instance for test isolation."""
    global _default_service
    _default_service = None


def classify_invoice(
    file_path: Path | str,
    service: InvoiceLogoClassifierService | None = None,
) -> ClassificationResult:
    """Module-level entry point function for end-to-end invoice logo classification.

    Args:
        file_path: Path to target invoice document (.pdf, .docx, .png, .jpg, etc.).
        service: Optional InvoiceLogoClassifierService instance.

    Returns:
        ClassificationResult object.
    """
    srv = service or get_classifier_service()
    return srv.classify_document(file_path)
