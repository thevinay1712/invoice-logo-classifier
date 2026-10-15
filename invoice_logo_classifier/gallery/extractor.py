"""Logo embedding feature extraction module supporting fine-tuned classifier backbone and CLIP encoders."""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from invoice_logo_classifier.augmentation import get_eval_transforms
from invoice_logo_classifier.config import settings
from invoice_logo_classifier.training.models import build_model

logger = logging.getLogger(__name__)


# Note: The classifier backbone feature extractor was fine-tuned specifically to separate its own 30-40 trained classes.
# A separate general-purpose CLIP encoder may generalize better to unseen vendor logos.
# Phase 15 empirically benchmarks and compares both feature extractors.


class FeatureExtractorBackbone(nn.Module):
    """Wrapper stripping final classification head to return raw feature embeddings."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if hasattr(self.base_model, "visual"):
            # CLIPClassifier architecture
            features = self.base_model.visual(x)
            if isinstance(features, tuple):
                features = features[0]
            return features
        elif hasattr(self.base_model, "fc"):
            # ResNet architecture: forward up to average pooling prior to self.base_model.fc
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)

            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)

            x = self.base_model.avgpool(x)
            features = torch.flatten(x, 1)
            return features
        else:
            raise ValueError(f"Unsupported base model architecture: {type(self.base_model)}")


class LogoEmbeddingExtractor:
    """Extracts normalized L2 embedding feature vectors from logo images."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        use_clip_embedding: bool = False,
        device: str | None = None,
    ):
        """Initialize embedding extractor.

        Args:
            model_path: Optional path to trained classifier checkpoint.
            use_clip_embedding: If True, uses pretrained OpenCLIP encoder.
            device: 'cuda' or 'cpu'.
        """
        self.use_clip_embedding = use_clip_embedding
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.eval_transform = get_eval_transforms(image_size=settings.image_size)

        if self.use_clip_embedding:
            logger.info("Initializing OpenCLIP ViT-B/32 general-purpose feature extractor...")
            try:
                import open_clip

                clip_model, _, _ = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k"
                )
                self.model = clip_model.visual.to(self.device).eval()
                self.embed_dim = getattr(clip_model.visual, "output_dim", 512)
            except ImportError:
                raise ImportError(
                    "open_clip_torch required for CLIP embeddings. Install via `pip install open-clip-torch`."
                )
        else:
            logger.info("Initializing fine-tuned classifier backbone feature extractor...")
            target_model_path = (
                Path(model_path) if model_path else settings.model_dir / "best_model.pt"
            )

            if target_model_path.exists():
                logger.info(f"Loading trained model checkpoint from '{target_model_path}'...")
                checkpoint = torch.load(target_model_path, weights_only=False, map_location=self.device)
                class_list = checkpoint.get("class_list", ["default"] * settings.num_classes)
                backbone_name = checkpoint.get("backbone_name", "resnet50")
                base_model = build_model(
                    backbone_name=backbone_name, num_classes=len(class_list), pretrained=False
                )
                base_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                logger.warning(
                    f"Trained model checkpoint not found at '{target_model_path}'. Using default pretrained ResNet50 backbone."
                )
                base_model = build_model(
                    backbone_name="resnet50", num_classes=settings.num_classes, pretrained=True
                )

            feature_model = FeatureExtractorBackbone(base_model)
            self.model = feature_model.to(self.device).eval()
            if hasattr(base_model, "fc"):
                self.embed_dim = base_model.fc.in_features
            else:
                self.embed_dim = settings.embedding_dim

    def extract(self, logo_image: Image.Image) -> np.ndarray:
        """Extract 1D L2-normalized float32 embedding vector from a logo crop PIL image.

        Args:
            logo_image: Input PIL crop image.

        Returns:
            Normalized float32 numpy array of shape (embed_dim,).
        """
        img_rgb = logo_image.convert("RGB")
        tensor = self.eval_transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)
            if isinstance(feat, tuple):
                feat = feat[0]
            feat = feat.squeeze(0).cpu().numpy().astype(np.float32)

        # L2 normalize vector for cosine similarity matching
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        return feat
