"""Model architecture factory for closed-set logo classifier backbones."""

import logging

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

logger = logging.getLogger(__name__)


class CLIPClassifier(nn.Module):
    """Wrapper around open_clip vision encoder with a trainable classification head."""

    def __init__(self, clip_model: nn.Module, num_classes: int):
        super().__init__()
        self.visual = clip_model.visual

        # Determine feature embedding dimension
        if hasattr(self.visual, "output_dim"):
            embed_dim = self.visual.output_dim
        elif hasattr(self.visual, "proj") and self.visual.proj is not None:
            embed_dim = self.visual.proj.shape[1]
        else:
            embed_dim = 512

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract vision features
        features = self.visual(x)
        if isinstance(features, tuple):
            features = features[0]
        return self.fc(features)


def build_model(
    backbone_name: str = "resnet50",
    num_classes: int = 50,
    pretrained: bool = True,
) -> nn.Module:
    """Build classification model with specified backbone and classification head.

    Args:
        backbone_name: Name of backbone ('resnet50' or 'clip_vit_b32').
        num_classes: Number of output logo classes.
        pretrained: Whether to load ImageNet / CLIP pretrained weights.

    Returns:
        torch.nn.Module with replaced classification head.
    """
    normalized_name = backbone_name.lower().replace("-", "_")

    if "resnet" in normalized_name:
        logger.info(f"Building ResNet50 backbone (pretrained={pretrained}, num_classes={num_classes})...")
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.backbone_name = "resnet50"
        return model

    elif "clip" in normalized_name:
        logger.info(f"Building CLIP ViT-B/32 backbone (pretrained={pretrained}, num_classes={num_classes})...")
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is required for CLIP backbones. Install via `pip install open-clip-torch`."
            )

        pretrained_tag = "laion2b_s34b_b79k" if pretrained else None
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=pretrained_tag
        )
        model = CLIPClassifier(clip_model, num_classes=num_classes)
        model.backbone_name = "clip_vit_b32"
        return model

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. Supported backbones: 'resnet50', 'clip_vit_b32'."
        )


def set_backbone_trainable(
    model: nn.Module,
    trainable: bool = True,
    unfreeze_last_n_blocks: int = 1,
) -> None:
    """Configure parameter freeze/unfreeze state for standard fine-tuning recipes.

    Args:
        model: Target PyTorch model.
        trainable: If False, freeze backbone and train only head (warmup mode).
                   If True, unfreeze head and last N backbone blocks for fine-tuning.
        unfreeze_last_n_blocks: Number of trailing backbone blocks/layers to unfreeze when trainable=True.
                                For ResNet50: 1 = layer4+fc, 2 = layer3+layer4+fc, etc.
    """
    backbone_type = getattr(model, "backbone_name", "resnet50")

    if not trainable:
        # Warmup phase: freeze all parameters except final classification head ('fc')
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        # Fine-tuning phase: unfreeze final head and last N backbone blocks
        if "resnet" in backbone_type:
            # ResNet50 layer structure: layer1, layer2, layer3, layer4, fc
            # unfreeze_last_n_blocks=1 -> [layer4, fc]
            # unfreeze_last_n_blocks=2 -> [layer3, layer4, fc]
            num_backbone_layers = 4
            start_layer = max(1, num_backbone_layers - unfreeze_last_n_blocks + 1)
            target_layers = [f"layer{i}" for i in range(start_layer, num_backbone_layers + 1)]
            target_layers.append("fc")
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in target_layers):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # CLIP / default: unfreeze all parameters for full fine-tuning
            for param in model.parameters():
                param.requires_grad = True
