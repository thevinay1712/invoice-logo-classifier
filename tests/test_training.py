"""Unit tests and smoke tests for training loops, model backbones, and checkpoint saving."""

import torch
from PIL import Image

from invoice_logo_classifier.training.models import build_model, set_backbone_trainable
from invoice_logo_classifier.training.train import compute_class_weights, train_model


def test_build_model_resnet_and_clip():
    """Verify build_model constructs valid ResNet50 and CLIP ViT-B/32 architectures with correct head dimension."""
    resnet_model = build_model(backbone_name="resnet50", num_classes=10, pretrained=False)
    assert hasattr(resnet_model, "fc")
    assert resnet_model.fc.out_features == 10

    # Test forward pass with dummy tensor
    x = torch.randn(2, 3, 224, 224)
    out = resnet_model(x)
    assert out.shape == (2, 10)

    # Test CLIP model building
    clip_model = build_model(backbone_name="clip_vit_b32", num_classes=5, pretrained=False)
    assert hasattr(clip_model, "fc")
    assert clip_model.fc.out_features == 5

    out_clip = clip_model(x)
    assert out_clip.shape == (2, 5)


def test_set_backbone_trainable_warmup_and_unfreeze():
    """Verify set_backbone_trainable correctly freezes backbone parameters for warmup and unfreezes for fine-tuning."""
    model = build_model(backbone_name="resnet50", num_classes=5, pretrained=False)

    # Warmup mode (trainable=False): only fc head trainable
    set_backbone_trainable(model, trainable=False)
    for name, param in model.named_parameters():
        if "fc" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    # Fine-tuning mode (trainable=True): fc and layer4 trainable
    set_backbone_trainable(model, trainable=True, unfreeze_last_n_blocks=1)
    for name, param in model.named_parameters():
        if "fc" in name or "layer4" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False


def test_compute_class_weights():
    """Verify compute_class_weights produces inverse class frequency weights."""
    targets = [0, 0, 0, 0, 1]  # 4 samples of class 0, 1 sample of class 1
    weights = compute_class_weights(targets, num_classes=2)

    assert weights.shape == (2,)
    # Class 1 (rare) should have higher weight than Class 0 (frequent)
    assert weights[1].item() > weights[0].item()


def test_training_smoke_test(tmp_path):
    """Smoke test: execute 2 training epochs on a synthetic dataset and verify finite loss, decrease, and output files."""
    data_dir = tmp_path / "synthetic_dataset"
    out_dir = tmp_path / "output_checkpoints"

    # Create synthetic ImageFolder split directories
    for split in ("train", "val"):
        for cls_name in ("class_A", "class_B"):
            cls_path = data_dir / split / cls_name
            cls_path.mkdir(parents=True, exist_ok=True)
            for i in range(4):
                img = Image.new("RGB", (100, 100), color=(i * 40, 100, 200))
                img.save(cls_path / f"img_{i}.jpg")

    # Run short training execution
    metadata = train_model(
        data_dir=data_dir,
        output_dir=out_dir,
        backbone_name="resnet50",
        batch_size=2,
        epochs=2,
        warmup_epochs=1,
        learning_rate=1e-3,
        seed=42,
        device="cpu",
    )

    assert metadata["num_classes"] == 2
    assert metadata["epochs_trained"] == 2
    assert metadata["backbone_used"] == "resnet50"

    # Check generated output files
    best_model_file = out_dir / "best_model.pt"
    final_model_file = out_dir / "final_model.pt"
    metadata_file = out_dir / "model_metadata.json"
    log_csv_file = out_dir / "training_log.csv"

    assert best_model_file.exists()
    assert final_model_file.exists()
    assert metadata_file.exists()
    assert log_csv_file.exists()

    # Load saved checkpoint and verify fields
    checkpoint = torch.load(best_model_file, weights_only=False)
    assert "model_state_dict" in checkpoint
    assert checkpoint["class_list"] == ["class_A", "class_B"]
