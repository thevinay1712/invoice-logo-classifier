"""Global configuration management for Invoice Logo Classifier using pydantic-settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration class storing all tunable parameters."""

    model_config = SettingsConfigDict(
        env_prefix="ILC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base Paths
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data"
    )
    raw_data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw"
    )
    processed_data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed"
    )
    model_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "models"
    )
    gallery_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "gallery_store"
    )
    review_queue_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "review_queue"
    )

    # Hugging Face Dataset Settings
    hf_dataset_name: str = Field(
        default="PodYapolsky/LogoDet-3K",
        description="Hugging Face dataset identifier for LogoDet-3K",
    )
    dataset_cache_dir: Path | None = Field(
        default=None,
        description="Custom directory path for dataset caching (None defaults to Hugging Face standard cache)",
    )
    label_column: str = Field(
        default="company_name",
        description="Dataset column name for brand/company class ID (confirmed via live Hugging Face schema inspection)",
    )
    image_column: str = Field(
        default="image_path",
        description="Dataset column name for PIL image data (confirmed via live Hugging Face schema inspection)",
    )
    bbox_column: str = Field(
        default="bbox",
        description="Dataset column name for bounding box coordinates [xmin, ymin, xmax, ymax] (confirmed via live schema inspection)",
    )
    category_column: str = Field(
        default="industry_name",
        description="Dataset column name for logo industry category string (confirmed via live schema inspection)",
    )
    class_counts_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "class_counts.csv",
        description="Path to export CSV file containing per-class image counts",
    )
    selected_classes_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "selected_classes.json",
        description="Path to JSON file storing final selected company class list",
    )
    min_images_per_class: int = Field(
        default=40,
        description="Minimum number of images per class required for inclusion in training set",
    )
    target_class_count: int | None = Field(
        default=None,
        description="Optional upper limit on number of selected classes (None selects all clearing min_images_per_class)",
    )
    max_images_per_class_train: int = Field(
        default=300,
        description="Maximum ceiling of images per class allowed in the training split to prevent class imbalance",
    )
    dataset_manifest_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed" / "dataset_manifest.json",
        description="Path to JSON file storing final dataset manifest breakdown",
    )

    # Ingestion & Preprocessing Settings
    pdf_dpi: int = Field(default=300, description="DPI for converting PDF pages to images")
    supported_file_extensions: list[str] = Field(
        default=[".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
        description="Supported invoice input document extensions",
    )

    # Image Settings & Augmentation
    image_size: tuple[int, int] = Field(
        default=(224, 224), description="(height, width) for model input images"
    )
    normalize_mean: list[float] = Field(default=[0.485, 0.456, 0.406])
    normalize_std: list[float] = Field(default=[0.229, 0.224, 0.225])

    # Logo Detection Settings
    detection_model_name: str = Field(
        default="yolov8n", description="Object detection model for logo localization"
    )
    detection_confidence_threshold: float = Field(
        default=0.50, description="Minimum confidence for logo region detection"
    )
    detection_nms_threshold: float = Field(
        default=0.45, description="Non-maximum suppression threshold"
    )

    # Classifier Model & Hyperparameters
    classifier_model_name: str = Field(
        default="efficientnet_b0", description="Backbone model for logo classification"
    )
    embedding_dim: int = Field(
        default=512, description="Embedding vector dimension for gallery matching"
    )
    num_classes: int = Field(
        default=50, description="Initial closed-set number of vendor classes"
    )

    # Training Parameters
    batch_size: int = Field(default=32, description="Training batch size")
    learning_rate: float = Field(default=1e-3, description="Initial learning rate")
    weight_decay: float = Field(default=1e-4, description="Optimizer weight decay")
    epochs: int = Field(default=20, description="Number of training epochs")
    num_workers: int = Field(default=4, description="DataLoader worker threads")
    val_split_ratio: float = Field(default=0.15, description="Validation split ratio")
    test_split_ratio: float = Field(default=0.15, description="Test split ratio")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Classification & Gallery Threshold Calibration
    closed_set_confidence_threshold: float = Field(
        default=0.45,
        description="Confidence threshold above which closed-set classification is accepted",
    )
    gallery_similarity_threshold: float = Field(
        default=0.75,
        description="Cosine similarity threshold for gallery embedding match",
    )
    top_k_gallery_matches: int = Field(
        default=5, description="Number of top nearest gallery matches to return"
    )

    # Review Queue & Gallery Updates
    auto_add_to_gallery_threshold: float = Field(
        default=0.92,
        description="Similarity threshold for auto-adding high confidence unassigned samples",
    )

    # API Service Config
    api_host: str = Field(default="0.0.0.0", description="FastAPI server host")
    api_port: int = Field(default=8000, description="FastAPI server port")
    api_debug: bool = Field(default=False, description="Enable FastAPI debug mode")
    api_key: str | None = Field(
        default=None,
        description="Optional API key for header authentication (X-API-Key). If None, auth is disabled.",
    )

    def get_active_model_dir(self) -> Path:
        """Resolve directory containing active model weights, considering ACTIVE_VERSION.json pointer."""
        pointer_file = self.model_dir / "ACTIVE_VERSION.json"
        if pointer_file.exists():
            try:
                import json
                with open(pointer_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                active_ver = data.get("active_version")
                if active_ver:
                    ver_dir = self.model_dir / active_ver
                    if ver_dir.exists():
                        return ver_dir
            except Exception:  # noqa: BLE001, S110 – malformed JSON on startup should fall back silently
                pass
        return self.model_dir


# Instantiation singleton pattern
settings = Settings()
