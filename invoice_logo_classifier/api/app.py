"""FastAPI production service for invoice logo classification and gallery management."""

import csv
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Security,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from invoice_logo_classifier.config import settings
from invoice_logo_classifier.inference import (
    get_classifier_service,
)
from invoice_logo_classifier.ingestion import IngestionError
from invoice_logo_classifier.review import (
    get_review_queue_manager,
)

logger = logging.getLogger(__name__)

# OpenAPI Security Scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """Dependency verifying API key header authentication when configured."""
    if settings.api_key and api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key header 'X-API-Key'.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager loading model and gallery services at startup."""
    logger.info("Initializing InvoiceLogoClassifierService and ReviewQueueManager at API startup...")
    try:
        get_classifier_service()
        get_review_queue_manager()
        logger.info("Startup complete. All model weights and vector galleries loaded.")
    except Exception as err:  # noqa: BLE001 – startup failure logged but service continues in degraded mode
        logger.error(f"Failed to load model services during API startup: {err}")
    yield
    logger.info("FastAPI service shutting down.")


app = FastAPI(
    title="Invoice Logo Classifier API",
    description=r"""
End-to-end service for localizing and classifying vendor logos on invoice documents (PDF, DOCX, and images).

### Status Decision Hierarchy Explanation:
- **`matched_trained`**: The detected logo was recognized by the closed-set deep learning model with high confidence ($\ge \text{threshold}$). Highest reliability prediction.
- **`matched_gallery`**: The closed-set classifier was uncertain, but the logo matched a previously human-verified vendor embedding in the **dynamic gallery**. Medium-high reliability.
- **`unknown`**: Neither classifier nor gallery matched above threshold. Document is automatically routed to the **offline review queue** for human verification.
""",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)],
)


# Pydantic Schemas
class ClassificationResponse(BaseModel):
    predicted_label: str | None = Field(
        None, description="Recognized vendor / brand name string (None if status is 'unknown')."
    )
    confidence_or_similarity: float = Field(
        ..., description="Softmax confidence score or cosine similarity score."
    )
    status: str = Field(
        ...,
        description="Decision outcome: 'matched_trained' (classifier match), 'matched_gallery' (gallery match), or 'unknown'.",
    )
    match_source: str = Field(
        ..., description="Source of match decision: 'classifier', 'gallery', or 'none'."
    )
    candidate_count: int = Field(1, description="Number of logo candidate boxes evaluated.")
    warnings: list[str] = Field(default_factory=list, description="Ingestion or evaluation warnings.")


class ResolveReviewRequest(BaseModel):
    label: str = Field(..., description="Human-verified correct vendor label string.")
    resolution_type: str = Field(
        "new_vendor",
        description="Resolution type: 'new_vendor' (adds to live gallery) or 'classifier_missed_trained_class' (logs to CSV for model retraining).",
    )


class ReviewItemResponse(BaseModel):
    id: int
    invoice_source_ref: str
    predicted_label: str | None
    confidence_or_similarity: float
    crop_image_path: str
    status: str
    resolved_label: str | None
    resolution_type: str | None
    created_at: str
    resolved_at: str | None


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = True
    gallery_loaded: bool = True
    total_gallery_vectors: int = 0
    timestamp: str = ""


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Liveness & readiness health check endpoint."""
    service = get_classifier_service()
    model_loaded = service.model is not None
    gallery_loaded = service.gallery is not None
    total_vectors = service.gallery.total_entries if gallery_loaded else 0

    return HealthResponse(
        status="healthy" if (model_loaded and gallery_loaded) else "degraded",
        model_loaded=model_loaded,
        gallery_loaded=gallery_loaded,
        total_gallery_vectors=total_vectors,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


@app.post("/invoices/classify", response_model=ClassificationResponse, tags=["Inference"])
async def classify_invoice_endpoint(file: UploadFile = File(...)):  # noqa: B008
    """Classify vendor logo from uploaded invoice document (.pdf, .docx, .png, .jpg, .jpeg, .tiff, .bmp)."""
    start_time = time.time()
    filename = file.filename or "uploaded_invoice"
    ext = Path(filename).suffix.lower()

    supported_exts = {e.lower() for e in settings.supported_file_extensions}
    if ext not in supported_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported extensions: {sorted(supported_exts)}",
        )

    # Save uploaded file to temp file
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(contents)
            tmp_path = Path(tmp_file.name)
    except Exception as err:  # noqa: BLE001 – any IO error on file upload wrapped as HTTP 500
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file stream: {err}")

    try:
        service = get_classifier_service()
        result = service.classify_document(tmp_path)
    except IngestionError as ing_err:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(ing_err))
    except Exception as err:  # noqa: BLE001 – unexpected inference errors wrapped as HTTP 500
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Inference error during classification: {err}")
    finally:
        tmp_path.unlink(missing_ok=True)

    latency_ms = (time.time() - start_time) * 1000.0

    # Structured request logging
    logger.info(
        f"Classification completed | File: '{filename}' ({ext}) | Status: {result.status} | "
        f"Label: '{result.predicted_label}' | Score: {result.confidence_or_similarity:.4f} | Latency: {latency_ms:.1f}ms"
    )

    return ClassificationResponse(
        predicted_label=result.predicted_label,
        confidence_or_similarity=result.confidence_or_similarity,
        status=result.status,
        match_source=result.match_source,
        candidate_count=len(result.candidate_results),
        warnings=result.warnings,
    )


@app.get("/review-queue", response_model=list[ReviewItemResponse], tags=["Review Queue"])
def list_review_queue(
    limit: int = Query(50, ge=1, le=200, description="Maximum items to return"),
    offset: int = Query(0, ge=0, description="Offset index for pagination"),
):
    """Retrieve paginated list of pending review queue items."""
    mgr = get_review_queue_manager()
    pending = mgr.list_pending(limit=limit + offset)
    paged = pending[offset : offset + limit]

    return [
        ReviewItemResponse(
            id=item.id,
            invoice_source_ref=item.invoice_source_ref,
            predicted_label=item.predicted_label,
            confidence_or_similarity=item.confidence_or_similarity,
            crop_image_path=str(item.crop_image_path),
            status=item.status,
            resolved_label=item.resolved_label,
            resolution_type=item.resolution_type,
            created_at=item.created_at,
            resolved_at=item.resolved_at,
        )
        for item in paged
    ]


@app.post("/review-queue/{id}/resolve", response_model=ReviewItemResponse, tags=["Review Queue"])
def resolve_review_item(id: int, request: ResolveReviewRequest):
    """Resolve a review queue item and update live gallery or retraining log."""
    mgr = get_review_queue_manager()
    service = get_classifier_service()

    try:
        item = mgr.resolve(
            review_item_id=id,
            label=request.label,
            resolution_type=request.resolution_type,
            gallery=service.gallery,
        )
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:  # noqa: BLE001 – gallery/DB errors on resolve wrapped as HTTP 500
        raise HTTPException(status_code=500, detail=f"Failed to resolve review item #{id}: {err}")

    return ReviewItemResponse(
        id=item.id,
        invoice_source_ref=item.invoice_source_ref,
        predicted_label=item.predicted_label,
        confidence_or_similarity=item.confidence_or_similarity,
        crop_image_path=str(item.crop_image_path),
        status=item.status,
        resolved_label=item.resolved_label,
        resolution_type=item.resolution_type,
        created_at=item.created_at,
        resolved_at=item.resolved_at,
    )


@app.get("/review-queue/{id}/crop", tags=["Review Queue"])
def get_review_item_crop(id: int):
    """Serve crop PNG image for a review queue item."""
    mgr = get_review_queue_manager()
    import sqlite3
    conn = sqlite3.connect(str(mgr.db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT crop_image_path FROM review_queue WHERE id = ?", (id,))
    row = cursor.fetchone()
    conn.close()

    if not row or not Path(row[0]).exists():
        raise HTTPException(status_code=404, detail=f"Crop image for review item #{id} not found.")
    return FileResponse(row[0], media_type="image/png")


@app.get("/dashboard", response_class=FileResponse, tags=["Dashboard"])
@app.get("/", response_class=FileResponse, tags=["Dashboard"])
def get_dashboard_ui():
    """Serve operations web dashboard UI frontend."""
    static_file = Path(__file__).parent / "static" / "index.html"
    if not static_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html static asset file not found.")
    return FileResponse(static_file, media_type="text/html")


@app.get("/classes", tags=["Metadata"])
def list_trained_classes():
    """List trained closed-set classifier classes with Phase 6 evaluation metrics."""
    service = get_classifier_service()
    class_list = service.class_list

    # Try to load evaluation metrics CSV
    metrics_csv = settings.base_dir / "evaluation_results" / "metrics.csv"
    metrics_map = {}
    if metrics_csv.exists():
        with open(metrics_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics_map[row["class_name"]] = {
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1_score": float(row["f1_score"]),
                    "support": int(row["support"]),
                }

    class_data = []
    for cls_name in class_list:
        m = metrics_map.get(cls_name, {"precision": None, "recall": None, "f1_score": None, "support": None})
        class_data.append(
            {
                "class_name": cls_name,
                "metrics": m,
            }
        )

    return {
        "total_classes": len(class_list),
        "classes": class_data,
    }


@app.get("/vendors", tags=["Metadata"])
def list_gallery_vendors():
    """List distinct gallery vendors with their stored embedding vector counts."""
    service = get_classifier_service()
    gallery = service.gallery
    summary = gallery.get_vendor_summary()

    return {
        "total_vectors": gallery.total_entries,
        "unique_vendors": len(summary),
        "vendors": [
            {"vendor_label": label, "embedding_count": count}
            for label, count in sorted(summary.items(), key=lambda x: x[1], reverse=True)
        ],
    }
