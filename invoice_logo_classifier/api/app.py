import base64
import csv
import io
import logging
import tempfile
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
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

# Enable CORS for browser dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static asset directory for dashboard and samples
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# In-memory storage for recently processed batches (ZIP and CSV downloads)
_batch_cache: dict[str, dict[str, Any]] = {}


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
    candidates: list[dict[str, Any]] = Field(default_factory=list, description="Evaluated candidate regions.")
    crop_image_base64: str | None = Field(None, description="Base64 encoded PNG data URI of cropped logo mark.")
    warnings: list[str] = Field(default_factory=list, description="Ingestion or evaluation warnings.")


class BatchClassificationItem(BaseModel):
    filename: str = Field(..., description="Uploaded invoice document filename.")
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
    candidates: list[dict[str, Any]] = Field(default_factory=list, description="Evaluated candidate regions.")
    crop_image_base64: str | None = Field(None, description="Base64 encoded PNG data URI of cropped logo mark.")
    warnings: list[str] = Field(default_factory=list, description="Ingestion or evaluation warnings.")


class BatchClassificationResponse(BaseModel):
    batch_id: str = Field("", description="Unique identifier for downloading prepared organized ZIP and manifest CSV.")
    total_files: int = Field(..., description="Total invoice documents received in batch.")
    processed_count: int = Field(..., description="Number of documents successfully processed.")
    total_companies: int = Field(..., description="Number of distinct company groups recognized.")
    company_counts: dict[str, int] = Field(..., description="Count of invoices organized per company.")
    groups: dict[str, list[BatchClassificationItem]] = Field(
        ..., description="Invoices organized and bucketed by company."
    )
    items: list[BatchClassificationItem] = Field(..., description="Flat list of all classified invoices.")


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


def _process_single_upload(contents: bytes, filename: str, service) -> dict[str, Any]:
    """Execute classification on raw document byte stream and extract full evaluation payload."""
    ext = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(contents)
        tmp_path = Path(tmp_file.name)

    try:
        result = service.classify_document(tmp_path, source_ref=filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    candidates_data = []
    for c in result.candidate_results:
        if hasattr(c.bbox, "xmin"):
            b = {
                "xmin": int(c.bbox.xmin),
                "ymin": int(c.bbox.ymin),
                "xmax": int(c.bbox.xmax),
                "ymax": int(c.bbox.ymax),
            }
        elif isinstance(c.bbox, (list, tuple)) and len(c.bbox) >= 4:
            b = {
                "xmin": int(c.bbox[0]),
                "ymin": int(c.bbox[1]),
                "xmax": int(c.bbox[2]),
                "ymax": int(c.bbox[3]),
            }
        else:
            b = {"xmin": 0, "ymin": 0, "xmax": 200, "ymax": 100}
        candidates_data.append(
            {
                "label": c.predicted_label or "Candidate Crop",
                "score": round(float(c.score), 4),
                "status": c.status,
                "bbox": b,
            }
        )

    crop_b64 = None
    if result.crop_image:
        buf = io.BytesIO()
        result.crop_image.save(buf, format="PNG")
        crop_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    return {
        "filename": filename,
        "predicted_label": result.predicted_label,
        "confidence_or_similarity": round(float(result.confidence_or_similarity), 4),
        "status": result.status,
        "match_source": result.match_source,
        "candidate_count": len(result.candidate_results),
        "candidates": candidates_data,
        "crop_image_base64": crop_b64,
        "warnings": result.warnings,
    }


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

    try:
        contents = await file.read()
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file stream: {err}")

    try:
        service = get_classifier_service()
        data = _process_single_upload(contents, filename, service)
    except IngestionError as ing_err:
        raise HTTPException(status_code=400, detail=str(ing_err))
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference error during classification: {err}")

    latency_ms = (time.time() - start_time) * 1000.0

    logger.info(
        f"Classification completed | File: '{filename}' ({ext}) | Status: {data['status']} | "
        f"Label: '{data['predicted_label']}' | Score: {data['confidence_or_similarity']:.4f} | Latency: {latency_ms:.1f}ms"
    )

    return ClassificationResponse(
        predicted_label=data["predicted_label"],
        confidence_or_similarity=data["confidence_or_similarity"],
        status=data["status"],
        match_source=data["match_source"],
        candidate_count=data["candidate_count"],
        candidates=data["candidates"],
        crop_image_base64=data["crop_image_base64"],
        warnings=data["warnings"],
    )


BATCH_CACHE_DIR = settings.base_dir / "batch_cache"
BATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_batch_cache: dict[str, dict[str, Any]] = {}


@app.post("/invoices/classify-batch", response_model=BatchClassificationResponse, tags=["Inference"])
async def classify_batch_endpoint(files: list[UploadFile] = File(...)):  # noqa: B008
    """Batch-classify multiple uploaded invoices and organize them into company buckets."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded in batch.")

    supported_exts = {e.lower() for e in settings.supported_file_extensions}
    service = get_classifier_service()

    batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    items: list[BatchClassificationItem] = []
    groups: dict[str, list[BatchClassificationItem]] = {}
    company_counts: dict[str, int] = {}
    files_raw: list[dict[str, Any]] = []

    zip_buffer = io.BytesIO()
    manifest_rows = [
        ["Filename", "Identified Company", "Confidence (%)", "Status", "Match Source", "Processed Timestamp"]
    ]
    seen_dirs = set()

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            filename = file.filename or "uploaded_invoice"
            clean_filename = Path(filename).name
            ext = Path(filename).suffix.lower()
            contents = b""

            if ext not in supported_exts:
                item = BatchClassificationItem(
                    filename=filename,
                    predicted_label=None,
                    confidence_or_similarity=0.0,
                    status="unsupported_format",
                    match_source="none",
                    candidate_count=0,
                    candidates=[],
                    crop_image_base64=None,
                    warnings=[f"Unsupported file format '{ext}'."],
                )
                company_folder = "Unsupported_Format"
                conf_pct = "0.00"
                status_str = "unsupported_format"
                source_str = "none"
            else:
                try:
                    contents = await file.read()
                    data = _process_single_upload(contents, filename, service)
                    item = BatchClassificationItem(
                        filename=filename,
                        predicted_label=data["predicted_label"],
                        confidence_or_similarity=data["confidence_or_similarity"],
                        status=data["status"],
                        match_source=data["match_source"],
                        candidate_count=data["candidate_count"],
                        candidates=data["candidates"],
                        crop_image_base64=data["crop_image_base64"],
                        warnings=data["warnings"],
                    )
                    if data["status"] in ("matched_trained", "matched_gallery") and data["predicted_label"]:
                        company_folder = data["predicted_label"]
                    else:
                        company_folder = "Needs_Review_Unknown"
                    conf_pct = f"{data['confidence_or_similarity'] * 100:.2f}"
                    status_str = data["status"]
                    source_str = data["match_source"]
                except Exception as err:  # noqa: BLE001
                    logger.error(f"Failed to process batch invoice file '{filename}': {err}")
                    item = BatchClassificationItem(
                        filename=filename,
                        predicted_label=None,
                        confidence_or_similarity=0.0,
                        status="error",
                        match_source="none",
                        candidate_count=0,
                        candidates=[],
                        crop_image_base64=None,
                        warnings=[f"Processing error: {err}"],
                    )
                    company_folder = "Processing_Error"
                    conf_pct = "0.00"
                    status_str = "error"
                    source_str = "none"

            items.append(item)
            files_raw.append({
                "filename": clean_filename,
                "contents": contents,
                "company": company_folder,
                "confidence": conf_pct,
                "status": status_str,
                "source": source_str,
            })

            # Organize into company bucket
            if item.status in ("matched_trained", "matched_gallery") and item.predicted_label:
                group_key = item.predicted_label
            else:
                group_key = "Needs Review / Unknown"

            if group_key not in groups:
                groups[group_key] = []
                company_counts[group_key] = 0

            groups[group_key].append(item)
            company_counts[group_key] += 1

            if contents:
                if company_folder not in seen_dirs:
                    zf.writestr(f"{company_folder}/", "")
                    seen_dirs.add(company_folder)
                zf.writestr(f"{company_folder}/{clean_filename}", contents)

            manifest_rows.append([
                clean_filename,
                item.predicted_label or "Needs Review / Unknown",
                conf_pct,
                item.status,
                item.match_source,
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            ])

        # Write manifest CSV with UTF-8 BOM
        manifest_csv_io = io.StringIO()
        csv.writer(manifest_csv_io, lineterminator="\r\n").writerows(manifest_rows)
        csv_bytes = manifest_csv_io.getvalue().encode("utf-8-sig")
        zf.writestr("batch_classification_manifest.csv", csv_bytes)

    zip_bytes = zip_buffer.getvalue()

    # Cache for instant direct GET download (memory and disk)
    now = time.time()
    for old_id in list(_batch_cache.keys()):
        if now - _batch_cache[old_id].get("timestamp", 0) > 3600:
            _batch_cache.pop(old_id, None)

    _batch_cache[batch_id] = {
        "zip_bytes": zip_bytes,
        "csv_bytes": csv_bytes,
        "timestamp": now,
        "files_raw": files_raw,
    }

    try:
        (BATCH_CACHE_DIR / f"{batch_id}.zip").write_bytes(zip_bytes)
        (BATCH_CACHE_DIR / "latest.zip").write_bytes(zip_bytes)
        (BATCH_CACHE_DIR / f"{batch_id}.csv").write_bytes(csv_bytes)
        (BATCH_CACHE_DIR / "latest.csv").write_bytes(csv_bytes)
    except Exception as io_err:  # noqa: BLE001
        logger.warning(f"Could not persist batch to disk: {io_err}")

    return BatchClassificationResponse(
        batch_id=batch_id,
        total_files=len(files),
        processed_count=len(items),
        total_companies=len(groups),
        company_counts=company_counts,
        groups=groups,
        items=items,
    )


@app.get("/invoices/batch/{batch_id}/zip", tags=["Inference"])
def download_batch_zip(batch_id: str):
    """Download prepared organized ZIP containing invoices sorted by corporate vendor folder."""
    zip_bytes = None
    if batch_id in _batch_cache and "zip_bytes" in _batch_cache[batch_id]:
        zip_bytes = _batch_cache[batch_id]["zip_bytes"]
    elif (BATCH_CACHE_DIR / f"{batch_id}.zip").exists():
        zip_bytes = (BATCH_CACHE_DIR / f"{batch_id}.zip").read_bytes()
    elif (BATCH_CACHE_DIR / "latest.zip").exists():
        zip_bytes = (BATCH_CACHE_DIR / "latest.zip").read_bytes()

    if not zip_bytes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' has expired or was not found. Please upload invoices again.",
        )
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="invoices_organized_by_vendor_{batch_id}.zip"',
            "Content-Length": str(len(zip_bytes)),
        },
    )


@app.get("/invoices/batch/{batch_id}/csv", tags=["Inference"])
def download_batch_csv(batch_id: str):
    """Download UTF-8 BOM CSV manifest for batch classification."""
    csv_bytes = None
    if batch_id in _batch_cache and "csv_bytes" in _batch_cache[batch_id]:
        csv_bytes = _batch_cache[batch_id]["csv_bytes"]
    elif (BATCH_CACHE_DIR / f"{batch_id}.csv").exists():
        csv_bytes = (BATCH_CACHE_DIR / f"{batch_id}.csv").read_bytes()
    elif (BATCH_CACHE_DIR / "latest.csv").exists():
        csv_bytes = (BATCH_CACHE_DIR / "latest.csv").read_bytes()

    if not csv_bytes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' has expired or was not found. Please upload invoices again.",
        )
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="invoices_manifest_{batch_id}.csv"',
            "Content-Length": str(len(csv_bytes)),
        },
    )



@app.post("/invoices/export-batch-zip", tags=["Inference"])
async def export_batch_zip_endpoint(files: list[UploadFile] = File(...)):  # noqa: B008
    """Organize uploaded invoices into company subfolders and download as a structured ZIP archive."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    supported_exts = {e.lower() for e in settings.supported_file_extensions}
    service = get_classifier_service()

    zip_buffer = io.BytesIO()
    manifest_rows = [
        ["Filename", "Identified Company", "Confidence (%)", "Status", "Match Source", "Processed Timestamp"]
    ]
    seen_dirs = set()

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            filename = file.filename or "invoice"
            clean_filename = Path(filename).name
            ext = Path(filename).suffix.lower()
            contents = await file.read()

            if ext not in supported_exts:
                company_folder = "Unsupported_Format"
                conf_pct = "0.00"
                status_str = "unsupported_format"
                source_str = "none"
            else:
                try:
                    data = _process_single_upload(contents, filename, service)
                    if data["status"] in ("matched_trained", "matched_gallery") and data["predicted_label"]:
                        company_folder = data["predicted_label"]
                    else:
                        company_folder = "Needs_Review_Unknown"
                    conf_pct = f"{data['confidence_or_similarity'] * 100:.2f}"
                    status_str = data["status"]
                    source_str = data["match_source"]
                except Exception as err:  # noqa: BLE001
                    logger.error(f"Zip packaging error on '{filename}': {err}")
                    company_folder = "Processing_Error"
                    conf_pct = "0.00"
                    status_str = "error"
                    source_str = "none"

            if company_folder not in seen_dirs:
                zf.writestr(f"{company_folder}/", "")
                seen_dirs.add(company_folder)

            # Store inside company directory within ZIP
            zf.writestr(f"{company_folder}/{clean_filename}", contents)
            manifest_rows.append([
                clean_filename,
                company_folder,
                conf_pct,
                status_str,
                source_str,
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            ])

        # Write manifest CSV at root of ZIP with UTF-8 BOM
        manifest_csv_io = io.StringIO()
        csv.writer(manifest_csv_io, lineterminator="\r\n").writerows(manifest_rows)
        zf.writestr("batch_classification_manifest.csv", manifest_csv_io.getvalue().encode("utf-8-sig"))

    zip_bytes = zip_buffer.getvalue()
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="organized_invoices_by_company.zip"',
            "Content-Length": str(len(zip_bytes)),
        },
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

    # Synchronize batch downloads so exports reflect newly verified corporate name
    try:
        ref_name = Path(item.invoice_source_ref).name if item.invoice_source_ref else ""
        for b_id, b_data in _batch_cache.items():
            files_raw = b_data.get("files_raw", [])
            updated = False
            for entry in files_raw:
                if entry["company"] in ("Needs_Review_Unknown", "Needs Review / Unknown") and (
                    not ref_name or entry["filename"] == ref_name or len(files_raw) == 1
                ):
                    entry["company"] = request.label
                    entry["status"] = "matched_gallery"
                    entry["source"] = "gallery"
                    entry["confidence"] = "100.00"
                    updated = True
            if updated:
                zbuf = io.BytesIO()
                mrows = [["Filename", "Identified Company", "Confidence (%)", "Status", "Match Source", "Processed Timestamp"]]
                sdirs = set()
                with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf_re:
                    for entry in files_raw:
                        cf = entry["company"]
                        fn = entry["filename"]
                        cnt = entry["contents"]
                        if cf not in sdirs:
                            zf_re.writestr(f"{cf}/", "")
                            sdirs.add(cf)
                        zf_re.writestr(f"{cf}/{fn}", cnt)
                        mrows.append([fn, cf, entry["confidence"], entry["status"], entry["source"], time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())])
                    m_io = io.StringIO()
                    csv.writer(m_io, lineterminator="\r\n").writerows(mrows)
                    c_bytes = m_io.getvalue().encode("utf-8-sig")
                    zf_re.writestr("batch_classification_manifest.csv", c_bytes)
                b_data["zip_bytes"] = zbuf.getvalue()
                b_data["csv_bytes"] = c_bytes
                try:
                    (BATCH_CACHE_DIR / f"{b_id}.zip").write_bytes(b_data["zip_bytes"])
                    (BATCH_CACHE_DIR / "latest.zip").write_bytes(b_data["zip_bytes"])
                    (BATCH_CACHE_DIR / f"{b_id}.csv").write_bytes(b_data["csv_bytes"])
                    (BATCH_CACHE_DIR / "latest.csv").write_bytes(b_data["csv_bytes"])
                except Exception:  # noqa: BLE001, S110
                    pass
    except Exception as upd_err:  # noqa: BLE001
        logger.warning(f"Failed to update batch zip with resolved label: {upd_err}")

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


@app.delete("/review-queue/{id}", tags=["Review Queue"])
def delete_review_item(id: int):
    """Delete a review queue item and its crop image."""
    mgr = get_review_queue_manager()
    success = mgr.delete_item(id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Review item #{id} not found.")
    return {"status": "deleted", "id": id, "message": f"Review item #{id} successfully removed."}


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
    return FileResponse(
        static_file,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"},
    )



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
                    "support": int(float(row["support"])),
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
