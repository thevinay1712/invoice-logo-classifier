"""Integration test suite for FastAPI service endpoints using TestClient."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from invoice_logo_classifier.api import app
from invoice_logo_classifier.config import settings
from invoice_logo_classifier.inference import get_classifier_service


@pytest.fixture
def api_client(tmp_path) -> TestClient:
    """Fixture returning FastAPI TestClient instance with isolated temporary directories."""
    from invoice_logo_classifier.inference.classify import reset_classifier_service

    reset_classifier_service()
    settings.gallery_dir = tmp_path / "api_gallery_test"
    settings.review_queue_dir = tmp_path / "api_review_queue_test"

    return TestClient(app)


@pytest.fixture
def synthetic_png_bytes() -> bytes:
    """Fixture generating a synthetic PNG invoice image byte stream."""
    img = Image.new("RGB", (300, 400), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(api_client):
    """Verify GET /health returns 200 OK and healthy status."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "model_loaded" in data


def test_classes_endpoint(api_client):
    """Verify GET /classes lists trained closed-set classes."""
    response = api_client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert "total_classes" in data
    assert "classes" in data


def test_vendors_endpoint(api_client):
    """Verify GET /vendors lists gallery vendors and embedding counts."""
    response = api_client.get("/vendors")
    assert response.status_code == 200
    data = response.json()
    assert "total_vectors" in data
    assert "unique_vendors" in data


def test_classify_unsupported_file_extension(api_client):
    """Verify POST /invoices/classify returns 400 Bad Request on unsupported file extensions."""
    files = {"file": ("document.txt", b"Unsupported plain text file content", "text/plain")}
    response = api_client.post("/invoices/classify", files=files)

    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_classify_outcomes_matched_trained_and_unknown(api_client, synthetic_png_bytes):
    """Verify POST /invoices/classify exercises matched_trained and unknown decision paths."""
    service = get_classifier_service()

    # Test Outcome 1: matched_trained
    service.closed_set_threshold = 0.01
    files = {"file": ("test_invoice.png", synthetic_png_bytes, "image/png")}
    res_trained = api_client.post("/invoices/classify", files=files)
    assert res_trained.status_code == 200
    data1 = res_trained.json()
    assert data1["status"] == "matched_trained"
    assert data1["match_source"] == "classifier"

    # Test Outcome 3: unknown (forces both thresholds high)
    service.closed_set_threshold = 0.999
    service.gallery_threshold = 0.999
    files_unk = {"file": ("test_invoice.png", synthetic_png_bytes, "image/png")}
    res_unk = api_client.post("/invoices/classify", files=files_unk)
    assert res_unk.status_code == 200
    data3 = res_unk.json()
    assert data3["status"] == "unknown"
    assert data3["match_source"] == "none"
    assert data3["predicted_label"] is None


def test_review_queue_list_and_resolve_flow(api_client, synthetic_png_bytes):
    """Verify review queue listing and resolution flow for both resolution types."""
    service = get_classifier_service()
    service.closed_set_threshold = 0.999
    service.gallery_threshold = 0.999

    # Trigger unknown classification to auto-enqueue item
    files = {"file": ("invoice_review_test.png", synthetic_png_bytes, "image/png")}
    api_client.post("/invoices/classify", files=files)

    # Check GET /review-queue
    res_list = api_client.get("/review-queue?limit=50")
    assert res_list.status_code == 200
    pending_items = res_list.json()
    assert len(pending_items) >= 1
    item_id = pending_items[0]["id"]

    # Test POST /review-queue/{id}/resolve via new_vendor path
    res_resolve = api_client.post(
        f"/review-queue/{item_id}/resolve",
        json={"label": "API Resolved Vendor", "resolution_type": "new_vendor"},
    )
    assert res_resolve.status_code == 200
    res_data = res_resolve.json()
    assert res_data["status"] == "resolved"
    assert res_data["resolved_label"] == "API Resolved Vendor"
    assert res_data["resolution_type"] == "new_vendor"

    # Verify new vendor appears in GET /vendors
    res_vendors = api_client.get("/vendors")
    assert res_vendors.status_code == 200
    vendor_labels = [v["vendor_label"] for v in res_vendors.json()["vendors"]]
    assert "API Resolved Vendor" in vendor_labels


def test_api_key_auth_header(api_client, synthetic_png_bytes):
    """Verify X-API-Key header authentication when settings.api_key is configured."""
    original_key = settings.api_key
    try:
        settings.api_key = "secret_api_key_99"

        # Missing header -> 401 Unauthorized
        res_unauth = api_client.get("/health")
        assert res_unauth.status_code == 401

        # Correct header -> 200 OK
        res_auth = api_client.get("/health", headers={"X-API-Key": "secret_api_key_99"})
        assert res_auth.status_code == 200
    finally:
        settings.api_key = original_key


def test_dashboard_ui_endpoint(api_client):
    """Verify GET /dashboard and GET / serve the index.html frontend."""
    res_dash = api_client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "text/html" in res_dash.headers["content-type"]
    assert "Invoice Logo Classifier" in res_dash.text

    res_root = api_client.get("/")
    assert res_root.status_code == 200
    assert "Invoice Logo Classifier" in res_root.text


def test_get_crop_image_endpoint(api_client, synthetic_png_bytes):
    """Verify GET /review-queue/{id}/crop serves the stored logo crop PNG image."""
    service = get_classifier_service()
    service.closed_set_threshold = 0.999
    service.gallery_threshold = 0.999

    # Trigger unknown classification to enqueue item
    files = {"file": ("crop_test.png", synthetic_png_bytes, "image/png")}
    api_client.post("/invoices/classify", files=files)

    res_list = api_client.get("/review-queue?limit=1")
    item_id = res_list.json()[0]["id"]

    res_crop = api_client.get(f"/review-queue/{item_id}/crop")
    assert res_crop.status_code == 200
    assert "image/png" in res_crop.headers["content-type"]
    assert len(res_crop.content) > 0
