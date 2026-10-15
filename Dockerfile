# ==============================================================================
# Multi-stage Dockerfile for Invoice Logo Classifier Service
# Target: Python 3.11-slim with system OCR (Tesseract) & PDF/Doc rendering (LibreOffice + Poppler)
# ==============================================================================

# Stage 1: Build stage for Python virtual environment & dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Final minimal runtime image
FROM python:3.11-slim AS runtime

LABEL maintainer="invoice-logo-classifier"
LABEL description="Invoice logo classifier API & Ops Dashboard"

WORKDIR /app

# Install OS runtime dependencies: Tesseract OCR, LibreOffice headless, Poppler tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libreoffice \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Create directories for persistent runtime data
RUN mkdir -p /app/models /app/gallery_store /app/review_queue/crops /app/backups

# Copy application source code
COPY invoice_logo_classifier/ /app/invoice_logo_classifier/

# Architecture Decision: Model Weights & Metadata Deployment Strategy
# Trained model weights (`best_model.pth`) and metadata (`model_metadata.json`) are copied
# directly into `/app/models/` during the image build.
# WHY: Baking model weights into the container image enables instant, zero-config deployment
# on free-tier container platforms (e.g. Render, Koyeb, Fly.io, or self-hosted VMs) without
# requiring external S3/GCS buckets or network storage mounts.
# OVERRIDE: Operators can mount a persistent volume over `/app/models` to supply custom
# updated weights at runtime without rebuilding the container image.
COPY models/ /app/models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "invoice_logo_classifier.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
