# Multi-stage build for Colombia RAG Chatbot
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/ requirements/
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/base.txt

# Download and cache sentence-transformers model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser appuser

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy model cache
COPY --from=builder /root/.cache /home/appuser/.cache
RUN chown -R appuser:appuser /home/appuser/.cache

# Copy application code
COPY app/ app/
COPY scripts/ scripts/
COPY data/ data/

# Set up data persistence
RUN mkdir -p data/vectorstore data/cache && \
    chown -R appuser:appuser data/

# Environment variables
ENV PYTHONPATH=/app
ENV APP_ENV=production
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/home/appuser/.cache

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]