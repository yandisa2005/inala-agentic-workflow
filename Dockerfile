FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source tree
COPY src/ ./src/

# Ensure /app is in PYTHONPATH so python can resolve modules cleanly
ENV PYTHONPATH=/app

EXPOSE 8080

# Shell form lets $PORT interpolate dynamically from Cloud Run
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}