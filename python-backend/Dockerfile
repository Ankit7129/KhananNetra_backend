FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including geospatial libraries
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY python-backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (excluding .h5 model files - they're on Kaggle)
COPY python-backend/app ./app
COPY python-backend/main.py .

# Create models and kagglehub cache directories
RUN mkdir -p /app/app/models /tmp/kagglehub

# Configure model loading from Kaggle
ENV DOWNLOAD_MODELS_ON_STARTUP=true
ENV MODEL_CACHE_DIR=/tmp/kagglehub
# Set these via GitHub Actions secrets or GCP Cloud Run env vars:
# KAGGLE_USERNAME=your-username
# KAGGLE_KEY=your-api-key
# KAGGLE_MODEL_PATH=soumyadiptadey/khanannetra-production/tensorFlow2/version1

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]
