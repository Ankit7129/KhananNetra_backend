# ============================================================================== 
# Production Dockerfile for KhananNetra Backend (Node.js + Python)
# Single stage image tuned for Cloud Run. Build for linux/amd64 using build args.
# ============================================================================== 

FROM node:22-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app

# Keep installs non-interactive and limit Python file noise
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# Install system dependencies for Python backend and geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates \
    wget \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    proj-data \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    dos2unix \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js production dependencies early for better caching
COPY package.json package-lock.json ./
ENV NODE_ENV=production
RUN npm ci --omit=dev --ignore-scripts && npm cache clean --force

# Prepare isolated Python environment before copying sources to maximize cache reuse
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    GDAL_DATA=/usr/share/gdal/ \
    PROJ_LIB=/usr/share/proj

# Install Python dependencies using requirements cache layer
COPY python-backend/requirements.txt python-backend/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir certifi==2023.11.17 && \
    pip install --no-cache-dir -r python-backend/requirements.txt

# Copy application source
COPY . .

# Normalize scripts and set permissions
RUN dos2unix /app/start-production.sh && \
    chmod +x /app/start-production.sh

# Prepare runtime directories and drop root privileges
RUN mkdir -p /tmp/kagglehub /app/logs && \
    chown -R node:node /app

USER node

# Environment variables (defaults - can be overridden in Cloud Run)
ENV PORT=8080 \
    PYTHON_BACKEND_PORT=9000 \
    PYTHON_BACKEND_URL=http://127.0.0.1:9000

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Start the application
CMD ["./start-production.sh"]