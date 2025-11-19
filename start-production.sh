#!/bin/bash

# ==============================================================================
# Production Startup Script for KhananNetra Backend (Cloud Run Compatible)
# Runs Python backend on internal port 9000 and Node.js on public port 8080
# ==============================================================================

set -e

echo "🚀 Starting KhananNetra Backend Services (Cloud Run Mode)..."

# Cloud Run automatically sets PORT env var (default 8080)
NODE_PORT=${PORT:-8080}
PYTHON_PORT=9000    # internal-only, not exposed

echo "📍 Node.js will run on public PORT: $NODE_PORT"
echo "📍 Python (FastAPI) will run internally on PORT: $PYTHON_PORT"

# Export correct URLs so Node.js can proxy to FastAPI
export PORT=$NODE_PORT
export PYTHON_BACKEND_PORT=$PYTHON_PORT
export PYTHON_BACKEND_URL="http://127.0.0.1:$PYTHON_PORT"

# ------------------------------------------------------------------------------
# Start Python backend (FastAPI)
# ------------------------------------------------------------------------------
echo "🐍 Starting Python backend (FastAPI) on port $PYTHON_PORT ..."
cd /app/python-backend

uvicorn main:app \
  --host 0.0.0.0 \
  --port $PYTHON_PORT \
  --workers 1 \
  --log-level info &

PYTHON_PID=$!
echo "✅ Python backend started with PID: $PYTHON_PID"

# ------------------------------------------------------------------------------
# Wait for FastAPI to be available
# ------------------------------------------------------------------------------
echo "⏳ Waiting for Python backend to become ready..."

MAX_RETRIES=40
RETRY_COUNT=0

until curl -f http://127.0.0.1:$PYTHON_PORT/health > /dev/null 2>&1; do
  RETRY_COUNT=$((RETRY_COUNT+1))
  if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "❌ Python backend failed to start in time."
    kill $PYTHON_PID 2>/dev/null || true
    exit 1
  fi
  echo "⏳ [$RETRY_COUNT/$MAX_RETRIES] Python backend not ready yet..."
  sleep 1
done

echo "✅ Python backend is healthy!"

# ------------------------------------------------------------------------------
# Start Node.js backend in foreground (keeps Cloud Run container alive)
# ------------------------------------------------------------------------------
echo "🟢 Starting Node.js backend on port $NODE_PORT ..."
cd /app/nodejs-backend

exec node server.js

# ------------------------------------------------------------------------------
# Cleanup if Node.js stops
# ------------------------------------------------------------------------------
trap "kill $PYTHON_PID 2>/dev/null || true" EXIT
