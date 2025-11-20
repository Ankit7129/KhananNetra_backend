"""
KhananNetra Python FastAPI Backend
Integrated with MERN stack for geospatial analysis
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import logging
import gc
import signal
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from app.utils.console import patch_console_outputs

# Load environment variables from .env file
load_dotenv()

# Ensure console output stays ASCII-only (prevents Windows code-page errors)
patch_console_outputs()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import routers
from app.routers import aoi, imagery
from app.routers import analysis_realtime as analysis
from app.routers import quantitative_analysis

# Import model loader
from app.utils.model_loader import get_model_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI startup and shutdown events.
    Handles model loading on startup and resource cleanup on shutdown.
    """
    # Startup: Load model
    logger.info("🚀 Starting KhananNetra Python Backend...")
    
    # Enable aggressive garbage collection during startup
    gc.set_debug(0)
    gc.collect()
    
    try:
        # Download and cache model on startup
        model_path = get_model_path()
        logger.info(f"✅ Model ready at: {model_path}")
        
        # Store model path in app state for use by services
        app.state.model_path = model_path
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.warning("⚠️ Backend will continue but inference may fail")
    
    logger.info("✅ Backend startup complete")
    
    yield
    
    # Shutdown: Cleanup resources
    logger.info("👋 Shutting down KhananNetra Python Backend...")
    
    # Import here to avoid issues if not yet initialized
    try:
        from app.routers.analysis_realtime import cleanup_analysis_results
        cleanup_analysis_results()
        logger.info("✅ Analysis results cleaned up")
    except Exception as e:
        logger.warning(f"⚠️ Error cleaning up analysis results: {e}")
    
    # Force garbage collection
    gc.collect()
    logger.info("✅ Garbage collection completed")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="KhananNetra Python API",
    description="Geospatial analysis backend for mining detection using satellite imagery and ML",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js frontend
        "http://localhost:5000",  # Node.js backend
        "http://localhost:3001",  # Alternative frontend port
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers from old backend
app.include_router(aoi.router, prefix="/api/v1/aoi", tags=["AOI"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(imagery.router, prefix="/api/v1/imagery", tags=["Imagery"])
app.include_router(quantitative_analysis.router, prefix="/api/v1/analysis", tags=["Analysis Quantitative"])


@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "KhananNetra Python API - Integrated with MERN Stack",
        "version": "2.0.0",
        "status": "running",
        "docs": "/api/docs",
        "endpoints": {
            "aoi": "/api/v1/aoi",
            "analysis": "/api/v1/analysis",
            "imagery": "/api/v1/imagery"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "python-backend",
        "version": "2.0.0"
    }


@app.get("/ping")
async def ping():
    """Ping endpoint for connectivity check."""
    return {
        "status": "pong",
        "service": "python-backend",
        "timestamp": str(__import__('datetime').datetime.now())
    }


if __name__ == "__main__":
    # Run on port 8000 to avoid conflict with Node.js backend (port 5000)
    # Configuration optimized for stability during long-running analysis
    import multiprocessing
    
    # Set multiprocessing start method to 'spawn' for better stability
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production to avoid resource leaks
        log_level="info",
        workers=1,  # Single worker to avoid multi-process issues with Earth Engine
        limit_max_requests=100,  # Restart worker more frequently to prevent memory buildup
        limit_concurrency=5,  # Reduce concurrent connections to prevent overload
        timeout_keep_alive=75,  # Increase keep-alive for long-running analysis
        timeout_graceful_shutdown=30,  # Give time for cleanup
    )
