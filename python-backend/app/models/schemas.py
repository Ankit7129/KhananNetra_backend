"""
Pydantic models and schemas for the KhananNetra API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class FileType(str, Enum):
    KML = "kml"
    GEOJSON = "geojson"
    SHAPEFILE = "shp"

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    CONNECTING = "connecting"
    REQUESTING = "requesting"
    PREPROCESSING = "preprocessing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Coordinates(BaseModel):
    """Geographic coordinates model."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")

class BoundingBox(BaseModel):
    """Bounding box for geographic areas."""
    north: float = Field(..., ge=-90, le=90)
    south: float = Field(..., ge=-90, le=90)
    east: float = Field(..., ge=-180, le=180)
    west: float = Field(..., ge=-180, le=180)

class AOIGeometry(BaseModel):
    """Area of Interest geometry model."""
    type: str = Field(..., description="Geometry type (Polygon, MultiPolygon)")
    coordinates: List[Any] = Field(..., description="Coordinate array")

class AOIProperties(BaseModel):
    """Properties associated with an AOI."""
    name: Optional[str] = Field(None, description="AOI name")
    description: Optional[str] = Field(None, description="AOI description")
    created_at: datetime = Field(default_factory=datetime.now)
    area_km2: Optional[float] = Field(None, description="Area in square kilometers")

class AOIFeature(BaseModel):
    """GeoJSON-like feature for AOI."""
    type: str = Field(default="Feature")
    geometry: AOIGeometry
    properties: AOIProperties

class AOIRequest(BaseModel):
    """Request model for creating an AOI."""
    geometry: AOIGeometry
    properties: Optional[AOIProperties] = Field(default_factory=AOIProperties)

class AOIResponse(BaseModel):
    """Response model for AOI operations."""
    id: str = Field(..., description="Unique AOI identifier")
    feature: AOIFeature
    bounding_box: BoundingBox
    status: str = Field(default="created")
    message: str = Field(default="AOI created successfully")

class SearchLocation(BaseModel):
    """Geographic location search result."""
    name: str
    coordinates: Coordinates
    bounding_box: Optional[BoundingBox] = None
    country: Optional[str] = None
    admin_area: Optional[str] = None

class AnalysisRequest(BaseModel):
    """Request model for starting analysis."""
    aoi_id: str = Field(..., description="AOI identifier")
    analysis_type: str = Field(default="mining_detection", description="Type of analysis")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)

class AnalysisResponse(BaseModel):
    """Response model for analysis operations."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    aoi_id: str = Field(..., description="Associated AOI identifier")
    status: AnalysisStatus
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage")
    current_step: Optional[str] = Field(None, description="Current processing step")
    estimated_completion: Optional[datetime] = Field(None)
    results_url: Optional[str] = Field(None, description="URL to results when completed")
    tiles_fetched: Optional[int] = Field(None, description="Number of tiles fetched from satellite imagery")
    imagery_data: Optional[Dict[str, Any]] = Field(None, description="Detailed imagery and tile information")

class ImageryMetadata(BaseModel):
    """Satellite imagery metadata."""
    satellite: str = Field(..., description="Satellite name (e.g., Sentinel-2)")
    acquisition_date: datetime
    cloud_coverage: float = Field(..., ge=0.0, le=100.0)
    resolution: str = Field(..., description="Spatial resolution")
    bands: List[str] = Field(..., description="Available spectral bands")

class ImageryResponse(BaseModel):
    """Response model for imagery requests."""
    imagery_id: str = Field(..., description="Unique imagery identifier")
    aoi_id: str = Field(..., description="Associated AOI identifier")
    metadata: ImageryMetadata
    download_url: str = Field(..., description="URL to download imagery")
    thumbnail_url: Optional[str] = Field(None, description="URL to preview thumbnail")

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)