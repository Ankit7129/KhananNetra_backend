"""
AOI (Area of Interest) router for handling geographic area operations.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, List

from ..models.schemas import (
    AOIRequest, AOIResponse, BoundingBox, SearchLocation, ErrorResponse
)
from ..services.geospatial_service import geospatial_service

router = APIRouter()

@router.post("/create", response_model=AOIResponse)
async def create_aoi(aoi_request: AOIRequest):
    """Create a new Area of Interest from geometry."""
    try:
        aoi_id, aoi_feature = geospatial_service.create_aoi_from_geometry(
            aoi_request.geometry, 
            aoi_request.properties
        )
        
        bounding_box = geospatial_service.get_bounding_box(aoi_request.geometry)
        
        return AOIResponse(
            id=aoi_id,
            feature=aoi_feature,
            bounding_box=bounding_box,
            status="created",
            message="AOI created successfully"
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/upload", response_model=AOIResponse)
async def upload_aoi_file(file: UploadFile = File(...)):
    """Upload and process a geospatial file (KML, GeoJSON, or Shapefile)."""
    
    # Validate file type
    allowed_extensions = ['.kml', '.geojson', '.json', '.zip']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (10MB limit)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    content = await file.read()
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10MB limit"
        )
    
    try:
        aoi_id, aoi_feature = geospatial_service.process_uploaded_file(
            content, file.filename
        )
        
        bounding_box = geospatial_service.get_bounding_box(aoi_feature.geometry)
        
        return AOIResponse(
            id=aoi_id,
            feature=aoi_feature,
            bounding_box=bounding_box,
            status="created",
            message=f"AOI created from {file.filename}"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )

@router.get("/{aoi_id}", response_model=AOIResponse)
async def get_aoi(aoi_id: str):
    """Retrieve an AOI by its ID."""
    aoi_feature = geospatial_service.get_aoi(aoi_id)
    
    if not aoi_feature:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AOI with ID {aoi_id} not found"
        )
    
    bounding_box = geospatial_service.get_bounding_box(aoi_feature.geometry)
    
    return AOIResponse(
        id=aoi_id,
        feature=aoi_feature,
        bounding_box=bounding_box,
        status="retrieved",
        message="AOI retrieved successfully"
    )

@router.get("/", response_model=Dict[str, AOIResponse])
async def list_aois():
    """List all stored AOIs."""
    aois = geospatial_service.list_aois()
    
    result = {}
    for aoi_id, aoi_feature in aois.items():
        bounding_box = geospatial_service.get_bounding_box(aoi_feature.geometry)
        result[aoi_id] = AOIResponse(
            id=aoi_id,
            feature=aoi_feature,
            bounding_box=bounding_box,
            status="stored"
        )
    
    return result

@router.delete("/{aoi_id}")
async def delete_aoi(aoi_id: str):
    """Delete an AOI by its ID."""
    success = geospatial_service.delete_aoi(aoi_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AOI with ID {aoi_id} not found"
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": f"AOI {aoi_id} deleted successfully"}
    )

@router.get("/search/location")
async def search_location(query: str) -> List[SearchLocation]:
    """Search for geographic locations by name."""
    
    # This is a placeholder implementation
    # In production, you would integrate with a geocoding service like:
    # - Nominatim (OpenStreetMap)
    # - Google Geocoding API
    # - Mapbox Geocoding API
    
    # Mock response for development
    mock_results = [
        SearchLocation(
            name=f"Location: {query}",
            coordinates={"latitude": 40.7128, "longitude": -74.0060},
            country="Mock Country",
            admin_area="Mock State"
        )
    ]
    
    return mock_results