"""
Google Earth Engine Service for fetching real Sentinel-2 satellite imagery.
Based on the Data_acquisition.ipynb approach from the notebooks.
"""

import ee
import os
import requests
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import tempfile
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import json


class EarthEngineService:
    """Service for fetching Sentinel-2 imagery from Google Earth Engine."""
    
    def __init__(self, project='mining-detection'):
        """Initialize Earth Engine with authentication and project."""
        try:
            # Check if EE is already initialized (by startup event)
            ee.List([1, 2, 3]).getInfo()  # Quick test to see if EE is ready
            print(f"✅ Earth Engine already initialized and ready")
        except Exception as init_error:
            # EE not initialized yet, try to initialize it
            try:
                print(f"🔓 Initializing Earth Engine...")
                # Check if service account credentials are available
                gee_credentials = os.environ.get('GEE_SERVICE_ACCOUNT_CREDENTIALS')
                
                if gee_credentials:
                    # Production: Use service account authentication
                    print("🔐 Using service account credentials...")
                    credentials_dict = json.loads(gee_credentials)
                    credentials = ee.ServiceAccountCredentials(
                        email=credentials_dict['client_email'],
                        key_data=gee_credentials
                    )
                    ee.Initialize(credentials=credentials, project=project)
                    print(f"✅ Earth Engine initialized with service account")
                else:
                    # Development: Use default authentication (requires earthengine authenticate)
                    print("🔓 Using default authentication...")
                    ee.Initialize(project=project)
                    print(f"✅ Earth Engine initialized")
            except Exception as e:
                print(f"⚠️ Earth Engine initialization issue: {e}")
                # Don't raise - EE may still be usable if initialized elsewhere
    
    def clean_coords(self, coords):
        """Remove Z values (altitude) from coordinate tuples."""
        return [(x, y) for x, y, *_ in coords]
    
    def geometry_to_ee_polygon(self, geometry: Dict) -> ee.Geometry:
        """
        Convert GeoJSON geometry to Earth Engine Polygon.
        Handles both Polygon and MultiPolygon types.
        """
        if geometry['type'] == 'Polygon':
            ee_coords = [self.clean_coords(ring) for ring in geometry['coordinates']]
            return ee.Geometry.Polygon(ee_coords)
        
        elif geometry['type'] == 'MultiPolygon':
            # Flatten MultiPolygon to single Polygon
            ee_coords = []
            for poly in geometry['coordinates']:
                for ring in poly:
                    ee_coords.append(self.clean_coords(ring))
            return ee.Geometry.Polygon(ee_coords)
        
        else:
            raise ValueError(f"Unsupported geometry type: {geometry['type']}")
    
    def fetch_sentinel2_imagery(
        self,
        aoi_geometry: Dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_cloud_cover: float = 20.0,
        bands: List[str] = None
    ) -> Tuple[str, Dict]:
        """
        Fetch Sentinel-2 imagery for the given AOI.
        
        Args:
            aoi_geometry: GeoJSON geometry dict
            start_date: Start date (YYYY-MM-DD), defaults to last 12 months
            end_date: End date (YYYY-MM-DD), defaults to today
            max_cloud_cover: Maximum cloud coverage percentage
            bands: List of bands to fetch, defaults to ['B2','B3','B4','B8','B11','B12']
        
        Returns:
            Tuple of (download_url, metadata_dict)
        """
        # Default date range: last 12 months
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Default bands: 6-band configuration for mining detection
        if bands is None:
            bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        
        print(f"🛰️  Fetching Sentinel-2 imagery...")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Max cloud cover: {max_cloud_cover}%")
        print(f"   Bands: {', '.join(bands)}")
        
        # Convert geometry to EE Polygon
        ee_polygon = self.geometry_to_ee_polygon(aoi_geometry)
        
        # Create Sentinel-2 collection filter
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(ee_polygon)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
        )
        
        # Check if any images are available
        try:
            count = collection.size().getInfo()
            print(f"   Found {count} images matching criteria")
            
            if count == 0:
                raise ValueError(f"No Sentinel-2 images found for the given AOI and date range")
        
        except Exception as e:
            print(f"   ⚠️  Error querying collection: {e}")
            raise
        
        # Get cloud-free median composite
        image = collection.select(bands).median().clip(ee_polygon)
        
        # Get metadata from the collection
        first_image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        try:
            metadata = {
                'date_acquired': first_image.date().format('YYYY-MM-dd').getInfo(),
                'cloud_coverage': first_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo(),
                'satellite': 'Sentinel-2',
                'bands': bands,
                'resolution': '10-20m',
                'processing_level': 'L2A'
            }
        except:
            metadata = {
                'satellite': 'Sentinel-2',
                'bands': bands,
                'resolution': '10-20m'
            }
        
        # Get download URL with intelligent scale handling
        try:
            # Get bounding box for export
            bounds = ee_polygon.bounds().getInfo()['coordinates']
            area = ee_polygon.area().getInfo()  # Area in square meters
            area_km2 = area / 1_000_000
            
            print(f"   AOI area: {area_km2:.2f} km²")
            
            # Determine appropriate scale based on area to stay under 50MB limit
            # Earth Engine has a 50MB limit for getDownloadURL
            # Estimate: 6 bands * 8 bytes * pixels = data size
            # For 10m resolution: ~100 pixels/km² → 4800 bytes/km²
            
            if area_km2 > 100:
                # Very large area - use 60m resolution
                scale = 60
                print(f"   ⚠️  Large AOI detected! Using 60m resolution to stay under download limit")
            elif area_km2 > 50:
                # Large area - use 30m resolution  
                scale = 30
                print(f"   ⚠️  Medium-large AOI detected! Using 30m resolution")
            elif area_km2 > 20:
                # Medium area - use 20m resolution
                scale = 20
                print(f"   ℹ️  Using 20m resolution for optimal balance")
            else:
                # Small area - use 10m resolution (full detail)
                scale = 10
                print(f"   ✅ Small AOI - using full 10m resolution")
            
            # Try to create download URL with retry logic for large areas
            max_retries = 3
            scales_to_try = [scale, scale * 2, scale * 4]  # Progressive downsampling
            
            url = None
            final_scale = scale
            
            for retry, try_scale in enumerate(scales_to_try):
                try:
                    if retry > 0:
                        print(f"   🔄 Retry {retry}: Attempting with {try_scale}m resolution...")
                    
                    url = image.getDownloadURL({
                        'scale': try_scale,
                        'region': bounds,
                        'format': 'GEO_TIFF',
                        'crs': 'EPSG:4326'
                    })
                    
                    final_scale = try_scale
                    break  # Success!
                    
                except Exception as retry_error:
                    error_msg = str(retry_error)
                    
                    if 'Total request size' in error_msg and retry < len(scales_to_try) - 1:
                        # Size limit exceeded, try coarser resolution
                        print(f"   ⚠️  Download size too large at {try_scale}m resolution")
                        continue
                    elif retry == len(scales_to_try) - 1:
                        # Final retry failed
                        print(f"   ❌ Failed even at {try_scale}m resolution")
                        raise ValueError(
                            f"AOI too large ({area_km2:.2f} km²) even at {try_scale}m resolution. "
                            f"Please select a smaller area (< {area_km2/4:.0f} km²) or contact support."
                        )
                    else:
                        raise
            
            if url is None:
                raise ValueError("Failed to generate download URL after all retries")
            
            metadata['scale_meters'] = final_scale
            metadata['area_km2'] = round(area_km2, 2)
            
            if final_scale > scale:
                print(f"   ⚠️  Used {final_scale}m resolution due to size constraints")
            
            print(f"   ✅ Image prepared for download")
            print(f"   Final resolution: {final_scale}m per pixel")
            print(f"   Metadata: {metadata}")
            
            return url, metadata
        
        except Exception as e:
            error_msg = str(e)
            print(f"   ⚠️  Error generating download URL: {error_msg}")
            
            # Provide helpful error message for common issues
            if 'Total request size' in error_msg:
                raise ValueError(
                    f"Area of Interest is too large for processing. "
                    f"Please select a smaller area (recommended: < 20 km²) and try again."
                )
            else:
                raise
    
    def download_imagery(self, url: str, output_path: str) -> str:
        """
        Download the imagery from the GEE URL.
        
        Args:
            url: Download URL from GEE
            output_path: Local path to save the imagery
        
        Returns:
            Path to downloaded file
        """
        print(f"📥 Downloading imagery to {output_path}...")
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"   ✅ Download complete: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            return output_path
        
        except Exception as e:
            print(f"   ⚠️  Download failed: {e}")
            raise
    
    def get_imagery_info(self, image_path: str) -> Dict:
        """
        Get information about the downloaded imagery.
        
        Args:
            image_path: Path to the GeoTIFF file
        
        Returns:
            Dictionary with image information
        """
        with rasterio.open(image_path) as src:
            return {
                'width': src.width,
                'height': src.height,
                'bands': src.count,
                'crs': src.crs.to_string(),
                'bounds': src.bounds,
                'transform': src.transform,
                'dtype': src.dtypes[0]
            }


# Singleton instance
_earth_engine_service = None
_init_lock = None  # Will be initialized in get_earth_engine_service
_is_initializing = False

def get_earth_engine_service() -> EarthEngineService:
    """Get or create the Earth Engine service singleton."""
    global _earth_engine_service, _init_lock, _is_initializing
    
    # Import asyncio here to avoid issues with event loop
    import asyncio
    
    # Initialize lock if needed
    if _init_lock is None:
        try:
            _init_lock = asyncio.Lock()
        except RuntimeError:
            # No event loop, just use synchronous initialization
            if _earth_engine_service is None:
                _earth_engine_service = EarthEngineService()
            return _earth_engine_service
    
    # If already initialized, return immediately
    if _earth_engine_service is not None:
        return _earth_engine_service
    
    # If currently initializing by another coroutine, wait briefly then return None
    # The caller should handle None gracefully or retry
    if _is_initializing:
        print("⏳ Earth Engine service is currently initializing, please wait...")
        import time
        time.sleep(0.1)  # Brief wait
        return _earth_engine_service  # May still be None
    
    # Initialize the service
    _is_initializing = True
    try:
        if _earth_engine_service is None:
            _earth_engine_service = EarthEngineService()
    finally:
        _is_initializing = False
    
    return _earth_engine_service
