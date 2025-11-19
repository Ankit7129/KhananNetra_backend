"""
Real-time tile fetching service for streaming satellite imagery.
Fetches and processes tiles individually, sending updates to frontend.
"""

import ee
import numpy as np
import requests
import io
from PIL import Image
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64


class TileFetchingService:
    """Service for fetching satellite tiles in real-time."""
    
    def __init__(self, ee_service):
        """Initialize with Earth Engine service."""
        self.ee_service = ee_service
        self.tile_size = 512  # 512x512 pixels to stay under Earth Engine's 50MB limit
        self.max_concurrent_downloads = 5
        self.cached_6band_data = []  # Store 6-band data for ML inference
        self.resolution = 10  # 10m resolution for Sentinel-2
        # Note: 512×512×6 bands×4 bytes = ~6.3 MB (well under 50MB limit)
        # 1024×1024 was 62MB which exceeded Earth Engine's download limit
    
    def calculate_tile_grid(
        self, 
        geometry: Dict, 
        resolution: int = 10
    ) -> List[Dict]:
        """
        Calculate grid of 1024×1024 pixel square tiles covering the AOI.
        Tiles are complete squares even if parts extend outside AOI (majority inside).
        
        Returns:
            List of tile dictionaries with bounds and metadata
        """
        # Convert to EE geometry
        ee_geom = self.ee_service.geometry_to_ee_polygon(geometry)
        bounds = ee_geom.bounds().getInfo()['coordinates'][0]
        
        # Get bounding box
        lons = [coord[0] for coord in bounds]
        lats = [coord[1] for coord in bounds]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Calculate center of AOI
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        
        # Calculate tile size in degrees
        # 512 pixels * 10m resolution = 5,120m ≈ 5.12km per tile
        # At equator: 1 degree ≈ 111 km
        # Adjust for latitude (tiles get smaller as you move away from equator)
        import math
        tile_size_km = (self.tile_size * resolution) / 1000  # 5.12 km
        tile_deg_lat = tile_size_km / 111.0  # Latitude degrees
        tile_deg_lon = tile_size_km / (111.0 * math.cos(math.radians(center_lat)))  # Longitude degrees (adjust for latitude)
        
        # Generate square tiles covering the ENTIRE AOI with overlap buffer
        # CRITICAL FIX: Ensure full AOI coverage by extending grid beyond AOI bounds
        tiles = []
        tile_id = 0
        
        # Add buffer around AOI to ensure nothing is missed at edges
        buffer_degrees = tile_deg_lat * 0.1  # 10% buffer around AOI
        extended_min_lat = min_lat - buffer_degrees
        extended_max_lat = max_lat + buffer_degrees
        extended_min_lon = min_lon - (tile_deg_lon * 0.1)
        extended_max_lon = max_lon + (tile_deg_lon * 0.1)
        
        lat = extended_min_lat
        row = 0
        while lat < extended_max_lat:
            lon = extended_min_lon
            col = 0
            while lon < extended_max_lon:
                # Create square tile bounds (512×512 pixels at 10m resolution)
                tile_bounds = [
                    [lon, lat],
                    [lon + tile_deg_lon, lat],
                    [lon + tile_deg_lon, lat + tile_deg_lat],
                    [lon, lat + tile_deg_lat],
                    [lon, lat]
                ]
                
                # Calculate tile center
                center_tile_lon = lon + tile_deg_lon / 2
                center_tile_lat = lat + tile_deg_lat / 2
                
                # For now, include all tiles in the buffered grid
                # The intersection check can be expensive and may cause issues
                # Better to have a few extra tiles than miss AOI coverage
                tiles.append({
                    'id': tile_id,
                    'tile_id': f"tile_{tile_id + 1}",  # Consistent naming
                    'row': row,
                    'col': col,
                    'bounds': tile_bounds,
                    'center': [center_tile_lon, center_tile_lat],
                    'size_pixels': self.tile_size,  # 512×512 pixels
                    'resolution': resolution,  # 10m per pixel
                    'intersects_aoi': True
                })
                tile_id += 1
                
                col += 1
                lon += tile_deg_lon
            
            row += 1
            lat += tile_deg_lat
        
        print(f"📐 Generated {len(tiles)} square tiles ({self.tile_size}×{self.tile_size} pixels at {resolution}m resolution, {row} rows × {col} cols)")
        return tiles
    
    def fetch_tile_6band_with_rgb_preview(
        self,
        tile_info: Dict,
        collection: ee.ImageCollection
    ) -> Tuple[Dict, Optional[np.ndarray]]:
        """
        Fetch a single tile with 6-band data for ML model + RGB preview for display.
        
        Args:
            tile_info: Tile metadata dict
            collection: Earth Engine image collection
        
        Returns:
            Tuple of (tile_info_with_rgb_preview, 6band_array) or (tile_info, None) if failed
        """
        # 6 bands required by UNet model: B2, B3, B4, B8, B11, B12
        ml_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        # RGB bands for preview: Red (B4), Green (B3), Blue (B2)
        rgb_bands = ['B4', 'B3', 'B2']
        
        try:
            # Create tile geometry
            tile_geom = ee.Geometry.Polygon([tile_info['bounds']])
            
            # Get median composite for this tile (all 6 bands for ML)
            image_6band = collection.select(ml_bands).median().clip(tile_geom)
            
            # STEP 1: Fetch RGB preview for frontend display
            rgb_image = collection.select(rgb_bands).median().clip(tile_geom)
            
            # RGB visualization parameters
            rgb_vis_params = {
                'min': 0,
                'max': 3000,  # Sentinel-2 surface reflectance range
                'bands': rgb_bands,
                'dimensions': f'{self.tile_size}x{self.tile_size}',
                'region': tile_geom,
                'format': 'png'
            }
            
            # Get RGB thumbnail URL
            rgb_url = rgb_image.getThumbURL(rgb_vis_params)
            
            # Download RGB preview with increased timeout (90 seconds for large tiles)
            rgb_response = requests.get(rgb_url, timeout=90)
            rgb_response.raise_for_status()
            
            # Convert RGB to base64 for frontend
            rgb_img = Image.open(io.BytesIO(rgb_response.content))
            img_bytes = io.BytesIO()
            rgb_img.save(img_bytes, format='PNG')
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            
            # STEP 2: Fetch 6-band data for ML model
            # Get download URL for all 6 bands
            band_data = {}
            six_band_array = None
            
            try:
                # Download 6-band GeoTIFF at NATIVE 10m resolution (matches notebook)
                # CRITICAL: Ensure exact pixel alignment and full coverage
                # 512×512 at 10m = ~6.3MB per tile (6 bands × 4 bytes/pixel)
                # Note: Earth Engine has a 50MB download limit per request
                
                # Retry logic for Earth Engine timeout issues
                max_retries = 3
                retry_count = 0
                download_url = None
                
                while retry_count < max_retries and download_url is None:
                    try:
                        if retry_count > 0:
                            print(f"      🔄 Retry {retry_count}/{max_retries} for 6-band download URL...")
                        
                        # Try different parameter combinations based on retry count
                        if retry_count == 0:
                            # Primary: Use region + scale (most reliable)
                            download_params = {
                                'scale': 10,  # NATIVE 10m resolution
                                'region': tile_geom,
                                'format': 'GEO_TIFF',
                                'bands': ml_bands,
                                'crs': 'EPSG:4326'
                            }
                        elif retry_count == 1:
                            # Fallback 1: Use dimensions + region (let EE determine scale)
                            download_params = {
                                'dimensions': f'{self.tile_size}x{self.tile_size}',
                                'region': tile_geom,
                                'format': 'GEO_TIFF',
                                'bands': ml_bands,
                                'crs': 'EPSG:4326'
                            }
                        else:
                            # Fallback 2: Minimal parameters
                            download_params = {
                                'scale': 10,
                                'region': tile_geom,
                                'format': 'GEO_TIFF',
                                'bands': ml_bands
                            }
                        
                        download_url = image_6band.getDownloadURL(download_params)
                    except Exception as url_error:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise Exception(f"Failed to get download URL after {max_retries} retries: {url_error}")
                        import time
                        time.sleep(2 * retry_count)  # Exponential backoff
                
                # Download the 6-band data with increased timeout
                band_response = requests.get(download_url, timeout=180)  # Increased to 3 minutes
                band_response.raise_for_status()
                band_response.raise_for_status()
                
                # Read as rasterio dataset with FULL precision preservation
                import rasterio
                from rasterio.transform import Affine
                
                with rasterio.open(io.BytesIO(band_response.content)) as src:
                    # CRITICAL: Store exact transform for perfect pixel alignment
                    transform = src.transform
                    crs = src.crs
                    
                    # Read all 6 bands with full precision
                    six_band_array = np.zeros((src.height, src.width, 6), dtype=np.float32)
                    for i in range(6):
                        band_data = src.read(i + 1).astype(np.float32)
                        six_band_array[:, :, i] = band_data
                    
                    # Store georeferencing info for precise coordinate mapping
                    tile_info['transform'] = transform
                    tile_info['crs'] = str(crs) if crs else 'EPSG:4326'
                    tile_info['exact_shape'] = six_band_array.shape[:2]  # (H, W)
                
                print(f"      ✅ 6-band data fetched: {six_band_array.shape}, transform: {transform}")
                
            except Exception as band_error:
                print(f"      ⚠️  Could not fetch 6-band data: {band_error}")
                # Continue anyway with RGB preview
            
            # Add fields expected by frontend
            tile_info['tile_id'] = str(tile_info['id'])
            tile_info['image_base64'] = img_base64  # RGB preview
            tile_info['status'] = 'fetched'
            tile_info['bands'] = ml_bands
            tile_info['has_6band_data'] = six_band_array is not None
            
            return tile_info, six_band_array
            
        except Exception as e:
            print(f"   ⚠️  Failed to fetch tile {tile_info['id']}: {e}")
            tile_info['status'] = 'failed'
            tile_info['error'] = str(e)
            return tile_info, None
    
    def fetch_all_tiles_realtime(
        self,
        geometry: Dict,
        tiles: List[Dict],
        callback: Optional[callable] = None,
        start_date: str = "2024-01-01",
        end_date: str = "2025-12-31"
    ) -> List[Dict]:
        """
        Fetch all tiles from the latest available year with real-time progress updates.
        Fetches 1024×1024 pixel square tiles at 10m resolution.
        
        Args:
            geometry: AOI geometry
            tiles: Pre-calculated tile grid (1024×1024 pixels each)
            callback: Function to call when each tile is fetched
            start_date: Start date for imagery (default: 2024-01-01 for latest year)
            end_date: End date for imagery (default: 2025-12-31)
        
        Returns:
            List of fetched tiles with RGB preview data and 6-band arrays cached
        """
        print(f"🛰️  Fetching {len(tiles)} tiles in real-time...")
        
        # Get Sentinel-2 collection
        ee_geom = self.ee_service.geometry_to_ee_polygon(geometry)
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start_date, end_date)
            .filterBounds(ee_geom)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        )
        
        print(f"   � Using Sentinel-2 collection with {collection.size().getInfo()} images")
        
        fetched_tiles = []
        tile_6band_data = []  # Store 6-band arrays for ML
        
        # Fetch tiles concurrently with thread pool
        with ThreadPoolExecutor(max_workers=self.max_concurrent_downloads) as executor:
            # Submit all tile fetch jobs (6-band + RGB preview)
            future_to_tile = {
                executor.submit(self.fetch_tile_6band_with_rgb_preview, tile, collection): tile
                for tile in tiles
            }
            
            # Process completed tiles as they finish
            for idx, future in enumerate(as_completed(future_to_tile), 1):
                tile_info, six_band_array = future.result()
                if tile_info:
                    fetched_tiles.append(tile_info)
                    
                    # Store 6-band data if available
                    if six_band_array is not None:
                        tile_6band_data.append({
                            'tile_id': str(tile_info['id']),
                            'data': six_band_array,
                            'bounds': tile_info['bounds'],
                            'row': tile_info.get('row'),
                            'col': tile_info.get('col'),
                            'transform': tile_info.get('transform'),
                            'crs': tile_info.get('crs', 'EPSG:4326'),
                            'exact_shape': tile_info.get('exact_shape')
                        })
                
                    progress = (idx / len(tiles)) * 100
                    print(f"   📥 Tile {tile_info['id']} ({idx}/{len(tiles)}) - {progress:.1f}% [6-band: {'✅' if six_band_array is not None else '❌'}]")
                    
                    # Call callback with tile data for real-time updates
                    if callback:
                        callback({
                            'type': 'tile_fetched',
                            'tile': tile_info,  # Has RGB preview in image_base64
                            'progress': progress,
                            'total': len(tiles),
                            'current': idx,
                            'has_ml_data': six_band_array is not None
                        })
        
        print(f"   ✅ All {len(fetched_tiles)} tiles fetched!")
        print(f"   📊 6-band data available for {len(tile_6band_data)}/{len(tiles)} tiles")
        
        # Store 6-band data for later ML inference
        self.cached_6band_data = tile_6band_data
        
        return fetched_tiles
    
    def get_6band_tile_data(self) -> List[Dict]:
        """
        Get cached 6-band tile data for ML inference.
        
        Returns:
            List of dicts with tile_id, data (numpy array), and bounds
        """
        return self.cached_6band_data
    
    def clear_cache(self):
        """Clear cached 6-band data to free memory."""
        self.cached_6band_data = []
        print(f"   🧹 Cleared 6-band tile cache")
