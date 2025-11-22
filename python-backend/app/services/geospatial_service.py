"""
Geospatial processing service for KhananNetra.
Handles AOI validation, coordinate transformations, and file processing.
"""

import json
import uuid
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import tempfile
import zipfile
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, mapping, shape
from shapely.validation import make_valid
from shapely.ops import transform
from pyproj import Transformer
import numpy as np

from ..models.schemas import (
    AOIGeometry, AOIFeature, AOIProperties, BoundingBox, 
    Coordinates, FileType
)

class GeospatialService:
    """Service for geospatial operations and AOI management."""
    
    def __init__(self):
        self.aois: Dict[str, AOIFeature] = {}
        
    def validate_coordinates(self, coordinates: List[Any]) -> bool:
        """Validate coordinate array structure."""
        try:
            # Check if coordinates form a valid polygon
            if not isinstance(coordinates, list) or len(coordinates) == 0:
                return False
                
            # For polygon, should be list of linear rings
            for ring in coordinates:
                if not isinstance(ring, list) or len(ring) < 4:
                    return False
                    
                # Check if ring is closed (first == last coordinate)
                if ring[0] != ring[-1]:
                    return False
                    
                # Validate coordinate pairs
                for coord in ring:
                    if (not isinstance(coord, list) or 
                        len(coord) != 2 or 
                        not all(isinstance(x, (int, float)) for x in coord)):
                        return False
                        
                    lon, lat = coord
                    if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                        return False
                        
            return True
        except Exception:
            return False
    
    def create_aoi_from_geometry(self, geometry: AOIGeometry, 
                               properties: Optional[AOIProperties] = None,
                               aoi_id: Optional[str] = None) -> Tuple[str, AOIFeature]:
        """Create an AOI from geometry data."""
        
        # Validate coordinates
        if not self.validate_coordinates(geometry.coordinates):
            raise ValueError("Invalid coordinate structure")
            
        # Generate unique ID
        target_aoi_id = aoi_id or str(uuid.uuid4())
        
        # Create properties if not provided
        if properties is None:
            properties = AOIProperties()
            
        # Calculate area
        try:
            shapely_geom = shape(geometry.dict())
            # Transform to equal area projection for accurate area calculation
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            projected_geom = transform(transformer.transform, shapely_geom)
            area_m2 = projected_geom.area
            properties.area_km2 = area_m2 / 1_000_000  # Convert to km²
        except Exception as e:
            print(f"Warning: Could not calculate area: {e}")
            properties.area_km2 = None
        
        # Create AOI feature
        aoi_feature = AOIFeature(
            geometry=geometry,
            properties=properties
        )
        
        # Store AOI
        self.aois[target_aoi_id] = aoi_feature
        
        return target_aoi_id, aoi_feature
    
    def get_bounding_box(self, geometry: AOIGeometry) -> BoundingBox:
        """Calculate bounding box for geometry."""
        try:
            shapely_geom = shape(geometry.dict())
            bounds = shapely_geom.bounds  # (minx, miny, maxx, maxy)
            
            return BoundingBox(
                west=bounds[0],
                south=bounds[1],
                east=bounds[2],
                north=bounds[3]
            )
        except Exception as e:
            raise ValueError(f"Could not calculate bounding box: {e}")
    
    def process_uploaded_file(self, file_content: bytes, 
                            filename: str) -> Tuple[str, AOIFeature]:
        """Process uploaded geospatial file and create AOI."""
        
        # Determine file type
        file_type = self._get_file_type(filename)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            if file_type == FileType.SHAPEFILE:
                return self._process_shapefile(file_content, temp_path)
            elif file_type == FileType.GEOJSON:
                return self._process_geojson(file_content)
            elif file_type == FileType.KML:
                return self._process_kml(file_content, temp_path)
            else:
                raise ValueError(f"Unsupported file type: {filename}")
    
    def _get_file_type(self, filename: str) -> FileType:
        """Determine file type from filename."""
        extension = Path(filename).suffix.lower()
        
        if extension == '.zip':
            return FileType.SHAPEFILE
        elif extension == '.geojson' or extension == '.json':
            return FileType.GEOJSON
        elif extension == '.kml':
            return FileType.KML
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
    
    def _process_shapefile(self, zip_content: bytes, temp_path: Path) -> Tuple[str, AOIFeature]:
        """Process zipped shapefile."""
        zip_file = temp_path / "shapefile.zip"
        with open(zip_file, 'wb') as f:
            f.write(zip_content)
        
        # Extract zip file
        extract_dir = temp_path / "extracted"
        extract_dir.mkdir()
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find .shp file
        shp_files = list(extract_dir.glob("*.shp"))
        if not shp_files:
            raise ValueError("No .shp file found in the zip archive")
        
        shp_file = shp_files[0]
        
        # Read with geopandas
        gdf = gpd.read_file(shp_file)
        
        # Convert to WGS84 if needed
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        
        # Get first geometry (union if multiple)
        if len(gdf) > 1:
            geometry = gdf.geometry.unary_union
        else:
            geometry = gdf.geometry.iloc[0]
        
        # Ensure valid geometry
        if not geometry.is_valid:
            geometry = make_valid(geometry)
        
        # Convert to AOI format
        return self._geometry_to_aoi(geometry, {"source": "shapefile"})
    
    def _process_geojson(self, content: bytes) -> Tuple[str, AOIFeature]:
        """Process GeoJSON file."""
        try:
            geojson_data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        # Handle different GeoJSON structures
        if geojson_data.get('type') == 'Feature':
            geometry = geojson_data['geometry']
        elif geojson_data.get('type') == 'FeatureCollection':
            if not geojson_data.get('features'):
                raise ValueError("Empty FeatureCollection")
            geometry = geojson_data['features'][0]['geometry']
        elif geojson_data.get('type') in ['Polygon', 'MultiPolygon']:
            geometry = geojson_data
        else:
            raise ValueError("Invalid GeoJSON structure")
        
        shapely_geom = shape(geometry)
        if not shapely_geom.is_valid:
            shapely_geom = make_valid(shapely_geom)
        
        return self._geometry_to_aoi(shapely_geom, {"source": "geojson"})
    
    def _process_kml(self, content: bytes, temp_path: Path) -> Tuple[str, AOIFeature]:
        """Process KML file."""
        kml_file = temp_path / "file.kml"
        with open(kml_file, 'wb') as f:
            f.write(content)
        
        # Use geopandas to read KML (requires fiona[KML])
        try:
            gdf = gpd.read_file(kml_file, driver='KML')
        except Exception as e:
            raise ValueError(f"Could not read KML file: {e}")
        
        # Convert to WGS84 if needed
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        
        # Get first geometry (union if multiple)
        if len(gdf) > 1:
            geometry = gdf.geometry.unary_union
        else:
            geometry = gdf.geometry.iloc[0]
        
        # Ensure valid geometry
        if not geometry.is_valid:
            geometry = make_valid(geometry)
        
        return self._geometry_to_aoi(geometry, {"source": "kml"})
    
    def _geometry_to_aoi(self, shapely_geom, metadata: Dict[str, Any]) -> Tuple[str, AOIFeature]:
        """Convert Shapely geometry to AOI format."""
        
        # Convert to GeoJSON-like geometry
        geom_dict = mapping(shapely_geom)
        
        geometry = AOIGeometry(
            type=geom_dict['type'],
            coordinates=geom_dict['coordinates']
        )
        
        properties = AOIProperties(**metadata)
        
        return self.create_aoi_from_geometry(geometry, properties)
    
    def get_aoi(self, aoi_id: str) -> Optional[AOIFeature]:
        """Retrieve AOI by ID."""
        return self.aois.get(aoi_id)
    
    def list_aois(self) -> Dict[str, AOIFeature]:
        """List all stored AOIs."""
        return self.aois.copy()
    
    def delete_aoi(self, aoi_id: str) -> bool:
        """Delete AOI by ID."""
        if aoi_id in self.aois:
            del self.aois[aoi_id]
            return True
        return False

# Global service instance
geospatial_service = GeospatialService()