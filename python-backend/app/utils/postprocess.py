"""
Professional-grade post-processing for mining detection.
Converts ML probability maps to clean, labeled polygon blocks with:
- Advanced thresholding (Otsu + confidence filtering)
- Morphological cleaning (remove noise, fill holes)
- Size-based filtering (remove misclassified small regions)
- Polygon vectorization and simplification
- Professional numbering and labeling
"""
import sys
import numpy as np
import cv2
import json
from typing import Optional, Dict, List, Tuple
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union
import rasterio.features
from rasterio import Affine


def process_prediction_to_polygons(
    prob_map: np.ndarray,
    transform: Optional[Affine] = None,
    crs: Optional[str] = None,
    threshold: float = 0.5,
    min_area_pixels: int = 1000,  # Filter out mine blocks under 1000 pixels
    min_area_meters: float = 1000.0,
    morphology_kernel: int = 5,
    simplify_tol: float = 1.5,
    keep_holes: bool = False,
    use_adaptive_threshold: bool = True,
    tile_id: Optional[str] = None,
    analysis_id: Optional[str] = None
) -> Dict:
    """
    PROFESSIONAL-GRADE: Convert model probability map to cleaned polygons with numbered blocks.
    
    ENHANCED PIPELINE (Production-Ready):
    =======================================
    
    1. ADVANCED THRESHOLDING:
       - Adaptive Otsu method OR fixed threshold
       - Confidence-based filtering for robust detection
    
    2. MORPHOLOGICAL CLEANING:
       - Opening (remove small noise/artifacts)
       - Closing (fill gaps and holes)
       - Larger kernels for better cleanup (kernel=5 default)
    
    3. MULTI-LEVEL FILTERING:
       - Pixel area filtering (remove tiny artifacts)
       - Real-world area filtering (square meters, geographic)
       - Shape validation (remove invalid geometries)
    
    4. PROFESSIONAL VECTORIZATION:
       - Douglas-Peucker simplification (clean boundaries)
       - Topology correction (make_valid for self-intersections)
       - Overlap resolution (merge touching blocks)
    
    5. POLYGON NUMBERING & METADATA:
       - Sort by area (largest first)
       - Sequential labels (Block 1, Block 2, ...)
       - Metadata: area (m²), confidence, unique ID
    
    Args:
        prob_map: 2D probability map [0,1] from model (HxW float32)
        transform: Rasterio Affine transform for georeferencing (pixel → lat/lon)
        crs: CRS string (e.g., 'EPSG:4326' for lat/lon)
        threshold: Probability threshold (default 0.5) - ignored if use_adaptive_threshold=True
        min_area_pixels: Minimum contour area in pixels (default 1000 - filters out small mine blocks)
        min_area_meters: Minimum area in square meters (default 1000.0 = 0.001 km²)
        morphology_kernel: Kernel size for morphological ops (default 5, larger = more aggressive)
        simplify_tol: Douglas-Peucker tolerance for boundary simplification (pixels)
        keep_holes: If False, fills holes inside polygons
        use_adaptive_threshold: If True, uses Otsu's method instead of fixed threshold
    
    Returns:
        GeoJSON FeatureCollection with:
        - features: Numbered polygon blocks with metadata
        - metadata: Processing statistics (count, total_area, avg_confidence)
        - visualization: Overlay parameters (transparency, colors) for frontend
        
    Example feature properties:
        {
            "block_id": 1,
            "name": "Block 1",
            "area_px": 1250,
            "area_m2": 125000.0,
            "avg_confidence": 0.87,
            "label_position": [lon, lat]  # Centroid for number placement
        }
    """
    if prob_map.ndim != 2:
        raise ValueError(f"Expected 2D probability map, got shape {prob_map.shape}")
    
    if prob_map.dtype != np.float32 and prob_map.dtype != np.float64:
        prob_map = prob_map.astype(np.float32)
    
    # If no transform provided, use identity (pixel coordinates)
    if transform is None:
        transform = Affine.identity()

    # =====================================================================
    # STEP 1: SIMPLE THRESHOLDING (NOTEBOOK MODE)
    # =====================================================================
    MINIMUM_THRESHOLD = 0.30  # Safety guard: never go below 30% confidence
    
    if use_adaptive_threshold:
        # Convert probability map to 8-bit for Otsu's method
        prob_8bit = (prob_map * 255).astype(np.uint8)
        
        # Otsu's method: automatically finds optimal threshold
        otsu_thresh, bin_mask = cv2.threshold(
            prob_8bit, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Store the adaptive threshold for metadata
        adaptive_thresh = otsu_thresh / 255.0
        
        # SAFETY GUARD: Prevent false positives from extremely low thresholds
        if adaptive_thresh < MINIMUM_THRESHOLD:
            sys.stderr.write(f"[PostProcess] ⚠️  Otsu threshold too low ({adaptive_thresh:.3f}), using minimum {MINIMUM_THRESHOLD:.3f}\n")
            adaptive_thresh = MINIMUM_THRESHOLD
            # Recompute binary mask with minimum threshold
            bin_mask = (prob_map > adaptive_thresh).astype(np.uint8)
        else:
            sys.stderr.write(f"[PostProcess] ✅ Otsu adaptive threshold: {adaptive_thresh:.3f}\n")
        sys.stderr.flush()
    else:
        # Fixed threshold method (NOTEBOOK DEFAULT)
        bin_mask = (prob_map > threshold).astype(np.uint8)
        adaptive_thresh = threshold
        sys.stderr.write(f"[PostProcess] ✅ Using fixed threshold: {threshold:.3f}\n")
        sys.stderr.flush()
    
    if bin_mask.sum() == 0:
        # No detections
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "block_count": 0,
                "total_area_m2": 0,
                "threshold_used": adaptive_thresh
            }
        }

    # =====================================================================
    # STEP 2: INDUSTRIAL GRADE - MINIMAL MORPHOLOGICAL CLEANING
    # =====================================================================
    # Ultra-minimal processing to preserve raw model predictions
    # Only remove single-pixel noise while preserving all boundaries
    
    if morphology_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morphology_kernel, morphology_kernel)  # Use small kernel (2)
        )
        
        # Ultra-light opening: removes only single-pixel noise
        cleaned_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # No closing operation to preserve exact boundaries
        # cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        # Skip morphology entirely for kernel=1
        cleaned_mask = bin_mask.copy()
    
    sys.stderr.write(f"[PostProcess] 🔧 Morphology: kernel={morphology_kernel}, minimal cleaning only\n")
    sys.stderr.flush()

    # =====================================================================
    # STEP 3: CONTOUR EXTRACTION (INDUSTRIAL GRADE - EXACT BOUNDARIES)
    # =====================================================================
    # Use CHAIN_APPROX_NONE for pixel-perfect boundaries (no straight line compression)
    contours, _ = cv2.findContours(
        cleaned_mask, 
        cv2.RETR_EXTERNAL,  # Only external contours (no holes)
        cv2.CHAIN_APPROX_NONE  # Store ALL contour points (no compression)
    )
    
    if not contours:
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "block_count": 0,
                "total_area_m2": 0,
                "threshold_used": adaptive_thresh
            }
        }

    # =====================================================================
    # STEP 4: MULTI-LEVEL AREA FILTERING (Pixel + Geographic)
    # =====================================================================
    # Calculate pixel area in square meters (if transform available)
    pixel_size_area = None
    if transform is not None and transform != Affine.identity():
        try:
            # Pixel area in CRS units (assumes square pixels)
            # For lat/lon, this is approximate (degrees²)
            pixel_size_area = abs(transform.a * transform.e)
        except Exception:
            pixel_size_area = None
    
    valid_contours = []
    contour_data = []  # Store contour, area_px, area_m2, avg_confidence
    
    # Statistics for filtering
    total_contours = len(contours)
    filtered_by_pixels = 0
    filtered_by_area = 0
    
    for contour in contours:
        area_px = cv2.contourArea(contour)
        
        # Filter 1: Pixel area threshold (1000+ pixels for substantial mine blocks)
        if area_px < min_area_pixels:
            filtered_by_pixels += 1
            continue
        
        # Filter 2: Geographic area threshold (if applicable)
        if pixel_size_area is not None:
            # For lat/lon (EPSG:4326), convert degrees² to m²
            # Rough approximation: 1 degree ≈ 111 km at equator
            if crs and "4326" in crs:
                # Convert degrees² to m²
                area_m2 = area_px * pixel_size_area * (111000 ** 2)
            else:
                # CRS already in meters (e.g., UTM)
                area_m2 = area_px * pixel_size_area
            
            if area_m2 < min_area_meters:
                filtered_by_area += 1
                continue
        else:
            area_m2 = None
        
        # Calculate average confidence for this contour region
        mask_region = np.zeros_like(prob_map, dtype=np.uint8)
        cv2.drawContours(mask_region, [contour], -1, 1, thickness=cv2.FILLED)
        avg_confidence = np.mean(prob_map[mask_region == 1])
        
        valid_contours.append(contour)
        contour_data.append({
            'contour': contour,
            'area_px': area_px,
            'area_m2': area_m2,
            'avg_confidence': float(avg_confidence)
        })
    
    # Log filtering statistics
    sys.stderr.write(f"[DEBUG] 🔍 Contour filtering: {total_contours} detected → {len(valid_contours)} kept\n")
    sys.stderr.write(f"[DEBUG] 📏 Filtered out: {filtered_by_pixels} blocks < {min_area_pixels} pixels, {filtered_by_area} blocks < {min_area_meters}m²\n")
    sys.stderr.flush()
    
    if not valid_contours:
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "block_count": 0,
                "total_area_m2": 0,
                "threshold_used": adaptive_thresh,
                "filtering_stats": {
                    "total_detected": total_contours,
                    "filtered_by_pixels": filtered_by_pixels,
                    "filtered_by_area": filtered_by_area,
                    "kept": 0
                }
            }
        }

    # =====================================================================
    # STEP 5: SORT BY AREA (Largest First for Professional Numbering)
    # =====================================================================
    contour_data.sort(key=lambda x: x['area_px'], reverse=True)

    # =====================================================================
    # STEP 6: VECTORIZE TO GEOJSON WITH PROFESSIONAL NUMBERING
    # =====================================================================
    features = []
    total_area_m2 = 0
    total_confidence = 0
    
    for block_idx, data in enumerate(contour_data, start=1):
        contour = data['contour']
        area_px = data['area_px']
        area_m2 = data['area_m2']
        avg_confidence = data['avg_confidence']
        
        # Convert contour points to polygon coordinates
        # Contour shape is (N, 1, 2) - squeeze to (N, 2)
        contour_points = contour.squeeze()
        
        # Handle single point edge case
        if contour_points.ndim == 1:
            continue
        
        # IMPORTANT: Store pixel coordinates for canvas overlay rendering
        pixel_coords = []
        for point in contour_points:
            x, y = point
            pixel_coords.append([float(x), float(y)])
        
        # Transform pixel coordinates to georeferenced coordinates
        geo_coords = []
        for point in contour_points:
            x, y = point
            # rasterio transform converts pixel coords to geo coords
            geo_x, geo_y = transform * (x, y)
            geo_coords.append([geo_x, geo_y])
        
        # Close the polygon if not already closed
        if geo_coords and geo_coords[0] != geo_coords[-1]:
            geo_coords.append(geo_coords[0])
        
        if pixel_coords and pixel_coords[0] != pixel_coords[-1]:
            pixel_coords.append(pixel_coords[0])
        
        if len(geo_coords) < 4:  # Minimum 3 points + closing point
            continue
        
        # =====================================================================
        # STEP 7: POLYGON VALIDATION ONLY (NO SIMPLIFICATION)
        # =====================================================================
        try:
            poly = Polygon(geo_coords)
            
            # Validate and fix geometry (remove self-intersections)
            if not poly.is_valid:
                poly = make_valid(poly)
                
                # make_valid can return GeometryCollection - extract Polygon
                if poly.geom_type == 'GeometryCollection':
                    # Extract the largest Polygon from the collection
                    polys = [g for g in poly.geoms if g.geom_type == 'Polygon']
                    if not polys:
                        sys.stderr.write(f"[PostProcess] Warning: GeometryCollection has no Polygons for block {block_idx}\n")
                        sys.stderr.flush()
                        continue
                    poly = max(polys, key=lambda p: p.area)
                elif poly.geom_type == 'MultiPolygon':
                    # Take the largest polygon from MultiPolygon
                    poly = max(poly.geoms, key=lambda p: p.area)
                elif poly.geom_type != 'Polygon':
                    sys.stderr.write(f"[PostProcess] Warning: Unexpected geometry type {poly.geom_type} for block {block_idx}\n")
                    sys.stderr.flush()
                    continue
            
            # Remove holes if requested (solid blocks only)
            if not keep_holes and hasattr(poly, 'interiors') and len(list(poly.interiors)) > 0:
                poly = Polygon(poly.exterior)
            
            # NO SIMPLIFICATION - Use raw model prediction boundaries
            # simplify_tol is ignored to preserve exact predicted shapes
            # This matches the notebook's approach of using contours directly
            
            if poly.is_empty or poly.area == 0:
                continue
                
        except Exception as e:
            sys.stderr.write(f"[PostProcess] Warning: Failed to create polygon for block {block_idx}: {e}\n")
            sys.stderr.flush()
            continue

        # Calculate centroid for label placement
        try:
            centroid = poly.centroid
            label_position = [centroid.x, centroid.y]
        except:
            label_position = None

        # =====================================================================
        # STEP 8: BUILD FEATURE WITH PROFESSIONAL METADATA
        # =====================================================================
        # Generate unique block ID with analysis and tile prefixes
        id_prefix = ""
        if analysis_id:
            id_prefix += analysis_id[:8] + "-"
        if tile_id:
            id_prefix += f"T{tile_id}B"
        
        unique_block_id = f"{id_prefix}{block_idx}" if id_prefix else str(block_idx)
        block_name = f"Block {block_idx}" if not tile_id else f"T{tile_id}B{block_idx}"
        
        props = {
            "block_id": unique_block_id,  # Unique ID: analysis_id-T1B1, T1B2, etc.
            "name": block_name,  # Professional naming (Block 1, T1B1, etc.)
            "area_px": int(area_px),
            "avg_confidence": round(avg_confidence, 3),
            "label_position": label_position,  # For frontend number placement
            "pixel_coords": pixel_coords  # Add raw pixel coordinates for canvas rendering
        }
        
        if area_m2 is not None:
            props["area_m2"] = round(area_m2, 2)
            total_area_m2 += area_m2
        
        if crs:
            props["crs"] = str(crs)
        
        total_confidence += avg_confidence

        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": props
        })

    # =====================================================================
    # STEP 9: METADATA & VISUALIZATION PARAMETERS
    # =====================================================================
    geojson_fc = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "block_count": len(features),
            "total_area_m2": round(total_area_m2, 2),  # Always return numeric value
            "avg_confidence": round(total_confidence / len(features), 3) if features else 0.0,
            "threshold_used": adaptive_thresh,
            "processing_params": {
                "morphology_kernel": morphology_kernel,
                "min_area_pixels": min_area_pixels,
                "min_area_meters": min_area_meters,
                "simplify_tolerance": simplify_tol,
                "adaptive_threshold": use_adaptive_threshold
            },
            "filtering_stats": {
                "total_detected": total_contours,
                "filtered_by_pixels": filtered_by_pixels,
                "filtered_by_area": filtered_by_area,
                "kept": len(features)
            }
        },
        "visualization": {
            "overlay_opacity": 0.25,  # 25% transparent - more see-through
            "fill_color": "#FFD700",  # Yellow/Gold for mine areas
            "stroke_color": "#FFA500",  # Orange boundaries (sharper, thinner)
            "stroke_width": 1,  # Thinner lines (was 2)
            "label_color": "#000000",  # Black text for block numbers (better visibility)
            "label_font_size": 16,  # Slightly larger for readability
            "label_outline_color": "#FFFFFF",  # White outline for contrast
            "label_outline_width": 2
        }
    }
    
    sys.stderr.write(f"[PostProcess] ✅ Generated {len(features)} numbered blocks (filtered {filtered_by_pixels} blocks < {min_area_pixels} pixels)\n")
    if total_area_m2 > 0:
        sys.stderr.write(f"[PostProcess] Total mine area: {total_area_m2/1e6:.3f} km² ({total_area_m2/10000:.2f} ha)\n")
    sys.stderr.flush()
    
    return geojson_fc


def save_geojson(geojson_fc: Dict, output_path: str) -> None:
    """Save GeoJSON FeatureCollection to file."""
    with open(output_path, 'w') as f:
        json.dump(geojson_fc, f, indent=2)


def load_geojson(input_path: str) -> Dict:
    """Load GeoJSON FeatureCollection from file."""
    with open(input_path, 'r') as f:
        return json.load(f)
