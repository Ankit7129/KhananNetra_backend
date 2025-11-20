"""
Real-time analysis router with parallel model loading and tile streaming.
This version loads the model in parallel while fetching and displaying RGB tiles in real-time.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Tuple, Optional
import asyncio
import uuid
import os
import tempfile
import rasterio
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import numpy as np
import base64
import io

from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import array_bounds, from_bounds
from PIL import Image

from app.models.schemas import AnalysisRequest, AOIGeometry
from app.services.earth_engine_service import get_earth_engine_service
from app.services.ml_inference_service import get_ml_service
from app.services.tile_fetching_service import TileFetchingService

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for analysis results - with automatic cleanup
analysis_results: Dict[str, dict] = {}
RESULT_RETENTION_HOURS = 24  # Keep results for 24 hours, then auto-cleanup
def _ensure_tile_transform(tile: dict) -> Tuple[np.ndarray, object, str]:
    """Ensure each tile has transform and CRS for mosaicking."""
    data = tile.get("data")
    if data is None:
        raise ValueError("Tile is missing data array for mosaicking")

    transform = tile.get("transform")
    crs = tile.get("crs") or "EPSG:4326"

    if transform is None:
        bounds = tile.get("bounds")
        if not bounds or len(bounds) < 4:
            raise ValueError("Tile is missing bounds required to derive transform")

        min_lon = min(pt[0] for pt in bounds)
        max_lon = max(pt[0] for pt in bounds)
        min_lat = min(pt[1] for pt in bounds)
        max_lat = max(pt[1] for pt in bounds)
        height, width = data.shape[0], data.shape[1]
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    return data, transform, crs


def _serialize_transform(transform: Optional[object]) -> Optional[List[float]]:
    """Convert a rasterio Affine transform into a JSON-serializable list."""
    if transform is None:
        return None
    try:
        if hasattr(transform, 'to_gdal'):
            values = list(transform.to_gdal())
        else:
            values = list(transform)
        return [float(value) for value in values]
    except Exception:
        return None


def _encode_rgb_mosaic(
    mosaic_array: np.ndarray,
    clip_min: float = 0.0,
    clip_max: float = 3000.0,
    max_dimension: int = 1536,
) -> Optional[str]:
    """Generate a true-colour PNG preview (base64) from the 6-band mosaic array."""
    if mosaic_array.ndim != 3 or mosaic_array.shape[2] < 3:
        return None

    rgb_stack = mosaic_array[:, :, [2, 1, 0]]  # B4 (R), B3 (G), B2 (B)
    if clip_max <= clip_min:
        clip_max = clip_min + 1.0

    scaled = np.clip((rgb_stack - clip_min) / (clip_max - clip_min), 0.0, 1.0)
    rgb_uint8 = (scaled * 255).astype(np.uint8)

    image = Image.fromarray(rgb_uint8, mode='RGB')
    if max(image.size) > max_dimension:
        resampling = getattr(Image, 'Resampling', None)
        resample_filter = getattr(resampling, 'LANCZOS', getattr(Image, 'LANCZOS', Image.BICUBIC)) if resampling else getattr(Image, 'LANCZOS', Image.BICUBIC)
        image.thumbnail((max_dimension, max_dimension), resample=resample_filter)

    buffer = io.BytesIO()
    image.save(buffer, format='PNG', optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def build_mosaic_from_tiles(tile_data_list: List[dict]) -> Tuple[np.ndarray, object, str, List[List[float]]]:
    """Merge individual tile arrays into a single georeferenced mosaic."""
    if not tile_data_list:
        raise ValueError("No tile data available to build mosaic")

    datasets = []
    memfiles = []

    try:
        target_crs = None

        for tile in tile_data_list:
            if tile.get("data") is None:
                continue

            data, transform, crs = _ensure_tile_transform(tile)
            if target_crs is None:
                target_crs = crs

            height, width, bands = data.shape
            mem = MemoryFile()
            dataset = mem.open(
                driver="GTiff",
                height=height,
                width=width,
                count=bands,
                dtype=data.dtype,
                transform=transform,
                crs=crs
            )
            dataset.write(np.moveaxis(data, -1, 0))
            datasets.append(dataset)
            memfiles.append(mem)

        if not datasets:
            raise ValueError("Tile dataset list is empty after filtering missing data")

        mosaic_data, mosaic_transform = merge(datasets, method="first")
        mosaic_array = np.moveaxis(mosaic_data, 0, -1).astype(np.float32, copy=False)

        first_dataset = datasets[0]
        mosaic_crs = first_dataset.crs.to_string() if first_dataset.crs else (target_crs or "EPSG:4326")

        height, width = mosaic_array.shape[0], mosaic_array.shape[1]
        min_x, min_y, max_x, max_y = array_bounds(height, width, mosaic_transform)
        mosaic_bounds = [
            [float(min_x), float(min_y)],
            [float(max_x), float(min_y)],
            [float(max_x), float(max_y)],
            [float(min_x), float(max_y)],
            [float(min_x), float(min_y)]
        ]

        return mosaic_array, mosaic_transform, mosaic_crs, mosaic_bounds

    finally:
        for dataset in datasets:
            try:
                dataset.close()
            except Exception:
                pass
        for mem in memfiles:
            try:
                mem.close()
            except Exception:
                pass

# Use a dedicated thread pool for analysis to prevent blocking the event loop
ANALYSIS_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analysis")

def cleanup_analysis_results():
    """Cleanup old analysis results to prevent memory leaks"""
    global analysis_results
    now = datetime.now()
    expired_ids = []
    
    for analysis_id, result in analysis_results.items():
        try:
            created_at = result.get("created_at")
            if created_at:
                created_time = datetime.fromisoformat(created_at)
                if now - created_time > timedelta(hours=RESULT_RETENTION_HOURS):
                    expired_ids.append(analysis_id)
        except:
            pass
    
    # Remove expired results
    for analysis_id in expired_ids:
        try:
            # Clear large data structures before deletion
            if "tiles" in analysis_results[analysis_id]:
                analysis_results[analysis_id]["tiles"] = []
            del analysis_results[analysis_id]
            logger.info(f"🧹 Cleaned up old analysis result: {analysis_id}")
        except:
            pass
    
    # Force garbage collection if we cleaned up anything
    if expired_ids:
        gc.collect()

@router.get("/cleanup")
async def cleanup_endpoint():
    """Manually trigger cleanup of old analysis results"""
    cleanup_analysis_results()
    return {
        "status": "success",
        "remaining_analyses": len(analysis_results),
        "message": "Cleanup completed"
    }

@router.get("/stats")
async def get_stats():
    """Get backend statistics for monitoring"""
    import psutil
    import sys
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory": {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
            "percent": process.memory_percent()
        },
        "analysis_results_stored": len(analysis_results),
        "thread_pool_workers": ANALYSIS_EXECUTOR._max_workers
    }

@router.post("/start")
async def start_analysis_realtime(request: AnalysisRequest):
    """Start analysis with real-time tile streaming and parallel model loading"""
    try:
        from app.services.geospatial_service import geospatial_service
        
        analysis_id = str(uuid.uuid4())
        
        # Fetch the AOI using aoi_id
        aoi = geospatial_service.get_aoi(request.aoi_id)
        if not aoi:
            raise HTTPException(status_code=404, detail=f"AOI {request.aoi_id} not found")
        
        aoi_geometry = aoi.geometry
        
        # Store initial status with creation timestamp
        analysis_results[analysis_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Initializing analysis...",
            "current_step": "initialization",
            "tiles": [],  # Real-time RGB tiles
            "created_at": datetime.now().isoformat(),  # For auto-cleanup
            "last_accessed": datetime.now().isoformat()  # Track activity
        }
        
        # Start processing in background thread (not event loop task)
        # This prevents analysis from blocking the event loop and other requests
        # Wrap in safety handler to prevent crashes
        async def safe_process():
            try:
                await process_analysis_realtime(analysis_id, aoi_geometry)
            except Exception as process_error:
                logger.error(f"❌❌❌ CRITICAL: Analysis {analysis_id} crashed: {process_error}")
                import traceback
                traceback.print_exc()
                # Update status to failed instead of crashing
                if analysis_id in analysis_results:
                    analysis_results[analysis_id].update({
                        "status": "failed",
                        "message": f"Analysis crashed: {str(process_error)}",
                        "current_step": "error",
                        "progress": 0
                    })
                # Force cleanup
                gc.collect()
        
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            ANALYSIS_EXECUTOR,
            lambda: asyncio.run(safe_process())
        )
        
        print(f"\n{'='*60}")
        print(f"🚀 STARTING REAL-TIME ANALYSIS")
        print(f"Analysis ID: {analysis_id}")
        print(f"AOI ID: {request.aoi_id}")
        print(f"AOI Type: {aoi_geometry.type}")
        print(f"{'='*60}\n")
        
        return {
            "analysis_id": analysis_id,
            "status": "started",
            "message": "Analysis started - tiles will stream in real-time"
        }
    
    except Exception as e:
        print(f"❌ Error starting analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Get current analysis status - MUST respond immediately, NO BLOCKING
    Also tracks last access time and triggers cleanup periodically
    """
    # Periodically cleanup old results (every 100 requests)
    if len(analysis_results) > 0 and len(analysis_results) % 100 == 0:
        cleanup_analysis_results()
    
    # This endpoint MUST be super fast - just a dict lookup
    if analysis_id not in analysis_results:
        # Return helpful message instead of generic 404
        available_ids = list(analysis_results.keys())[:5]
        raise HTTPException(
            status_code=404, 
            detail={
                "message": "Analysis not found",
                "analysis_id": analysis_id,
                "total_active_analyses": len(analysis_results),
                "available_ids": available_ids,
                "hint": "Start a new analysis from the frontend by drawing an AOI and clicking 'Analyze'"
            }
        )
    
    # Update last accessed time for cleanup tracking
    try:
        analysis_results[analysis_id]["last_accessed"] = datetime.now().isoformat()
    except:
        pass  # Non-critical, ignore errors
    
    # Return immediately from dictionary
    return analysis_results[analysis_id]


async def process_analysis_realtime(analysis_id: str, aoi_geometry: AOIGeometry):
    """Process analysis with real-time tile streaming and parallel model loading"""
    temp_dir = None
    model_load_task = None
    executor = ThreadPoolExecutor(max_workers=1)
    
    try:
        # Step 1: Validate AOI area (5%)
        print(f"\n{'='*50}")
        print(f"Step 1: Validating AOI")
        print(f"{'='*50}")
        analysis_results[analysis_id].update({
            "progress": 5,
            "message": "Validating Area of Interest...",
            "current_step": "validating"
        })
        
        # Initialize EE early for area calculation
        import ee
        try:
            ee.Initialize(project='mining-detection')
            print(f"   ✅ Earth Engine initialized for validation")
        except Exception as ee_error:
            print(f"   ℹ️  EE already initialized")
        
        # Calculate area
        try:
            geometry_dict = {
                'type': aoi_geometry.type,
                'coordinates': aoi_geometry.coordinates
            }
            
            if aoi_geometry.type == 'Polygon':
                ee_coords = [[(x, y) for x, y, *_ in ring] for ring in aoi_geometry.coordinates]
                temp_geom = ee.Geometry.Polygon(ee_coords)
            else:
                temp_geom = ee.Geometry.Polygon(aoi_geometry.coordinates[0])
            
            area_m2 = temp_geom.area().getInfo()
            area_km2 = area_m2 / 1_000_000
            
            print(f"   📏 AOI area: {area_km2:.2f} km²")
            
            if area_km2 > 200:
                raise ValueError(f"AOI too large ({area_km2:.1f} km²). Max: 200 km²")
            elif area_km2 > 100:
                print(f"   ⚠️  Large area - may take 20-30 minutes")
            
            analysis_results[analysis_id]["area_km2"] = round(area_km2, 2)
            
        except ValueError:
            raise
        except Exception as e:
            print(f"   ⚠️  Could not calculate area: {e}")
        
        print(f"✅ AOI validated")
        
        # Step 2: Initialize services (10%)
        print(f"\n{'='*50}")
        print(f"Step 2: Initializing services")
        print(f"{'='*50}")
        analysis_results[analysis_id].update({
            "progress": 10,
            "message": "Connecting to Google Earth Engine...",
            "current_step": "connecting"
        })
        
        # Earth Engine is initialized at startup - just get the service
        ee_service = get_earth_engine_service()
        tile_service = TileFetchingService(ee_service)
        print(f"✅ Services ready")
        
        # Step 3: Calculate tile grid (15%)
        print(f"\n{'='*50}")
        print(f"Step 3: Calculating tile grid")
        print(f"{'='*50}")
        analysis_results[analysis_id].update({
            "progress": 15,
            "message": "Calculating optimal tile grid...",
            "current_step": "requesting"
        })
        
        tiles = await asyncio.to_thread(tile_service.calculate_tile_grid, geometry_dict)
        print(f"✅ Tile grid: {len(tiles)} tiles")
        
        # CRITICAL: Adaptive tile limit based on available system memory
        # Can process up to 25 tiles with aggressive memory management
        MAX_TILES = 25  # Increased from 12 - system is robust now with 7-layer defense
        if len(tiles) > MAX_TILES:
            print(f"⚠️  Warning: {len(tiles)} tiles requested, limiting to {MAX_TILES} for stability")
            tiles = tiles[:MAX_TILES]
            print(f"   📌 Processing first {len(tiles)} tiles only (can increase if needed)")
        
        # Update progress: Grid calculated (18%)
        analysis_results[analysis_id].update({
            "progress": 18,
            "message": f"Grid calculated: {len(tiles)} tiles to process",
            "current_step": "requesting", 
            "total_tiles": len(tiles)
        })
        
        # Step 4: Fetch tiles in REAL-TIME (20% - 65%)
        print(f"\n{'='*50}")
        print(f"Step 4: Fetching satellite imagery tiles in REAL-TIME")
        print(f"{'='*50}")
        analysis_results[analysis_id].update({
            "progress": 20,
            "message": f"Fetching satellite tiles...",
            "current_step": "preprocessing",
            "total_tiles": len(tiles),
            "tiles_fetched": 0
        })
        
        def tile_callback(callback_data):
            """Callback to update progress as each tile is fetched"""
            tile_info = callback_data['tile']
            current = callback_data['current']
            total = callback_data['total']
            
            progress = 20 + int((current / total) * 45)  # 20% to 65%
            
            # Add tile to results immediately (even if failed)
            tile_to_add = {
                "tile_id": tile_info.get("tile_id", tile_info.get("id")),
                "bounds": tile_info.get("bounds", []),
                "image_base64": tile_info.get("image_base64", None),
                "row": tile_info.get("row", 0),
                "col": tile_info.get("col", 0),
                "status": tile_info.get("status", "unknown"),
                "error": tile_info.get("error", None),
                "bands_used": tile_info.get("bands", ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']),
                "cloud_coverage": 0,  # TODO: Get from tile_info if available
                "timestamp": tile_info.get("timestamp", ""),
                "transform": _serialize_transform(tile_info.get("transform")),
                "crs": tile_info.get("crs")
            }
            
            # Debug: Log what we're adding
            print(f"   🔍 DEBUG: Adding tile with bounds: {tile_to_add['bounds'][:2] if tile_to_add['bounds'] else 'None'}")
            print(f"   🔍 DEBUG: Has image: {tile_to_add['image_base64'] is not None}")
            
            analysis_results[analysis_id]["tiles"].append(tile_to_add)
            
            analysis_results[analysis_id].update({
                "progress": progress,
                "message": f"Fetching tile {current}/{total}...",
                "current_step": "preprocessing",
                "total_tiles": total,
                "tiles_fetched": current
            })
            
            print(f"   ✅ Tile {current}/{total} fetched (Row {tile_info['row']}, Col {tile_info['col']})")
            print(f"   🔍 DEBUG: Total tiles in results now: {len(analysis_results[analysis_id]['tiles'])}")
        
        # Fetch all tiles with real-time updates
        print(f"   📡 Starting real-time tile download...")
        
        # Update progress: Starting tile fetch (20%)
        analysis_results[analysis_id].update({
            "progress": 20,
            "message": "Starting satellite tile download...",
            "current_step": "preprocessing"
        })
        
        all_tiles = await asyncio.to_thread(
            tile_service.fetch_all_tiles_realtime,
            geometry_dict,
            tiles,
            callback=tile_callback
        )
        
        # Update progress: All tiles fetched (65%)
        analysis_results[analysis_id].update({
            "progress": 65,
            "message": f"✅ All {len(all_tiles)} tiles fetched!",
            "current_step": "preprocessing"
        })
        
        print(f"✅ All {len(all_tiles)} RGB tiles fetched and displayed!")
        
        # Step 5: Get ML service instance (70%)
        print(f"\n{'='*50}")
        print(f"Step 5: Initializing ML service")
        print(f"{'='*50}")
        analysis_results[analysis_id].update({
            "progress": 70,
            "message": "Preparing ML inference engine...",
            "current_step": "processing"
        })
        
        # Get ML service instance (should be fast since ML model loads in subprocess)
        print(f"   🔍 Getting ML service instance...")
        ml_service = get_ml_service()
        print(f"   ✅ ML service ready (inference will run in subprocess)")
        
        # Update progress: Service ready (76%)
        analysis_results[analysis_id].update({
            "progress": 76,
            "message": "ML service ready - starting analysis...",
            "current_step": "processing"
        })
        
        # Step 6: Run ML inference on unified mosaic (80%)
        print(f"\n{'='*50}")
        print(f"Step 6: Running ML inference on unified mosaic")
        print(f"{'='*50}")
        analysis_results[analysis_id].update({
            "progress": 80,
            "message": "🔬 Assembling satellite mosaic for AI analysis...",
            "current_step": "ml_inference_tiles"
        })

        print(f"   🔍 DEBUG: Fetching cached 6-band tile data for mosaicking...")
        tile_6band_data = await asyncio.to_thread(tile_service.get_6band_tile_data)
        print(f"    DEBUG: Found {len(tile_6band_data)} tiles with 6-band data")

        if len(tile_6band_data) == 0:
            print(f"   ❌ DEBUG: No 6-band tile data available for ML inference")
            analysis_results[analysis_id].update({
                "status": "error",
                "progress": 80,
                "message": "❌ No satellite data available for AI analysis",
                "current_step": "error"
            })
            raise ValueError("No 6-band tile data available for ML inference")

        # Build geospatially aligned mosaic before inference
        try:
            mosaic_array, mosaic_transform, mosaic_crs, mosaic_bounds = await asyncio.to_thread(
                build_mosaic_from_tiles,
                tile_6band_data
            )
        except Exception as mosaic_error:
            print(f"   ❌ Failed to build mosaic: {mosaic_error}")
            analysis_results[analysis_id].update({
                "status": "error",
                "progress": 80,
                "message": "❌ Could not assemble satellite mosaic for analysis",
                "current_step": "error"
            })
            raise

        print(f"   ✅ Mosaic assembled with shape {mosaic_array.shape} and CRS {mosaic_crs}")

        analysis_results[analysis_id].update({
            "progress": 82,
            "message": "🤖 Running AI analysis on merged mosaic...",
            "current_step": "processing"
        })

        inference_payload = {
            'image_data': mosaic_array,
            'tile_index': 0,
            'tile_id': 'mosaic',
            'analysis_id': analysis_id,
            'bounds': mosaic_bounds,
            'transform': mosaic_transform,
            'crs': mosaic_crs
        }

        mosaic_result = await asyncio.to_thread(ml_service.run_inference_on_tile, inference_payload)

        if not mosaic_result.get('success', False):
            error_message = mosaic_result.get('error', 'Unknown error during mosaic inference')
            print(f"   ❌ Mosaic inference failed: {error_message}")
            analysis_results[analysis_id].update({
                "status": "error",
                "progress": 82,
                "message": f"❌ AI analysis failed: {error_message}",
                "current_step": "error"
            })
            raise ValueError(error_message)

        mining_pixels = mosaic_result.get('mining_pixels', 0)
        total_pixels = mosaic_result.get('total_pixels', int(mosaic_array.shape[0] * mosaic_array.shape[1]))
        mining_percentage = mosaic_result.get('mining_percentage', 0.0)
        confidence_primary = mosaic_result.get('confidence', 0.0) / 100.0
        max_pred = mosaic_result.get('max_prediction', 0.0)
        mean_pred = mosaic_result.get('mean_prediction', 0.0)
        confidence_value = max(confidence_primary, max_pred, mean_pred)
        mine_blocks = mosaic_result.get('mine_blocks') or []
        num_mine_blocks = mosaic_result.get('num_mine_blocks', len(mine_blocks))
        total_area_m2 = mosaic_result.get('total_area_m2')
        mask_shape = mosaic_result.get('mask_shape', [mosaic_array.shape[0], mosaic_array.shape[1]])
        prob_map_base64 = mosaic_result.get('probability_map_base64')
        mining_detected = mining_pixels > 0

        print("   ✅ Mosaic inference complete")
        print(f"      📊 Mining pixels: {mining_pixels}/{total_pixels}")
        print(f"      📊 Mining coverage: {mining_percentage:.2f}%")
        print(f"      📊 Mine blocks detected: {num_mine_blocks}")
        print(f"      📊 Confidence (peak/mean): {max_pred:.3f}/{mean_pred:.3f}")

        serialized_transform = _serialize_transform(mosaic_transform)
        mosaic_rgb_base64 = _encode_rgb_mosaic(mosaic_array)

        tile_predictions = [{
            'tile_id': 'mosaic',
            'mining_detected': mining_detected,
            'mining_percentage': mining_percentage,
            'mining_pixels': mining_pixels,
            'bounds': mosaic_bounds,
            'confidence': confidence_value,
            'mine_blocks': mine_blocks,
            'num_mine_blocks': num_mine_blocks,
            'total_area_m2': total_area_m2,
            'mask_shape': mask_shape,
            'probability_map_base64': prob_map_base64,
            'image_base64': mosaic_rgb_base64,
            'transform': serialized_transform,
            'crs': mosaic_crs
        }]

        analysis_results[analysis_id].update({
            "progress": 86,
            "message": "✅ Mosaic analysis complete",
            "current_step": "processing",
            "ml_progress": {
                "current": 1,
                "total": 1,
                "currentTileId": "mosaic"
            }
        })

        # Append unified mosaic overlay for frontend visualization
        mosaic_center_lat = (mosaic_bounds[0][1] + mosaic_bounds[2][1]) / 2
        mosaic_center_lon = (mosaic_bounds[0][0] + mosaic_bounds[1][0]) / 2
        visualization_index = len(analysis_results[analysis_id]["tiles"]) + 1
        mosaic_tile_entry = {
            'id': 'mosaic',
            'tile_id': 'mosaic',
            'index': visualization_index,
            'row': None,
            'col': None,
            'coordinates': {
                'lat': mosaic_center_lat,
                'lng': mosaic_center_lon
            },
            'bounds': mosaic_bounds,
            'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
            'bands_used': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
            'cloudCoverage': 0,
            'cloud_coverage': 0,
            'timestamp': datetime.now().isoformat(),
            'size': f"{mosaic_array.shape[1]}x{mosaic_array.shape[0]}",
            'status': 'mosaic',
            'miningDetected': mining_detected,
            'mining_detected': mining_detected,
            'miningPercentage': mining_percentage,
            'mining_percentage': mining_percentage,
            'confidence': confidence_value,
            'mine_blocks': mine_blocks,
            'num_mine_blocks': num_mine_blocks,
            'total_area_m2': total_area_m2,
            'mask_shape': mask_shape,
            'probability_map_base64': prob_map_base64,
            'image_base64': mosaic_rgb_base64,
            'transform': serialized_transform,
            'crs': mosaic_crs
        }
        analysis_results[analysis_id]["tiles"].append(mosaic_tile_entry)

        # Aggregate summary statistics for downstream consumers
        source_tile_total = len(all_tiles) if 'all_tiles' in locals() else len(analysis_results[analysis_id]["tiles"])
        mining_area_m2 = total_area_m2 or 0.0

        summary_payload = {
            "analysis_id": analysis_id,
            "total_tiles": source_tile_total,
            "tiles_with_detections": 1 if mining_detected else 0,
            "mining_detected": mining_detected,
            "mine_block_count": num_mine_blocks,
            "mining_pixels": mining_pixels,
            "total_pixels": total_pixels,
            "mining_percentage": mining_percentage,
            "mining_area_m2": mining_area_m2,
            "confidence": confidence_value,
            "mask_shape": mask_shape,
            "bounds": mosaic_bounds,
            "crs": mosaic_crs
        }

        analysis_results[analysis_id]["summary"] = summary_payload
        analysis_results[analysis_id]["total_tiles"] = source_tile_total
        analysis_results[analysis_id]["tiles_with_mining"] = summary_payload["tiles_with_detections"]
        analysis_results[analysis_id]["total_mine_blocks"] = num_mine_blocks
        analysis_results[analysis_id]["total_mining_area_m2"] = mining_area_m2
        analysis_results[analysis_id]["mining_coverage_percentage"] = mining_percentage

        print("\n----- ANALYSIS SUMMARY -----")
        print(f"Analysis ID        : {analysis_id}")
        print(f"Tiles (source/mosaic): {source_tile_total}/1")
        print(f"Mine blocks        : {num_mine_blocks}")
        print(f"Mining coverage    : {mining_percentage:.2f}%")
        print(f"Mining area        : {mining_area_m2/10_000:.2f} ha ({mining_area_m2/1_000_000:.4f} km²)")
        print(f"Confidence (max)   : {confidence_value * 100:.1f}%")
        print(f"Mask dimensions    : {mask_shape[1]} x {mask_shape[0]} pixels")
        print("---------------------------\n")

        # Cleanup large mosaic array to free memory
        del mosaic_array
        gc.collect()

        print(f"✅ ML inference complete on unified mosaic (tiles merged: {len(tile_6band_data)})")
        
        # Update final ML progress (save length before cleanup)
        total_tiles_processed = len(tile_6band_data) if 'tile_6band_data' in locals() else len(tile_predictions)
        analysis_results[analysis_id].update({
            "ml_progress": {
                "current": total_tiles_processed,
                "total": total_tiles_processed,
                "currentTileId": None
            }
        })
        
        # Clean up tile data after inference and after using it
        try:
            if 'tile_6band_data' in locals():
                del tile_6band_data
            gc.collect()
            print(f"   🧹 Cleaned up tile data arrays")
        except Exception as cleanup_error:
            print(f"   ⚠️  Cleanup warning: {cleanup_error}")
        
        # Step 7: Post-process and extract detections (88%)
        print(f"\n{'='*50}")
        print(f"Step 7: Extracting mining locations from tiles")
        print(f"{'='*50}")
        analysis_results[analysis_id].update({
            "progress": 88,
            "message": "🔧 Extracting mining locations...",
            "current_step": "extract_detections"
        })
        
        # Extract detections from each tile
        from math import cos, radians

        detections = []
        detection_id = 1
        
        for tile_pred in tile_predictions:
            if tile_pred['confidence'] < 0.3:
                # Skip tiles with low confidence
                continue
            
            # Get tile bounds [[min_lon, min_lat], [max_lon, max_lat], ...]
            bounds = tile_pred['bounds']
            min_lon = min([b[0] for b in bounds])
            max_lon = max([b[0] for b in bounds])
            min_lat = min([b[1] for b in bounds])
            max_lat = max([b[1] for b in bounds])
            
            # Calculate center of tile
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2
            
            # Calculate tile area in m²
            # Approximate: 1 degree ≈ 111 km at equator
            width_km = (max_lon - min_lon) * 111 * cos(radians(center_lat))
            height_km = (max_lat - min_lat) * 111
            area_m2 = int(width_km * height_km * 1_000_000)
            
            detections.append({
                "id": detection_id,
                "latitude": center_lat,
                "longitude": center_lon,
                "confidence": tile_pred['confidence'],
                "tile_id": tile_pred['tile_id'],
                "area_m2": area_m2,
                "bounds": bounds
            })
            
            detection_id += 1
        
        print(f"✅ Found {len(detections)} potential mining areas across {len(tile_predictions)} tiles")
        
        # Step 8: Merge adjacent polygons across tiles (95%)
        print(f"\n{'='*50}")
        print(f"Step 8: Merging adjacent mine blocks across tiles")
        print(f"{'='*50}")
        analysis_results[analysis_id].update({
            "progress": 95,
            "message": "🔗 Merging adjacent mine blocks...",
            "current_step": "merge_polygons"
        })
        
        # Automatically merge adjacent polygons to eliminate tile boundary artifacts
        merged_blocks_geojson = None
        try:
            from app.utils.merge_polygons import merge_adjacent_mine_blocks
            
            # Only merge if we have tiles with mine blocks
            tiles_with_blocks = [t for t in analysis_results[analysis_id]["tiles"] if t.get('mine_blocks')]
            
            if tiles_with_blocks:
                print(f"🔗 Merging polygons from {len(tiles_with_blocks)} tiles with detections...")
                merged_blocks_geojson = merge_adjacent_mine_blocks(tiles_with_blocks, analysis_id=analysis_id)
                
                merged_count = merged_blocks_geojson['metadata']['merged_block_count']
                original_count = merged_blocks_geojson['metadata']['original_block_count']
                total_area_ha = merged_blocks_geojson['metadata']['total_area_m2'] / 10000
                
                print(f"✅ Merged {original_count} blocks → {merged_count} unified blocks")
                print(f"✅ Total mining area: {total_area_ha:.2f} hectares")
                
                # Store merged blocks in analysis results
                analysis_results[analysis_id]["merged_blocks"] = merged_blocks_geojson
                analysis_results[analysis_id]["merged_block_count"] = merged_count
                analysis_results[analysis_id]["total_mining_area_ha"] = total_area_ha
            else:
                print("📋 No mine blocks to merge")
                
        except Exception as merge_error:
            print(f"⚠️  Polygon merging failed: {merge_error}")
            # Continue without merging - individual tile polygons will be used
        
        # Step 9: Complete (100%)
        print(f"\n{'='*50}")
        print(f"Step 9: Analysis complete!")
        print(f"{'='*50}")
        
        analysis_results[analysis_id].update({
            "status": "completed",
            "progress": 100,
            "message": "✅ Analysis complete!",
            "current_step": "complete",
            "detections": detections,
            "detection_count": len(detections)
        })
        analysis_results[analysis_id]["analysis_id"] = analysis_id
        analysis_results[analysis_id]["completed_at"] = datetime.now().isoformat()
        
        print(f"✅✅✅ Analysis {analysis_id} completed successfully!")
        print(f"   Total tiles displayed: {len(all_tiles)}")
        print(f"   Mining areas detected: {len(detections)}")
        
    except Exception as e:
        print(f"❌ Analysis {analysis_id} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        analysis_results[analysis_id].update({
            "status": "failed",
            "message": f"❌ Error: {str(e)}",
            "current_step": "error"
        })
    
    finally:
        # Comprehensive cleanup to prevent resource leaks
        logger.info(f"🧹 Starting cleanup for analysis {analysis_id}")
        
        try:
            # Cleanup temporary directory
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"   ✅ Temporary directory cleaned")
                except Exception as cleanup_err:
                    logger.warning(f"   ⚠️  Could not remove temp dir: {cleanup_err}")
            
            # Shutdown executor properly to release threads
            if executor:
                executor.shutdown(wait=True)
                logger.info(f"   ✅ Thread executor shutdown")
            
            # Clear tile data from memory (keep metadata for results page)
            if analysis_id in analysis_results:
                try:
                    if "tiles" in analysis_results[analysis_id]:
                        tiles = analysis_results[analysis_id]["tiles"]
                        logger.info(f"   ✅ Preserving {sum(1 for tile in tiles if tile.get('image_base64'))} tile previews for downstream persistence")
                except Exception as clear_err:
                    logger.warning(f"   ⚠️  Error clearing tile data: {clear_err}")
            
            # Force garbage collection
            gc.collect()
            logger.info(f"   ✅ Garbage collection triggered")
            
        except Exception as final_cleanup_err:
            logger.error(f"   ❌ Error during final cleanup: {final_cleanup_err}")
        
        logger.info(f"🧹 Cleanup completed for analysis {analysis_id}")


@router.post("/merge-polygons/{analysis_id}")
async def merge_mine_block_polygons(analysis_id: str):
    """
    Merge adjacent mine block polygons across tiles.
    This creates unified polygons for mining operations that span multiple tiles.
    """
    try:
        from app.utils.merge_polygons import merge_adjacent_mine_blocks
        
        # Get analysis results
        if analysis_id not in analysis_results:
            raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
        
        result = analysis_results[analysis_id]
        
        if result.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Analysis not completed yet")
        
        tiles = result.get("tiles", [])
        if not tiles:
            raise HTTPException(status_code=400, detail="No tiles found in analysis")
        
        print(f"\n🔄 Merging polygons for analysis {analysis_id}")
        print(f"Processing {len(tiles)} tiles...")
        
        # Merge adjacent mine blocks
        merged_geojson = merge_adjacent_mine_blocks(tiles)
        
        print(f"✅ Merged into {merged_geojson['metadata']['merged_block_count']} blocks")
        print(f"   (from {merged_geojson['metadata']['original_block_count']} original blocks)")
        
        return {
            "analysis_id": analysis_id,
            "merged_blocks": merged_geojson,
            "summary": {
                "original_blocks": merged_geojson['metadata']['original_block_count'],
                "merged_blocks": merged_geojson['metadata']['merged_block_count'],
                "total_area_ha": merged_geojson['metadata']['total_area_m2'] / 10000,
                "tiles_processed": merged_geojson['metadata']['tiles_processed']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error merging polygons: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error merging polygons: {str(e)}")
