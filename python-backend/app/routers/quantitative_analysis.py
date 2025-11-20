"""
Quantitative volumetric analysis endpoints.
Transforms detection polygons into DEM-aligned volume metrics.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import math
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
import rasterio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import array_bounds, xy
from rasterio.warp import calculate_default_transform, reproject
from scipy import integrate, ndimage
from shapely.geometry import MultiPolygon, Point, shape
from shapely.ops import transform as shapely_transform, unary_union
import pyproj


router = APIRouter()

SRTM_CACHE_DIR = Path(os.getenv("QUANT_ANALYSIS_DEM_CACHE", tempfile.gettempdir()) ) / "khanan_srtm_cache"
SRTM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_RESOLUTION_METERS = 10.0  # Target DEM resolution after resampling
RIM_MIN_WIDTH_PIXELS = 8
RIM_DILATION_ITERATIONS = 2
MAX_TILE_DOWNLOAD_SECONDS = 60

try:
    MAX_VISUALIZATION_DIMENSION = max(16, min(160, int(os.getenv("QUANT_ANALYSIS_MAX_GRID", "96"))))
except ValueError:  # noqa: PERF203 - tiny scope
    MAX_VISUALIZATION_DIMENSION = 96

MIN_VISUALIZATION_POINTS = 12


def _read_float_env(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):  # noqa: PERF203 - simple defensive guard
        return float(default)


VOLUME_PRIORITY_THRESHOLD = _read_float_env("QUANT_ANALYSIS_VOLUME_THRESHOLD", 100_000.0)
DEPTH_PRIORITY_THRESHOLD = _read_float_env("QUANT_ANALYSIS_DEPTH_THRESHOLD", 15.0)


class QuantitativeAnalysisRequest(BaseModel):
    results: Dict[str, Any] = Field(..., description="Full analysis results payload including tiles and merged blocks")

    model_config = {
        "extra": "ignore"
    }


@dataclass
class StepLog:
    name: str
    status: str
    duration_ms: int
    details: List[str]


class StepLogger:
    """Collects sequential step logs with execution time."""

    def __init__(self) -> None:
        self._steps: List[StepLog] = []

    @contextmanager
    def step(self, name: str) -> Iterable[List[str]]:
        details: List[str] = []
        start = time.perf_counter()
        status = "completed"
        try:
            yield details
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            details.append(f"Error: {exc}")
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            self._steps.append(StepLog(name=name, status=status, duration_ms=duration_ms, details=details))

    @property
    def steps(self) -> List[Dict[str, Any]]:
        return [step.__dict__ for step in self._steps]


class QuantitativeProcessingError(Exception):
    """Raised when quantitative processing fails in a recoverable way."""


def _tile_name(lat: int, lon: int) -> str:
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"
    return f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}"


def _ensure_srtm_tile(lat: int, lon: int, temp_dir: Path, step_details: List[str]) -> Path:
    tile_name = _tile_name(lat, lon)
    cache_path = SRTM_CACHE_DIR / f"{tile_name}.tif"
    if cache_path.exists():
        step_details.append(f"Cache hit for tile {tile_name}")
        return cache_path

    hgt_cache_path = SRTM_CACHE_DIR / f"{tile_name}.hgt"
    if not hgt_cache_path.exists():
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{tile_name[:3]}/{tile_name}.hgt.gz"
        step_details.append(f"Downloading {tile_name} from {url}")
        response = requests.get(url, timeout=MAX_TILE_DOWNLOAD_SECONDS)
        if response.status_code != 200:
            raise QuantitativeProcessingError(f"Failed to download DEM tile {tile_name}: HTTP {response.status_code}")
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
            raw_data = gz.read()
        hgt_cache_path.write_bytes(raw_data)
    else:
        step_details.append(f"Using cached HGT for tile {tile_name}")

    # Convert HGT to GeoTIFF for faster subsequent reads
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(hgt_cache_path) as src:
        profile = src.profile.copy()
        data = src.read(1)
        profile.update(driver="GTiff", compress="LZW")
        with rasterio.open(cache_path, "w", **profile) as dst:
            dst.write(data, 1)

    step_details.append(f"Converted {tile_name} to GeoTIFF cache")
    return cache_path


def _extract_block_features(results: Dict[str, Any], step_logger: StepLogger) -> Tuple[List[Dict[str, Any]], str]:
    with step_logger.step("Extract block geometries") as details:
        features: List[Dict[str, Any]] = []
        source = "merged_blocks"

        merged = results.get("merged_blocks") or results.get("mergedBlocks")
        if isinstance(merged, dict):
            for feature in merged.get("features", []) or []:
                geometry = feature.get("geometry")
                props = feature.get("properties") or {}
                if not geometry:
                    continue
                shapely_geom = shape(geometry)
                if shapely_geom.is_empty:
                    continue
                features.append({
                    "geometry": shapely_geom,
                    "properties": props,
                    "label": props.get("name") or props.get("block_id") or props.get("id") or "Merged Block",
                    "persistent_id": props.get("persistent_id") or props.get("persistentId"),
                    "source": "merged"
                })

        if not features:
            source = "tiles"
            tiles = results.get("tiles") or []
            for tile in tiles:
                tile_label = tile.get("tile_label") or tile.get("tile_id") or tile.get("tileId")
                blocks = tile.get("mine_blocks") or []
                for block in blocks:
                    geometry = block.get("geometry") or {}
                    props = block.get("properties") or {}
                    shapely_geom = shape(geometry) if geometry else None
                    if not shapely_geom or shapely_geom.is_empty:
                        continue
                    features.append({
                        "geometry": shapely_geom,
                        "properties": props,
                        "label": props.get("name") or f"{tile_label or 'Tile'} Block",
                        "persistent_id": props.get("persistent_id") or props.get("persistentId"),
                        "source": "tile"
                    })

        details.append(f"Collected {len(features)} block geometries from {source}")
        if not features:
            raise QuantitativeProcessingError("No mine block geometries available for quantitative analysis")
        return features, source


def _union_bounds(features: List[Dict[str, Any]]) -> Tuple[float, float, float, float, MultiPolygon]:
    geometries = [feature["geometry"] for feature in features]
    combined: MultiPolygon = unary_union(geometries)
    bounds = combined.bounds
    return bounds[0], bounds[1], bounds[2], bounds[3], combined


def _compute_utm_crs(union_geometry: MultiPolygon) -> pyproj.CRS:
    centroid: Point = union_geometry.centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return pyproj.CRS.from_epsg(epsg)


@dataclass
class DemData:
    array: np.ndarray
    transform: rasterio.Affine
    crs: pyproj.CRS
    resolution: float
    tile_paths: List[Path]
    bounds_utm: Tuple[float, float, float, float]
    bounds_wgs84: Tuple[float, float, float, float]


def _build_dem(bounds: Tuple[float, float, float, float], target_crs: pyproj.CRS, step_logger: StepLogger) -> DemData:
    minx, miny, maxx, maxy = bounds
    buffer_deg = 0.02
    lat_min = math.floor(miny - buffer_deg)
    lat_max = math.ceil(maxy + buffer_deg)
    lon_min = math.floor(minx - buffer_deg)
    lon_max = math.ceil(maxx + buffer_deg)

    with step_logger.step("Acquire SRTM tiles") as details:
        temp_dir = Path(tempfile.mkdtemp(prefix="quant_dem_"))
        tile_paths: List[Path] = []
        try:
            for lat in range(lat_min, lat_max):
                for lon in range(lon_min, lon_max):
                    path = _ensure_srtm_tile(lat, lon, temp_dir, details)
                    tile_paths.append(path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if not tile_paths:
            raise QuantitativeProcessingError("No DEM tiles available for requested area")
        details.append(f"Prepared {len(tile_paths)} DEM tile(s)")

    with step_logger.step("Merge & resample DEM") as details:
        gdalbuildvrt = shutil.which("gdalbuildvrt")
        gdalwarp = shutil.which("gdalwarp")

        transformer_to_target = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(4326), target_crs, always_xy=True
        )
        transformer_to_wgs84 = pyproj.Transformer.from_crs(
            target_crs, pyproj.CRS.from_epsg(4326), always_xy=True
        )

        minx_t, miny_t = transformer_to_target.transform(minx, miny)
        maxx_t, maxy_t = transformer_to_target.transform(maxx, maxy)
        te_minx, te_maxx = sorted([minx_t, maxx_t])
        te_miny, te_maxy = sorted([miny_t, maxy_t])

        if gdalbuildvrt and gdalwarp:
            temp_root = Path(tempfile.mkdtemp(prefix="quant_gdal_"))
            vrt_path = temp_root / "mosaic.vrt"
            warped_path = temp_root / "warped.tif"
            try:
                build_command = [
                    gdalbuildvrt,
                    "-q",
                    str(vrt_path),
                    *[str(path) for path in tile_paths],
                ]
                subprocess.run(build_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                warp_command = [
                    gdalwarp,
                    "-q",
                    "-multi",
                    "-r",
                    "bilinear",
                    "-t_srs",
                    target_crs.to_wkt(),
                    "-tr",
                    str(PIXEL_RESOLUTION_METERS),
                    str(PIXEL_RESOLUTION_METERS),
                    "-te",
                    str(te_minx),
                    str(te_miny),
                    str(te_maxx),
                    str(te_maxy),
                    "-dstnodata",
                    "-9999",
                    "-overwrite",
                    str(vrt_path),
                    str(warped_path),
                ]
                subprocess.run(warp_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                with rasterio.open(warped_path) as src:
                    destination = src.read(1).astype(np.float32)
                    destination[src.read_masks(1) == 0] = np.nan
                    dst_transform = src.transform

                bounds_utm = array_bounds(destination.shape[0], destination.shape[1], dst_transform)
                minx_utm, miny_utm, maxx_utm, maxy_utm = bounds_utm
                ll_lon, ll_lat = transformer_to_wgs84.transform(minx_utm, miny_utm)
                ur_lon, ur_lat = transformer_to_wgs84.transform(maxx_utm, maxy_utm)

                details.append(
                    "GDAL warp completed with precise alignment and resolution control"
                )

                return DemData(
                    array=destination,
                    transform=dst_transform,
                    crs=target_crs,
                    resolution=PIXEL_RESOLUTION_METERS,
                    tile_paths=tile_paths,
                    bounds_utm=(minx_utm, miny_utm, maxx_utm, maxy_utm),
                    bounds_wgs84=(ll_lon, ll_lat, ur_lon, ur_lat),
                )
            finally:
                shutil.rmtree(temp_root, ignore_errors=True)

        details.append("GDAL binaries not found, falling back to Rasterio reprojection")
        datasets = [rasterio.open(path) for path in tile_paths]
        try:
            mosaic, mosaic_transform = merge(datasets)
            mosaic_array = mosaic[0].astype(np.float32, copy=False)
            src_crs = datasets[0].crs or "EPSG:4326"
            src_height, src_width = mosaic_array.shape
            src_bounds = array_bounds(src_height, src_width, mosaic_transform)

            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs,
                target_crs,
                src_width,
                src_height,
                *src_bounds,
                resolution=PIXEL_RESOLUTION_METERS,
            )

            destination = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
            reproject(
                source=mosaic_array,
                destination=destination,
                src_transform=mosaic_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                src_nodata=-32768,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

            bounds_utm = array_bounds(destination.shape[0], destination.shape[1], dst_transform)
            minx_utm, miny_utm, maxx_utm, maxy_utm = bounds_utm
            ll_lon, ll_lat = transformer_to_wgs84.transform(minx_utm, miny_utm)
            ur_lon, ur_lat = transformer_to_wgs84.transform(maxx_utm, maxy_utm)
            details.append(
                f"Resampled DEM to {destination.shape[1]}x{destination.shape[0]} grid @ {PIXEL_RESOLUTION_METERS:.0f}m"
            )

            return DemData(
                array=destination,
                transform=dst_transform,
                crs=target_crs,
                resolution=PIXEL_RESOLUTION_METERS,
                tile_paths=tile_paths,
                bounds_utm=(minx_utm, miny_utm, maxx_utm, maxy_utm),
                bounds_wgs84=(ll_lon, ll_lat, ur_lon, ur_lat),
            )
        finally:
            for ds in datasets:
                ds.close()


def _geometry_to_utm_transformer(target_crs: pyproj.CRS) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), target_crs, always_xy=True)


def _robust_rim_elevation(dem: np.ndarray, block_mask: np.ndarray, iterations: int = RIM_DILATION_ITERATIONS) -> Optional[float]:
    structure = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(block_mask, structure=structure, iterations=iterations)
    rim_mask = np.logical_and(dilated, ~block_mask)
    rim_values = dem[rim_mask]
    rim_values = rim_values[np.isfinite(rim_values)]

    if rim_values.size == 0:
        return None

    median = float(np.median(rim_values))
    mad = float(np.median(np.abs(rim_values - median)))
    if mad > 0:
        threshold = 3.0 * 1.4826 * mad
        inliers = rim_values[np.abs(rim_values - median) <= threshold]
        if inliers.size > 0:
            return float(np.median(inliers))
    return median


def _simpson_volume(depth_surface: np.ndarray, pixel_size: float) -> float:
    try:
        first_pass = integrate.simpson(depth_surface, dx=pixel_size, axis=0)
        volume = integrate.simpson(first_pass, dx=pixel_size)
        return float(max(0.0, volume))
    except (ValueError, RuntimeError):
        # Fallback to summation
        return float(depth_surface.sum() * (pixel_size ** 2))


def _array_to_serializable(matrix: np.ndarray) -> List[List[Optional[float]]]:
    if matrix.size == 0:
        return []
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in matrix
    ]


def _prepare_visualization_payload(
    block_mask: np.ndarray,
    dem_array: np.ndarray,
    depth_surface: np.ndarray,
    transform: rasterio.Affine,
    rim_elevation: float,
    pixel_resolution: float,
) -> Optional[Dict[str, Any]]:
    rows, cols = np.where(block_mask)
    if rows.size == 0 or cols.size == 0:
        return None

    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    subset_elev = dem_array[row_min:row_max + 1, col_min:col_max + 1]
    subset_depth = depth_surface[row_min:row_max + 1, col_min:col_max + 1]
    subset_mask = block_mask[row_min:row_max + 1, col_min:col_max + 1]

    row_count, col_count = subset_elev.shape
    if row_count == 0 or col_count == 0:
        return None

    row_step = max(1, math.ceil(row_count / MAX_VISUALIZATION_DIMENSION))
    col_step = max(1, math.ceil(col_count / MAX_VISUALIZATION_DIMENSION))

    subset_elev = subset_elev[::row_step, ::col_step]
    subset_depth = subset_depth[::row_step, ::col_step]
    subset_mask = subset_mask[::row_step, ::col_step]

    if subset_mask.sum() < MIN_VISUALIZATION_POINTS:
        return None

    subset_elev = np.where(subset_mask, subset_elev, np.nan)
    subset_depth = np.where(subset_mask, subset_depth, np.nan)

    row_indices = np.arange(row_min, row_max + 1, row_step)
    col_indices = np.arange(col_min, col_max + 1, col_step)

    sample_row = int(row_indices[0]) if row_indices.size else row_min
    sample_col = int(col_indices[0]) if col_indices.size else col_min

    x_coords = [
        float(xy(transform, sample_row, int(col_idx), offset="center")[0])
        for col_idx in col_indices
    ]
    y_coords = [
        float(xy(transform, int(row_idx), sample_col, offset="center")[1])
        for row_idx in row_indices
    ]

    elevation_values = subset_elev[np.isfinite(subset_elev)]
    depth_values = subset_depth[np.isfinite(subset_depth)]

    stats: Dict[str, Optional[float]] = {
        "minElevation": float(np.nanmin(elevation_values)) if elevation_values.size else None,
        "maxElevation": float(np.nanmax(elevation_values)) if elevation_values.size else None,
        "minDepth": float(np.nanmin(depth_values)) if depth_values.size else None,
        "maxDepth": float(np.nanmax(depth_values)) if depth_values.size else None,
    }

    metadata = {
        "rowCount": int(subset_elev.shape[0]),
        "columnCount": int(subset_elev.shape[1]),
        "rowStep": int(row_step),
        "columnStep": int(col_step),
        "pixelResolutionMeters": float(pixel_resolution),
        "sampledPixelCount": int(subset_mask.sum()),
    }

    extent = {
        "minX": float(min(x_coords)) if x_coords else None,
        "maxX": float(max(x_coords)) if x_coords else None,
        "minY": float(min(y_coords)) if y_coords else None,
        "maxY": float(max(y_coords)) if y_coords else None,
    }

    return {
        "grid": {
            "x": x_coords,
            "y": y_coords,
            "elevation": _array_to_serializable(subset_elev),
            "depth": _array_to_serializable(subset_depth),
            "rimElevation": float(rim_elevation),
            "resolutionX": float(pixel_resolution * col_step),
            "resolutionY": float(pixel_resolution * row_step),
            "unit": "meters",
        },
        "stats": stats,
        "extentUTM": extent,
        "metadata": metadata,
    }


def _build_executive_summary(block_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not block_metrics:
        return {
            "headline": {
                "totalVolumeCubicMeters": 0.0,
                "totalAreaHectares": 0.0,
                "blockCount": 0,
            },
            "priorityBlocks": [],
            "insights": {
                "averageMeanDepthMeters": 0.0,
                "averageMaxDepthMeters": 0.0,
            },
            "updatedAt": datetime.utcnow().isoformat(),
        }

    total_volume = sum(block["volumeCubicMeters"] for block in block_metrics)
    total_area_hectares = sum(block["areaHectares"] for block in block_metrics)
    block_count = len(block_metrics)
    average_mean_depth = (
        sum(block["meanDepthMeters"] for block in block_metrics) / block_count
        if block_count else 0.0
    )
    average_max_depth = (
        sum(block["maxDepthMeters"] for block in block_metrics) / block_count
        if block_count else 0.0
    )

    priority_candidates = [
        block
        for block in block_metrics
        if block["volumeCubicMeters"] >= VOLUME_PRIORITY_THRESHOLD
        or block["maxDepthMeters"] >= DEPTH_PRIORITY_THRESHOLD
    ]
    priority_sorted = sorted(
        priority_candidates,
        key=lambda item: (item["volumeCubicMeters"], item["maxDepthMeters"]),
        reverse=True,
    )[:5]

    priority_blocks = [
        {
            "label": block["blockLabel"],
            "blockId": block["blockId"],
            "source": block.get("source"),
            "volumeCubicMeters": block["volumeCubicMeters"],
            "maxDepthMeters": block["maxDepthMeters"],
            "areaHectares": block["areaHectares"],
        }
        for block in priority_sorted
    ]

    deepest_block = max(block_metrics, key=lambda item: item["maxDepthMeters"])
    largest_volume_block = max(block_metrics, key=lambda item: item["volumeCubicMeters"])

    return {
        "headline": {
            "totalVolumeCubicMeters": total_volume,
            "totalAreaHectares": total_area_hectares,
            "blockCount": block_count,
        },
        "priorityBlocks": priority_blocks,
        "insights": {
            "averageMeanDepthMeters": average_mean_depth,
            "averageMaxDepthMeters": average_max_depth,
            "deepestBlock": {
                "label": deepest_block["blockLabel"],
                "maxDepthMeters": deepest_block["maxDepthMeters"],
            },
            "largestBlock": {
                "label": largest_volume_block["blockLabel"],
                "volumeCubicMeters": largest_volume_block["volumeCubicMeters"],
            },
        },
        "policyFlags": {
            "requiresAttention": bool(priority_blocks),
            "highVolumeThreshold": VOLUME_PRIORITY_THRESHOLD,
            "highDepthThreshold": DEPTH_PRIORITY_THRESHOLD,
        },
        "updatedAt": datetime.utcnow().isoformat(),
    }


def _generate_block_metrics(
    features: List[Dict[str, Any]],
    dem: DemData,
    transformer: pyproj.Transformer,
    step_logger: StepLogger,
) -> List[Dict[str, Any]]:
    with step_logger.step("Rasterize mine blocks") as details:
        shapes = []
        transformed_geometries: List[Any] = []
        for idx, feature in enumerate(features, start=1):
            geom_wgs84 = feature["geometry"]
            geom_utm = shapely_transform(transformer.transform, geom_wgs84)
            transformed_geometries.append(geom_utm)
            shapes.append((geom_utm, idx))

        label_raster = rasterize(
            shapes,
            out_shape=dem.array.shape,
            transform=dem.transform,
            fill=0,
            dtype="int32",
            all_touched=True,
        )
        details.append(
            f"Rasterized {len(features)} block(s) to DEM grid with shape {dem.array.shape[1]}x{dem.array.shape[0]}"
        )

    block_metrics: List[Dict[str, Any]] = []
    with step_logger.step("Compute volumetric metrics") as details:
        pixel_area = dem.resolution ** 2
        for idx, feature in enumerate(features, start=1):
            block_mask = label_raster == idx
            coverage_pixels = int(block_mask.sum())
            if coverage_pixels == 0:
                details.append(f"Block {feature['label']} skipped (no DEM coverage)")
                continue

            block_dem = np.where(block_mask, dem.array, np.nan)
            elevation_values = block_dem[np.isfinite(block_dem)]
            if elevation_values.size == 0:
                details.append(f"Block {feature['label']} skipped (DEM nodata)")
                continue

            rim_elevation = _robust_rim_elevation(dem.array, block_mask)
            if rim_elevation is None:
                rim_elevation = float(np.nanmax(elevation_values))

            depth_surface = rim_elevation - dem.array
            depth_surface[depth_surface < 0] = 0
            depth_surface = np.where(block_mask, depth_surface, 0)
            depth_surface = np.nan_to_num(depth_surface, nan=0.0)

            volume_simpson = _simpson_volume(depth_surface, dem.resolution)
            volume_sum = float(depth_surface.sum() * pixel_area)
            area_sq_m = transformed_geometries[idx - 1].area
            max_depth = float(np.nanmax(depth_surface))
            mean_depth = float(volume_simpson / area_sq_m) if area_sq_m > 0 else 0.0
            median_depth = float(np.nanmedian(depth_surface[block_mask]))

            geom_wgs84 = feature["geometry"]
            centroid = geom_wgs84.centroid

            visualization_payload = _prepare_visualization_payload(
                block_mask,
                dem.array,
                depth_surface,
                dem.transform,
                rim_elevation,
                dem.resolution,
            )

            block_metrics.append({
                "blockLabel": feature["label"],
                "blockId": feature["properties"].get("block_id") or feature.get("persistent_id") or f"block-{idx}",
                "source": feature.get("source", "unknown"),
                "persistentId": feature.get("persistent_id"),
                "areaSquareMeters": float(area_sq_m),
                "areaHectares": float(area_sq_m / 10_000.0),
                "pixelCount": coverage_pixels,
                "rimElevationMeters": float(rim_elevation),
                "maxDepthMeters": max_depth,
                "meanDepthMeters": mean_depth,
                "medianDepthMeters": median_depth,
                "volumeCubicMeters": volume_simpson,
                "volumeTrapezoidalCubicMeters": volume_sum,
                "centroid": {
                    "lon": float(centroid.x),
                    "lat": float(centroid.y),
                },
                "visualization": visualization_payload,
                "computedAt": datetime.utcnow().isoformat(),
            })
            details.append(
                f"Processed {feature['label']}: area={area_sq_m:.1f} m², volume={volume_simpson:.1f} m³, max depth={max_depth:.2f} m"
            )

    return block_metrics


def _aggregate_summary(block_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not block_metrics:
        return {
            "totalVolumeCubicMeters": 0.0,
            "totalAreaSquareMeters": 0.0,
            "totalAreaHectares": 0.0,
            "averageMaxDepthMeters": 0.0,
            "averageMeanDepthMeters": 0.0,
            "blockCount": 0,
            "deepestBlock": None,
            "largestBlock": None,
        }

    total_volume = sum(block["volumeCubicMeters"] for block in block_metrics)
    total_area = sum(block["areaSquareMeters"] for block in block_metrics)
    average_max_depth = (
        sum(block["maxDepthMeters"] for block in block_metrics) / len(block_metrics)
        if block_metrics else 0.0
    )
    average_mean_depth = (
        sum(block["meanDepthMeters"] for block in block_metrics) / len(block_metrics)
        if block_metrics else 0.0
    )
    deepest_block = max(block_metrics, key=lambda item: item["maxDepthMeters"])
    largest_block = max(block_metrics, key=lambda item: item["volumeCubicMeters"])

    return {
        "totalVolumeCubicMeters": total_volume,
        "totalAreaSquareMeters": total_area,
        "totalAreaHectares": total_area / 10_000.0,
        "averageMaxDepthMeters": average_max_depth,
        "averageMeanDepthMeters": average_mean_depth,
        "blockCount": len(block_metrics),
        "deepestBlock": {
            "label": deepest_block["blockLabel"],
            "maxDepthMeters": deepest_block["maxDepthMeters"],
            "volumeCubicMeters": deepest_block["volumeCubicMeters"],
        },
        "largestBlock": {
            "label": largest_block["blockLabel"],
            "volumeCubicMeters": largest_block["volumeCubicMeters"],
            "areaHectares": largest_block["areaHectares"],
        },
    }


def _run_quantitative_sync(analysis_id: str, payload: QuantitativeAnalysisRequest) -> Dict[str, Any]:
    step_logger = StepLogger()

    results = payload.results
    if not isinstance(results, dict):
        raise QuantitativeProcessingError("Invalid results payload supplied")

    features, source = _extract_block_features(results, step_logger)
    minx, miny, maxx, maxy, union_geom = _union_bounds(features)
    with step_logger.step("Prepare analysis extent") as details:
        details.append(
            f"Bounding box (WGS84): {minx:.4f}, {miny:.4f} → {maxx:.4f}, {maxy:.4f}"
        )

    target_crs = _compute_utm_crs(union_geom)
    dem = _build_dem((minx, miny, maxx, maxy), target_crs, step_logger)
    transformer = _geometry_to_utm_transformer(target_crs)
    block_metrics = _generate_block_metrics(features, dem, transformer, step_logger)
    summary = _aggregate_summary(block_metrics)
    executive_summary = _build_executive_summary(block_metrics)
    generated_at = datetime.utcnow().isoformat()

    visualization_ready = any(block.get("visualization") for block in block_metrics)

    return {
        "analysisId": analysis_id,
        "status": "completed",
        "blockCount": len(block_metrics),
        "steps": step_logger.steps,
        "summary": summary,
        "executiveSummary": executive_summary,
        "blocks": block_metrics,
        "dem": {
            "crs": dem.crs.to_string(),
            "resolutionMeters": dem.resolution,
            "tileCount": len(dem.tile_paths),
            "boundsUTM": dem.bounds_utm,
            "boundsWGS84": dem.bounds_wgs84,
        },
        "source": {
            "blockCollection": source,
        },
        "metadata": {
            "generatedAt": generated_at,
            "visualizationAvailable": visualization_ready,
            "pixelResolutionMeters": dem.resolution,
        },
    }


@router.post("/{analysis_id}/quantitative")
async def run_quantitative_analysis(analysis_id: str, payload: QuantitativeAnalysisRequest):
    """Execute quantitative volumetric analysis for a detection result."""

    try:
        result = await asyncio.to_thread(_run_quantitative_sync, analysis_id, payload)
        return result
    except QuantitativeProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
