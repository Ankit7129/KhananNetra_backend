"""
merge_mine_blocks.py

Reliable merging of adjacent mine-block polygons across tiled outputs.
- Bridges tiny gaps at tile seams
- Detects and merges edge-to-edge contact (long straight seams)
- Removes the seam by buffering/dissolving and polygonizing when needed
- Returns a GeoJSON FeatureCollection with sequential block IDs and label positions

Usage: call merge_adjacent_mine_blocks(all_tiles_data)
each tile entry in all_tiles_data should contain 'mine_blocks' (GeoJSON FeatureCollection or list of features)
and optional 'tile_id' / 'index' metadata.
"""
import sys
import math
from typing import List, Dict, Any, Optional, Tuple

from shapely.geometry import shape, mapping, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, polygonize
from shapely.validation import make_valid


# Tunable constants (adjust to your map units / resolution)
BUFFER_EPS = 1e-5          # small buffer to bridge tiny seams (~1e-5 deg ~ ~1m at equator for lon/lat)
TOLERANT_BOUNDARY_TOL = 5e-6
MIN_AREA_FILTER = 1e-12    # filter tiny slivers
EDGE_SHARE_AREA_RATIO = 0.05
BOUNDARY_SHARED_RATIO_FOR_MERGE = 0.10  # More aggressive: reduced from 0.15 to 0.10
ABS_SHARED_LENGTH_MIN = 3e-6  # More aggressive: reduced from 5e-6 to 3e-6
MERGE_PROXIMITY_RATIO = 0.08  # More aggressive: reduced from 0.15 to 0.08


def merge_adjacent_mine_blocks(all_tiles_data: List[Dict], buffer_eps: float = BUFFER_EPS, min_confidence: float = 0.92, analysis_id: str = None) -> Dict:
    """
    Merge adjacent mine-block polygons from multiple tiles into unified mine blocks.

    Args:
        all_tiles_data: list of tiles; each tile dict should have 'mine_blocks' (GeoJSON FeatureCollection or list)
        buffer_eps: tiny buffer used during merging/dissolve (units same as your coords)
        analysis_id: unique analysis identifier to include in block IDs

    Returns:
        GeoJSON FeatureCollection dict with each feature having:
          - properties.block_id (unique ID: analysis_id-M1, analysis_id-T1B1, etc.)
          - properties.name
          - properties.area_m2 (sum of contributing original areas if present)
          - properties.is_merged (bool)
          - properties.source_blocks (list of contributing names if merged)
          - properties.label_position (x,y inside polygon)
    """
    all_polygons = []
    all_polygons_original = []  # CRITICAL: Keep unbuffered originals for touch detection
    all_metadata = []
    skipped_low_confidence = 0

    # Generate a short prefix from analysis_id for block IDs
    id_prefix = ""
    if analysis_id:
        # Use first 8 chars of analysis_id for brevity
        id_prefix = analysis_id[:8] + "-"

    # Track blocks per tile for proper hierarchical numbering
    tile_block_counters = {}

    # collect polygons
    for tile_idx, tile in enumerate(all_tiles_data):
        mine_blocks = tile.get("mine_blocks")
        if not mine_blocks:
            continue

        # support either FeatureCollection dict or list of feature dicts
        if isinstance(mine_blocks, dict) and "features" in mine_blocks:
            features = mine_blocks["features"]
        elif isinstance(mine_blocks, list):
            features = mine_blocks
        else:
            continue

        # Extract and normalize tile_id consistently
        raw_tile_id = tile.get("tile_id") or tile.get("id")
        
        # DEBUG: Log the tile structure to understand the issue
        sys.stderr.write(f"[DEBUG] Tile {tile_idx}: keys={list(tile.keys())}, tile_id={raw_tile_id}\n")
        
        if raw_tile_id and str(raw_tile_id).strip():
            tile_id_clean = str(raw_tile_id).strip()
            # Remove 'tile_' prefix if present to normalize (tile_1 -> 1, tile_3 -> 3)
            if tile_id_clean.lower().startswith('tile_'):
                tile_id_clean = tile_id_clean[5:]  # Remove 'tile_' prefix
        else:
            # Use 1-based indexing for missing IDs
            tile_id_clean = str(tile_idx + 1)
            sys.stderr.write(f"[DEBUG] No tile_id found, using fallback: {tile_id_clean}\n")
        
        # Initialize block counter for this tile
        if tile_id_clean not in tile_block_counters:
            tile_block_counters[tile_id_clean] = 0

        for feat in features:
            try:
                geom = shape(feat["geometry"])
                original_area = geom.area
                sys.stderr.write(f"[Polygon] Processing polygon from tile {tile_id_clean}, area: {original_area:.8f}\n")
                
                # ensure valid geometry
                if not geom.is_valid:
                    sys.stderr.write(f"[Polygon] Invalid geometry detected, making valid...\n")
                    geom = make_valid(geom)
                    if geom.area != original_area:
                        sys.stderr.write(f"[Polygon] Area changed after make_valid: {original_area:.8f} -> {geom.area:.8f}\n")

                # Validate geometry type and boundary
                if not isinstance(geom, Polygon):
                    sys.stderr.write(f"[Merge] Warning: Skipping non-Polygon geometry type: {type(geom).__name__}\n")
                    continue
                    
                if not hasattr(geom, 'boundary') or geom.boundary is None:
                    sys.stderr.write(f"[Merge] Warning: Skipping polygon with None boundary\n")
                    continue

                # keep original unbuffered for area/confidence attribution
                original_geom = geom

                # work on a slightly buffered copy to help merge pixel gaps
                # use a very small positive buffer; if geom is multipolygon it's OK
                buffered = original_geom.buffer(buffer_eps)
                if not buffered.is_valid:
                    buffered = make_valid(buffered)

                # ensure we only keep polygons
                if buffered.is_empty:
                    continue

                # Increment block counter for this tile
                tile_block_counters[tile_id_clean] += 1
                block_num = tile_block_counters[tile_id_clean]

                all_polygons.append(buffered)
                all_polygons_original.append(original_geom)  # CRITICAL: Store unbuffered for touch checks
                props = feat.get("properties", {}) or {}

                # Normalize confidence value (support 0-1 or 0-100 inputs)
                raw_conf = props.get("avg_confidence") if props.get("avg_confidence") is not None else props.get("confidence")
                try:
                    conf_val = float(raw_conf) if raw_conf is not None else 0.0
                except Exception:
                    conf_val = 0.0
                if conf_val > 1.5:  # likely in percent (e.g. 90.0)
                    conf_val = conf_val / 100.0

                # Skip low-confidence detections completely
                if conf_val < float(min_confidence):
                    skipped_low_confidence += 1
                    sys.stderr.write(f"[MergeBlocks] Skipping low-confidence block (conf={conf_val:.3f}) from tile {tile_id_clean}\n")
                    continue
                
                # Create consistent hierarchical naming: T3B1 = Tile 3, Block 1
                original_block_name = f"T{tile_id_clean}B{block_num}"
                # Unique block ID with analysis prefix
                unique_block_id = f"{id_prefix}T{tile_id_clean}B{block_num}"
                
                sys.stderr.write(f"[MergeBlocks] Block '{original_block_name}' (ID: {unique_block_id}) from tile {tile_id_clean}\n")
                
                all_metadata.append({
                    "unique_block_id": unique_block_id,  # Full unique ID with analysis prefix
                    "original_block_id": props.get("block_id"),
                    "original_name": original_block_name,  # Hierarchical name: T3B1, T3B2, etc.
                    "area_m2": props.get("area_m2", 0),
                    "avg_confidence": conf_val,
                    "tile_index": tile_idx + 1,  # 1-based tile index
                    "tile_id": tile_id_clean,    # Clean consistent ID (1, 2, 3, not tile_1)
                    "block_number": block_num,   # Block number within tile (1, 2, 3...)
                    "original_geom": original_geom
                })
            except Exception as e:
                sys.stderr.write(f"[MergeBlocks] Warning reading feature: {e}\n")
                continue

    if not all_polygons:
        return {"type": "FeatureCollection", "features": [], "metadata": {"merged_block_count": 0, "original_block_count": 0}}

    sys.stderr.write(f"[MergeBlocks] Collected {len(all_polygons)} polygons from {len(all_tiles_data)} tiles\n")
    sys.stderr.flush()

    # Phase 1: group polygons that should be merged
    # CRITICAL FIX: Pass both buffered and original geometries for proper touch detection
    groups = _group_polygons_for_merge(all_polygons, all_polygons_original, overlap_threshold=BOUNDARY_SHARED_RATIO_FOR_MERGE)

    sys.stderr.write(f"[MergeBlocks] Formed {len(groups)} candidate groups\n")

    # Phase 2: merge each group robustly
    merged_polygons = []
    for gi, group in enumerate(groups, start=1):
        try:
            if len(group) == 1:
                merged_polygons.append(group[0])
            else:
                merged = _merge_polygon_group(group, base_eps=buffer_eps)
                if merged is None or merged.is_empty:
                    # fallback to unary_union
                    merged = unary_union(group)
                # convert multi -> list of polygons (we'll handle later)
                if isinstance(merged, (MultiPolygon, GeometryCollection)):
                    for g in getattr(merged, "geoms", [merged]):
                        if getattr(g, "area", 0) > MIN_AREA_FILTER:
                            merged_polygons.append(g)
                else:
                    if getattr(merged, "area", 0) > MIN_AREA_FILTER:
                        merged_polygons.append(merged)
        except Exception as e:
            sys.stderr.write(f"[MergeBlocks] Warning merging group {gi}: {e}\n")
            # best-effort fallback: add original group members
            merged_polygons.extend([p for p in group if getattr(p, "area", 0) > MIN_AREA_FILTER])

    sys.stderr.write(f"[MergeBlocks] After merging groups: {len(merged_polygons)} polygons\n")

    # Phase 3: create features and attribution (which original blocks contributed)
    # sort largest to smallest for stable numbering
    merged_polygons_sorted = sorted(merged_polygons, key=lambda p: p.area if p is not None else 0, reverse=True)

    features = []
    total_area_m2 = 0.0
    total_confidence = 0.0

    for idx, poly in enumerate(merged_polygons_sorted, start=1):
        try:
            contributing_blocks = []  # Store detailed block info
            sum_area = 0.0
            sum_conf = 0.0
            tiles_set = set()
            contributing_details = []  # For detailed merge info

            for i, orig_buf in enumerate(all_polygons):
                orig_unbuffered = all_metadata[i]["original_geom"]
                # intersection test with unbuffered original geom (safer for attribution)
                if poly.intersects(orig_unbuffered):
                    md = all_metadata[i]
                    original_name = md.get("original_name")
                    contributing_blocks.append(original_name)
                    sum_area += md.get("area_m2", 0) or 0
                    sum_conf += md.get("avg_confidence", 0) or 0
                    
                    # IMPROVED: Better tile ID handling with detailed tracking
                    tid = md.get("tile_id")
                    tile_index = md.get("tile_index", "?")
                    block_area_ha = (md.get("area_m2", 0) or 0) / 10000
                    block_conf = (md.get("avg_confidence", 0) or 0) * 100
                    
                    if tid:
                        tid_str = str(tid).strip()
                        # Only add valid tile IDs (not empty, not 'tile_0')
                        if tid_str and tid_str.lower() not in ['tile_0', 'none', 'null', '']:
                            tiles_set.add(tid_str)
                            # Store detailed contributing info
                            block_num = md.get("block_number", "?")
                            contributing_details.append({
                                "tile_id": tid_str,
                                "tile_index": tile_index,
                                "block_name": original_name,
                                "block_number": block_num,
                                "area_ha": round(block_area_ha, 3),
                                "confidence": round(block_conf, 1)
                            })
                            sys.stderr.write(f"[MergeBlocks] Contributing: '{original_name}' (Tile {tid_str}, Block {block_num}): {block_area_ha:.3f}ha, {block_conf:.1f}%\n")

            avg_conf = (sum_conf / len(contributing_blocks)) if contributing_blocks else 0.0

            # choose safe label point inside polygon
            label_point = None
            try:
                rp = poly.representative_point()  # guaranteed to be inside
                label_point = [rp.x, rp.y]
            except Exception:
                try:
                    c = poly.centroid
                    label_point = [c.x, c.y]
                except Exception:
                    label_point = None

            # Determine if this is a merged block
            is_merged = len(contributing_blocks) > 1
            
            # Create unique block IDs with analysis prefix
            if is_merged:
                unique_block_id = f"{id_prefix}M{idx}"  # Merged blocks: analysis_id-M1, M2, etc.
                tile_list = sorted(list(tiles_set))
                if len(tile_list) == 1:
                    block_name = f"Merged Block M{idx} (from Tile {tile_list[0]})"
                else:
                    block_name = f"Merged Block M{idx} (from Tiles {', '.join(tile_list)})"
            else:
                # Single block - use original hierarchical ID
                if contributing_details and len(contributing_details) > 0:
                    original_detail = contributing_details[0]
                    tile_id = original_detail.get("tile_id", "?")
                    block_num = original_detail.get("block_name", "").split("B")[-1] if "B" in original_detail.get("block_name", "") else "?"
                    unique_block_id = f"{id_prefix}T{tile_id}B{block_num}"  # Individual blocks: analysis_id-T1B1, T1B2, etc.
                    block_name = f"Block T{tile_id}B{block_num}"
                elif contributing_blocks and len(contributing_blocks) > 0:
                    # Fallback to first contributing block name
                    block_name = contributing_blocks[0]
                    # Try to extract unique ID from metadata
                    for md in all_metadata:
                        if md.get("original_name") == block_name:
                            unique_block_id = md.get("unique_block_id", f"{id_prefix}{block_name}")
                            break
                    else:
                        unique_block_id = f"{id_prefix}{block_name}"
                else:
                    # Last resort fallback
                    tile_id = list(tiles_set)[0] if tiles_set else "?"
                    unique_block_id = f"{id_prefix}T{tile_id}B1"
                    block_name = f"Block T{tile_id}B1"

            props = {
                "block_id": unique_block_id,  # Unique ID with analysis prefix
                "name": block_name,
                "area_m2": round(sum_area, 2),
                "avg_confidence": round(avg_conf, 4),
                "label_position": label_point,
                "is_merged": is_merged,
                "source_blocks": contributing_blocks if is_merged else None,
                "spanning_tiles": sorted(list(tiles_set)) if tiles_set else None,
                "contributing_details": contributing_details if is_merged else None
            }
            
            # DEBUG: Log final block properties
            sys.stderr.write(f"[DEBUG] Final block {idx}: name='{block_name}', spanning_tiles={props['spanning_tiles']}\n")

            features.append({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": props
            })

            total_area_m2 += sum_area
            total_confidence += avg_conf
        except Exception as e:
            sys.stderr.write(f"[MergeBlocks] Warning creating feature {idx}: {e}\n")
            continue

    result = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "merged_block_count": len(features),
            "original_block_count": len(all_polygons),
            "total_area_m2": round(total_area_m2, 2),
            "avg_confidence": round((total_confidence / len(features)) if features else 0.0, 4),
            "tiles_processed": len(all_tiles_data)
        }
    }

    # Include skipped low-confidence count for visibility
    try:
        result["metadata"]["skipped_low_confidence"] = int(skipped_low_confidence)
    except Exception:
        result["metadata"]["skipped_low_confidence"] = 0

    sys.stderr.write(f"[MergeBlocks] ✅ Done: {len(features)} merged blocks from {len(all_polygons)} originals\n")
    sys.stderr.flush()
    return result


# -------------------- helper functions --------------------


def _group_polygons_for_merge(polys: List[Polygon], polys_original: List[Polygon], overlap_threshold: float = 0.15) -> List[List[Polygon]]:
    """
    Agglomerative grouping: start with first poly and add any poly
    that meets merge criteria (area overlap or shared boundary fraction).
    REDESIGNED: More aggressive grouping with detailed logging.
    
    CRITICAL: Uses polys_original (unbuffered) for touch detection to find tile seams,
    but returns groups of polys (buffered) for merging.
    """
    unprocessed_indices = list(range(len(polys)))  # Track indices
    groups = []
    
    sys.stderr.write(f"[Grouping] Starting with {len(polys)} polygons\n")

    while unprocessed_indices:
        base_idx = unprocessed_indices.pop(0)
        group_indices = [base_idx]
        changed = True
        iteration = 0
        
        while changed:
            iteration += 1
            changed = False
            remove_idxs = []
            
            for i, cand_idx in enumerate(unprocessed_indices):
                # Check if candidate should merge with any polygon in current group
                # CRITICAL: Use ORIGINAL unbuffered geometries for touch detection
                should_merge = False
                for group_idx in group_indices:
                    if _should_merge_polygons(polys_original[group_idx], polys_original[cand_idx], overlap_threshold=overlap_threshold):
                        should_merge = True
                        break
                
                if should_merge:
                    group_indices.append(cand_idx)
                    remove_idxs.append(i)
                    changed = True
                    sys.stderr.write(f"[Grouping] Iteration {iteration}: Added polygon {cand_idx} to group (group size now {len(group_indices)})\n")
                    
            for i in reversed(remove_idxs):
                unprocessed_indices.pop(i)
        
        # Convert indices back to actual polygons (buffered ones for merging)
        group_polys = [polys[idx] for idx in group_indices]
        groups.append(group_polys)
        sys.stderr.write(f"[Grouping] ✅ Group {len(groups)} complete: {len(group_polys)} polygons (indices: {group_indices})\n")

    sys.stderr.write(f"[Grouping] Final: {len(groups)} groups from {len(polys)} polygons\n")
    return groups


def _should_merge_polygons(poly1: Polygon, poly2: Polygon, overlap_threshold: float = 0.15) -> bool:
    """
    Decide whether two polygons should be merged.
    REDESIGNED: Ultra-aggressive for tile seam removal.
    
    Criteria (ANY of these triggers merge):
      1. Polygons touch at all (touches=True) → MERGE (tile seam!)
      2. Distance between polygons < threshold → MERGE (very close = same block)
      3. Any shared boundary detected → MERGE
      4. Area overlap > 1% → MERGE
    """
    try:
        if poly1.is_empty or poly2.is_empty:
            sys.stderr.write(f"[MergeCheck] ❌ SKIP: empty geometry\n")
            return False

        # Get bounds for debugging
        try:
            b1 = poly1.bounds  # (minx, miny, maxx, maxy)
            b2 = poly2.bounds
            sys.stderr.write(f"[MergeCheck] Poly1 bounds: {b1}, Poly2 bounds: {b2}\n")
        except Exception:
            pass

        # RULE 1: Check distance first (more reliable than touches for near-polygons)
        try:
            dist = poly1.distance(poly2)
            sys.stderr.write(f"[MergeCheck] Distance: {dist:.8f}\n")
            
            # More aggressive distance threshold for tile boundary splits
            if dist < 8e-5:  # MORE AGGRESSIVE: ~8 meters (was 5e-5) - aggressive for tile boundaries
                sys.stderr.write(f"[MergeCheck] ✅ MERGE: very close distance ({dist:.8f}) - likely tile boundary split\n")
                return True
            elif dist < 2e-4:  # MORE AGGRESSIVE: ~20 meters (was 1e-4) - needs additional validation but more lenient
                # Check if polygons are elongated/linear (common for cut polygons)
                area1, area2 = poly1.area, poly2.area
                try:
                    # Check aspect ratio - elongated polygons might be cuts
                    bounds1 = poly1.bounds
                    bounds2 = poly2.bounds
                    aspect1 = (bounds1[2] - bounds1[0]) / max(bounds1[3] - bounds1[1], 1e-10)  # width/height
                    aspect2 = (bounds2[2] - bounds2[0]) / max(bounds2[3] - bounds2[1], 1e-10)
                    
                    # If either polygon is very elongated, likely a boundary cut
                    if aspect1 > 5 or aspect1 < 0.2 or aspect2 > 5 or aspect2 < 0.2:
                        sys.stderr.write(f"[MergeCheck] ✅ MERGE: close distance + elongated shape (aspect1={aspect1:.2f}, aspect2={aspect2:.2f})\n")
                        return True
                    
                    # Check relative size - if very similar sizes, likely same original polygon
                    size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
                    if size_ratio > 0.2:  # More aggressive - similar sized polygons close together
                        sys.stderr.write(f"[MergeCheck] ✅ MERGE: close distance + similar sizes (ratio={size_ratio:.3f})\n")
                        return True
                    
                except Exception:
                    pass
                    
                sys.stderr.write(f"[MergeCheck] ⚠️  Close but additional checks failed ({dist:.8f})\n")
            elif dist < 0.0002:  # More aggressive - ~20 meters
                # Check for proximity area sharing
                try:
                    # Create buffers around both polygons to check proximity area
                    buffer1 = poly1.buffer(dist * 2)  # Buffer by twice the distance
                    buffer2 = poly2.buffer(dist * 2)
                    proximity_intersection = buffer1.intersection(buffer2)
                    
                    if proximity_intersection.area > 0:
                        # Calculate proximity sharing ratio
                        proximity_ratio = proximity_intersection.area / min(poly1.area, poly2.area)
                        if proximity_ratio > MERGE_PROXIMITY_RATIO:  # proximity area sharing threshold
                            sys.stderr.write(f"[MergeCheck] ✅ MERGE: significant proximity area sharing (ratio={proximity_ratio:.3f})\n")
                            return True
                except Exception:
                    pass
                    
                sys.stderr.write(f"[MergeCheck] ⚠️  Moderate distance ({dist:.8f}) - checking other criteria\n")
        except Exception as e:
            sys.stderr.write(f"[MergeCheck] Warning: distance calc failed: {e}\n")

        # RULE 2: If polygons touch geometrically, they share a tile seam → MERGE
        touches = False
        intersects = False
        try:
            touches = poly1.touches(poly2)
            intersects = poly1.intersects(poly2)
            sys.stderr.write(f"[MergeCheck] Testing: touches={touches}, intersects={intersects}\n")
            
            if touches:
                sys.stderr.write(f"[MergeCheck] ✅ MERGE: polygons touch (tile seam detected)\n")
                return True
        except Exception as e:
            sys.stderr.write(f"[MergeCheck] Warning: touches/intersects failed: {e}\n")

        # RULE 3: if they have area overlap large enough -> merge (more aggressive)
        try:
            inter = poly1.intersection(poly2)
            if inter and getattr(inter, "area", 0) > 0:
                overlap_ratio = inter.area / min(poly1.area if poly1.area else 1.0, poly2.area if poly2.area else 1.0)
                if overlap_ratio >= 0.005:  # 0.5% overlap - more aggressive
                    sys.stderr.write(f"[MergeCheck] ✅ MERGE: area overlap ({overlap_ratio:.3f})\n")
                    return True
        except Exception as e:
            sys.stderr.write(f"[MergeCheck] Warning: intersection failed: {e}\n")

        # RULE 4: Shared boundary analysis (only if geometries are valid)
        try:
            shared_len = _calculate_shared_boundary_length(poly1, poly2)

            # Require minimum shared boundary length for merging
            if shared_len > 1e-6:  # Require meaningful shared boundary
                sys.stderr.write(f"[MergeCheck] ✅ MERGE: significant shared boundary detected (length={shared_len:.8f})\n")
                return True
            elif shared_len > 0:
                sys.stderr.write(f"[MergeCheck] ❌ Shared boundary too small ({shared_len:.8f})\n")
        except Exception as e:
            sys.stderr.write(f"[MergeCheck] Warning: shared boundary check failed: {e}\n")

        sys.stderr.write(f"[MergeCheck] ❌ NO MERGE: no criteria met\n")
        return False
        
    except Exception as e:
        sys.stderr.write(f"[MergeCheck] Warning: exception in should_merge: {e}\n")
        # conservative fallback: if they touch or intersect, merge them
        try:
            return poly1.touches(poly2) or poly1.intersects(poly2)
        except Exception:
            return False


def _calculate_shared_boundary_length(poly1: Polygon, poly2: Polygon) -> float:
    """
    Compute the length of the shared boundary between two polygons.
    Uses direct boundary intersection first, then tolerant buffered method if needed.
    """
    try:
        # Check if polygons have valid boundaries
        if not hasattr(poly1, 'boundary') or not hasattr(poly2, 'boundary'):
            return 0.0
            
        b1 = poly1.boundary
        b2 = poly2.boundary
        
        # Check if boundaries are None or empty
        if b1 is None or b2 is None:
            sys.stderr.write(f"[CalcShared] Warning: None boundary detected\n")
            return 0.0
        
        if getattr(b1, 'is_empty', True) or getattr(b2, 'is_empty', True):
            return 0.0

        inter = b1.intersection(b2)
        if inter and (not inter.is_empty):
            # inter could be LineString / MultiLineString / GeometryCollection
            try:
                return float(inter.length)
            except Exception:
                # sum all lengths
                if hasattr(inter, "geoms"):
                    return sum(getattr(g, "length", 0.0) for g in inter.geoms)
                return 0.0

        # tolerant approach: small buffer around boundaries and estimate shared line length from overlap area
        try:
            tol = TOLERANT_BOUNDARY_TOL
            
            # Check if boundaries support buffer operation
            if not hasattr(b1, 'buffer') or not hasattr(b2, 'buffer'):
                return 0.0
                
            b1b = b1.buffer(tol)
            b2b = b2.buffer(tol)
            
            if b1b is None or b2b is None:
                return 0.0
                
            inter2 = b1b.intersection(b2b)
            if inter2 and (not inter2.is_empty) and getattr(inter2, "area", 0) > 0:
                approx_len = inter2.area / (2.0 * tol)
                return float(approx_len)
        except Exception as e:
            sys.stderr.write(f"[CalcShared] tolerant boundary failed: {e}\n")

        return 0.0

    except Exception as e:
        sys.stderr.write(f"[CalcShared] Warning: boundary calc failed: {e}\n")
        return 0.0


def _merge_polygon_group(polygon_group: List[Polygon], base_eps: float = BUFFER_EPS) -> Optional[Polygon]:
    """
    Merge a group of polygons robustly with REDESIGNED ultra-aggressive strategy.
    Goal: Remove internal seams and produce single unified polygon.
    
    Strategy:
      1. Use larger buffer to completely fill seams
      2. Union all buffered polygons
      3. Remove most of buffer but keep small amount to prevent splits
      4. Return largest component if still split
    """
    try:
        if len(polygon_group) == 1:
            return polygon_group[0]
            
        sys.stderr.write(f"[GroupMerge] Merging {len(polygon_group)} polygons with ultra-aggressive strategy\n")
        
        cleaned_polys = [make_valid(p) if not p.is_valid else p for p in polygon_group]
        
        # REDESIGNED STRATEGY: Use progressive larger buffers
        # Start with 3x base buffer to completely fill seams
        buffer_size = base_eps * 3.0  # ~3 meters
        
        sys.stderr.write(f"[GroupMerge] Applying buffer of {buffer_size:.8f} to fill seams\n")
        
        # Buffer all polygons
        buffered_polys = []
        for p in cleaned_polys:
            try:
                b = p.buffer(buffer_size)
                if b and not b.is_empty:
                    buffered_polys.append(b)
            except Exception as e:
                sys.stderr.write(f"[GroupMerge] Warning buffering polygon: {e}\n")
                buffered_polys.append(p)  # use original
        
        # Union all buffered polygons
        buffered_union = unary_union(buffered_polys)
        sys.stderr.write(f"[GroupMerge] Buffered union type: {type(buffered_union).__name__}\n")
        
        # Remove 90% of buffer (keep 10% to prevent splits at seams)
        removal_size = buffer_size * 0.9
        sys.stderr.write(f"[GroupMerge] Removing {removal_size:.8f} buffer\n")
        
        cleaned = buffered_union.buffer(-removal_size)
        
        if not cleaned.is_valid:
            cleaned = make_valid(cleaned)
            
        # Check result
        if isinstance(cleaned, Polygon):
            sys.stderr.write(f"[GroupMerge] ✅ Successfully merged into single polygon\n")
            return cleaned
            
        if isinstance(cleaned, MultiPolygon):
            num_parts = len(list(cleaned.geoms))
            sys.stderr.write(f"[GroupMerge] Result is MultiPolygon with {num_parts} parts\n")
            
            # If still multiple parts, try without buffer removal
            if isinstance(buffered_union, Polygon):
                sys.stderr.write(f"[GroupMerge] Using buffered union directly (no removal)\n")
                return buffered_union
                
            # Otherwise return largest component
            largest = max(cleaned.geoms, key=lambda g: g.area)
            total_area = sum(g.area for g in cleaned.geoms)
            largest_pct = (largest.area / total_area * 100) if total_area > 0 else 0
            sys.stderr.write(f"[GroupMerge] Using largest component ({largest_pct:.1f}% of total)\n")
            return largest
            
        # Fallback: try direct union
        sys.stderr.write(f"[GroupMerge] Fallback to direct union\n")
        direct = unary_union(cleaned_polys)
        if isinstance(direct, MultiPolygon):
            return max(direct.geoms, key=lambda g: g.area)
        return direct

    except Exception as e:
        sys.stderr.write(f"[GroupMerge] Exception merging polygon group: {e}\n")
        # Last resort fallback
        try:
            fallback = unary_union(polygon_group)
            if isinstance(fallback, MultiPolygon):
                return max(fallback.geoms, key=lambda g: g.area)
            return fallback
        except Exception:
            return polygon_group[0] if polygon_group else None
