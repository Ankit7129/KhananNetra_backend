# ================= FILE: ml_inference_subprocess.py ==================
"""
Subprocess script for TensorFlow inference.
This version accepts either JSON (legacy) or a binary numpy .npz blob on stdin.
The binary path preserves float32 values exactly and mirrors the notebook preprocessing
and postprocessing logic (patching, averaging, morphological filtering, polygon extraction).

Usage (legacy JSON):
  python ml_inference_subprocess.py /path/to/model.keras < tile.json

Usage (binary npz):
  The parent process should write a single JSON header line, then the raw bytes of a
  numpy .npz (from np.savez_compressed) immediately following the newline.
  Example header: {"binary": true, "tile_index": 0, "npz_size": 123456}\n<NPZ BYTES>

Output: JSON printed to stdout (same shape as notebook result dictionary)
"""

import sys
import json
import os
import io

# Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
# Memory management for TensorFlow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# Limit CPU threads to reduce memory pressure
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'


def load_input_from_stdin():
    """Reads input from stdin. Supports two modes:
       1) Text JSON on stdin (legacy): entire stdin is JSON with keys 'image_data' and 'tile_index'
       2) Binary stream: first line is JSON header (with 'binary': true), followed by raw npz bytes
          containing array named 'image'. This preserves dtype (float32) exactly.
    Returns: (image_array (np.ndarray), tile_index, tile_id, analysis_id)
    """
    try:
        import numpy as np
    except Exception:
        raise

    # Read raw bytes from stdin buffer
    raw = sys.stdin.buffer.read()
    if not raw:
        raise ValueError('No input received on stdin')

    # Try to decode first line as JSON header
    split_idx = raw.find(b"\n")
    if split_idx == -1:
        # No newline -> assume legacy full-text JSON
        text = raw.decode('utf-8')
        payload = json.loads(text)
        image = np.array(payload['image_data'], dtype=np.float32)
        tile_index = int(payload.get('tile_index', 0))
        tile_id = payload.get('tile_id')
        analysis_id = payload.get('analysis_id')
        return image, tile_index, tile_id, analysis_id

    header_bytes = raw[:split_idx]
    rest = raw[split_idx + 1:]

    try:
        header = json.loads(header_bytes.decode('utf-8'))
    except Exception:
        # Fallback: try full JSON
        text = raw.decode('utf-8')
        payload = json.loads(text)
        image = np.array(payload['image_data'], dtype=np.float32)
        tile_index = int(payload.get('tile_index', 0))
        tile_id = payload.get('tile_id')
        analysis_id = payload.get('analysis_id')
        return image, tile_index, tile_id, analysis_id

    if header.get('binary'):
        # rest should be the npz bytes
        bio = io.BytesIO(rest)
        npz = np.load(bio, allow_pickle=False)
        if 'image' in npz:
            image = np.array(npz['image'], dtype=np.float32)
        else:
            # try common alternatives
            keys = list(npz.files)
            if len(keys) == 0:
                raise ValueError('NPZ contained no arrays')
            image = np.array(npz[keys[0]], dtype=np.float32)
        tile_index = int(header.get('tile_index', 0))
        tile_id = header.get('tile_id')
        analysis_id = header.get('analysis_id')
        return image, tile_index, tile_id, analysis_id
    else:
        # Non-binary header -> treat rest as text JSON fragment
        text = raw.decode('utf-8')
        payload = json.loads(text)
        image = np.array(payload['image_data'], dtype=np.float32)
        tile_index = int(payload.get('tile_index', 0))
        tile_id = payload.get('tile_id')
        analysis_id = payload.get('analysis_id')
        return image, tile_index, tile_id, analysis_id


def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps({'success': False, 'error': 'Missing model path'}))
            sys.exit(1)

        model_path = sys.argv[1]

        # Import heavy libs after environment set with memory management
        import numpy as np
        import cv2
        from math import ceil
        import gc
        
        # Configure TensorFlow before import to prevent memory issues
        import tensorflow as tf
        
        # Limit TensorFlow memory growth to prevent OOM
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                sys.stderr.write(f"[DEBUG] Configured {len(gpus)} GPU(s) for memory growth\n")
            except RuntimeError as e:
                sys.stderr.write(f"[WARNING] GPU config failed: {e}\n")
        else:
            # Limit CPU memory for TensorFlow
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(4)
            sys.stderr.write("[DEBUG] Configured CPU threading limits\n")
        
        from tensorflow.keras.models import load_model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
        import contextlib

        with contextlib.redirect_stdout(sys.stderr):
            import segmentation_models as sm

        sys.stderr.write("[DEBUG] Libraries imported with memory management\n")
        sys.stderr.flush()

        # Load model with error handling and memory management
        try:
            sys.stderr.write("[DEBUG] Loading model...\n")
            sys.stderr.flush()
            
            # Clear any existing TensorFlow graphs/sessions
            tf.keras.backend.clear_session()
            gc.collect()
            
            dice_loss = sm.losses.DiceLoss()
            focal_loss = sm.losses.BinaryFocalLoss()
            total_loss = dice_loss + (1 * focal_loss)

            metrics = [
                sm.metrics.IOUScore(threshold=0.5, name='iou_score'),
                sm.metrics.FScore(threshold=0.5, name='f1_score'),
                sm.metrics.Precision(threshold=0.5),
                sm.metrics.Recall(threshold=0.5)
            ]

            # Load model without immediate compilation to save memory
            model = load_model(model_path, compile=False)
            sys.stderr.write("[DEBUG] Model loaded, compiling...\n")
            sys.stderr.flush()
            
            # Compile with reduced learning rate and memory-efficient optimizer
            model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss=total_loss,
                metrics=metrics,
                run_eagerly=False  # Use graph mode for better memory efficiency
            )
            
            sys.stderr.write("[DEBUG] Model compiled successfully\n")
            sys.stderr.flush()
            
            # Force garbage collection after model loading
            gc.collect()
            
        except Exception as e:
            sys.stderr.write(f"[ERROR] Model loading failed: {e}\n")
            sys.stderr.flush()
            raise RuntimeError(f"Failed to load model: {e}") from e

        # Read input from stdin (image numpy array)
        image_array, tile_index, tile_id, analysis_id = load_input_from_stdin()
        sys.stderr.write(f"[DEBUG] Received tile_index={tile_index}, tile_id={tile_id}, analysis_id={analysis_id[:8] if analysis_id else None}, image shape={image_array.shape}\n")
        sys.stderr.flush()

        H, W, C = image_array.shape

        # Normalize to [0,255] exactly like the notebook
        min_val, max_val = np.min(image_array), np.max(image_array)
        if max_val > min_val:
            large_image = 255.0 * (image_array - min_val) / (max_val - min_val)
        else:
            large_image = image_array.astype(np.float32)

        # Apply EfficientNet preprocess (notebook uses this on the 6-band image)
        preprocessed_large_image = efficientnet_preprocess(large_image)

        # Patch inference params (match notebook)
        patch_size = 256
        overlap = 0.25
        step = int(patch_size * (1 - overlap))

        pad_H = (ceil(H / step)) * step + patch_size - H
        pad_W = (ceil(W / step)) * step + patch_size - W
        padded_image = np.pad(preprocessed_large_image, ((0, pad_H), (0, pad_W), (0, 0)), mode='constant')

        # Create weighted blending for smooth transitions (eliminates squared artifacts)
        prediction_canvas = np.zeros((padded_image.shape[0], padded_image.shape[1], 1), dtype=np.float32)
        weight_canvas = np.zeros_like(prediction_canvas, dtype=np.float32)
        
        # Create Gaussian weight map for smooth blending
        def create_gaussian_weight_map(size):
            """Create a 2D Gaussian weight map for smooth blending at patch boundaries"""
            center = size // 2
            y, x = np.ogrid[:size, :size]
            # Distance from center
            dist_from_center = np.sqrt((x - center)**2 + (y - center)**2) 
            # Normalize to [0, 1] where center=1, edges approach 0
            max_dist = np.sqrt(2 * (center**2))
            weight_map = 1.0 - (dist_from_center / max_dist)
            weight_map = np.clip(weight_map, 0.1, 1.0)  # Minimum weight 0.1 to avoid division by zero
            return weight_map[:, :, np.newaxis]  # Add channel dimension
        
        # Pre-compute weight map for patches
        patch_weight = create_gaussian_weight_map(patch_size)

        # We'll collect patches into batches to avoid predict call per patch (faster and more stable)
        # Use smaller batch size to prevent memory exhaustion and segmentation faults
        batch_patches = []
        batch_coords = []
        batch_size = 4  # Reduced from 8 to prevent memory issues

        patch_count = 0
        for y in range(0, padded_image.shape[0] - patch_size + 1, step):
            for x in range(0, padded_image.shape[1] - patch_size + 1, step):
                patch = padded_image[y:y+patch_size, x:x+patch_size, :]
                batch_patches.append(patch)
                batch_coords.append((y, x))
                if len(batch_patches) >= batch_size:
                    try:
                        batch_arr = np.stack(batch_patches, axis=0)
                        sys.stderr.write(f"[DEBUG] Predicting batch of {len(batch_patches)} patches...\n")
                        sys.stderr.flush()
                        
                        preds = model.predict(batch_arr, verbose=0)
                        
                        for p_idx, (yy, xx) in enumerate(batch_coords):
                            pred_patch = preds[p_idx]
                            # Apply weighted blending instead of simple accumulation
                            prediction_canvas[yy:yy+patch_size, xx:xx+patch_size] += pred_patch * patch_weight
                            weight_canvas[yy:yy+patch_size, xx:xx+patch_size] += patch_weight
                            patch_count += 1
                        
                        # Clear memory after each batch
                        del batch_arr, preds
                        gc.collect()
                        
                        batch_patches = []
                        batch_coords = []
                        
                    except Exception as e:
                        sys.stderr.write(f"[ERROR] Batch prediction failed: {e}\n")
                        sys.stderr.flush()
                        raise RuntimeError(f"Inference failed on batch: {e}") from e

        # flush remaining patches
        if batch_patches:
            try:
                batch_arr = np.stack(batch_patches, axis=0)
                sys.stderr.write(f"[DEBUG] Predicting final batch of {len(batch_patches)} patches...\n")
                sys.stderr.flush()
                
                preds = model.predict(batch_arr, verbose=0)
                
                for p_idx, (yy, xx) in enumerate(batch_coords):
                    pred_patch = preds[p_idx]
                    # Apply weighted blending instead of simple accumulation
                    prediction_canvas[yy:yy+patch_size, xx:xx+patch_size] += pred_patch * patch_weight
                    weight_canvas[yy:yy+patch_size, xx:xx+patch_size] += patch_weight
                    patch_count += 1
                
                # Clear memory after final batch
                del batch_arr, preds
                gc.collect()
                
            except Exception as e:
                sys.stderr.write(f"[ERROR] Final batch prediction failed: {e}\n")
                sys.stderr.flush()
                raise RuntimeError(f"Inference failed on final batch: {e}") from e

        sys.stderr.write(f"[DEBUG] Processed {patch_count} patches with Gaussian weighted blending\n")
        sys.stderr.flush()

        # Normalize by accumulated weights for smooth blending
        weight_canvas[weight_canvas == 0] = 1  # Avoid division by zero
        prediction_canvas = prediction_canvas / weight_canvas

        # Crop to original size
        full_mask_prob = prediction_canvas[:H, :W, :]
        
        # Apply mild Gaussian smoothing to eliminate any remaining patch boundary artifacts
        from scipy import ndimage
        sys.stderr.write(f"[DEBUG] Applying Gaussian smoothing to eliminate patch boundaries...\n")
        sys.stderr.flush()
        
        # Very mild smoothing (sigma=1.0) to smooth patch boundaries without losing detail
        full_mask_prob = ndimage.gaussian_filter(full_mask_prob, sigma=1.0, mode='reflect')
        
        # ============= POLYGON POST-PROCESSING (Notebook Logic) =============
        # Import the post-processing function
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from utils.postprocess import process_prediction_to_polygons
        
        # Get georeference from environment (passed by parent process)
        # Format: "a,b,c,d,e,f|EPSG:4326" where a-f are affine transform coefficients
        georef_str = os.environ.get('TILE_GEOREF', '')
        transform_coeffs = None
        crs_str = None
        
        if georef_str:
            parts = georef_str.split('|')
            if len(parts) == 2:
                try:
                    coeffs = [float(x) for x in parts[0].split(',')]
                    if len(coeffs) == 6:
                        # Rasterio Affine transform: (a, b, c, d, e, f)
                        from rasterio.transform import Affine
                        transform_coeffs = Affine(coeffs[0], coeffs[1], coeffs[2],
                                                 coeffs[3], coeffs[4], coeffs[5])
                        crs_str = parts[1]
                        sys.stderr.write(f"[DEBUG] Loaded georef: {crs_str}, transform={transform_coeffs}\n")
                        sys.stderr.flush()
                except Exception as e:
                    sys.stderr.write(f"[WARNING] Failed to parse georef: {e}\n")
                    sys.stderr.flush()
        
        # Post-processing parameters - SMOOTH PREDICTIONS WITH PROPER BLENDING
        # With Gaussian blending, we can use a more appropriate threshold
        threshold = 0.7               # Standard threshold (50%) - smooth predictions allow this
        min_area_pixels = 1000       # Filter out mine blocks under 1000 pixels (substantial blocks only)
        min_area_meters = 1000.0     # 0.1 hectares minimum (capture small operations)
        morphology_kernel = 2        # Smaller kernel to preserve exact boundaries
        use_adaptive_threshold = False  # DISABLED - use fixed threshold

        sys.stderr.write(f"[DEBUG] INDUSTRIAL RAW Post-processing: threshold={threshold}, min_area={min_area_pixels}px/{min_area_meters}m², kernel={morphology_kernel}\n")
        sys.stderr.flush()

        # Run polygon post-processing with MINIMAL FILTERING (Industrial Grade)
        sys.stderr.write(f"[DEBUG] Processing raw predictions with minimal filtering - preserving model output...\n")
        sys.stderr.flush()
        
        geojson_result = process_prediction_to_polygons(
            prob_map=full_mask_prob[:, :, 0],
            threshold=threshold,
            min_area_pixels=min_area_pixels,
            min_area_meters=min_area_meters,
            morphology_kernel=morphology_kernel,
            simplify_tol=0.0,  # NO SIMPLIFICATION - preserve exact boundaries
            transform=transform_coeffs,
            crs=crs_str,
            use_adaptive_threshold=use_adaptive_threshold,
            keep_holes=False,  # Fill holes for solid blocks
            tile_id=tile_id,  # Pass tile ID for unique block IDs
            analysis_id=analysis_id  # Pass analysis ID for unique block IDs
        )
        
        # Extract statistics from GeoJSON (enhanced structure)
        mine_blocks = geojson_result.get('features', [])
        metadata = geojson_result.get('metadata', {})
        visualization = geojson_result.get('visualization', {})
        
        num_mine_blocks = metadata.get('block_count', len(mine_blocks))
        total_area_m2 = metadata.get('total_area_m2', 0.0)  # Default to 0 if missing
        threshold_used = metadata.get('threshold_used', threshold)
        
        # Calculate aggregate statistics
        pred_values = full_mask_prob[:, :, 0]
        total_pixels = H * W
        
        # Count mining pixels from all detected blocks
        mining_pixels = 0
        total_block_confidence = 0.0
        
        for feature in mine_blocks:
            props = feature.get('properties', {})
            mining_pixels += props.get('area_px', 0)
            total_block_confidence += props.get('avg_confidence', 0.0)
        
        mining_percentage = float(mining_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
        avg_confidence = float(total_block_confidence / num_mine_blocks) if num_mine_blocks > 0 else 0.0

        sys.stderr.write(f"[DEBUG] ✅ ENHANCED Post-processing complete: {num_mine_blocks} numbered blocks\n")
        if total_area_m2 > 0:
            sys.stderr.write(f"[DEBUG] Total mine area: {total_area_m2/1e6:.3f} km² ({total_area_m2/10000:.2f} ha, {mining_percentage:.2f}% coverage)\n")
        sys.stderr.write(f"[DEBUG] Threshold used: {threshold_used:.3f} (adaptive={use_adaptive_threshold})\n")
        sys.stderr.flush()

        # ============= GENERATE HIGH-QUALITY PROBABILITY MAP =============
        # Create smooth blue-scheme heatmap with better color gradients
        sys.stderr.write(f"[DEBUG] Generating high-quality probability map visualization...\n")
        sys.stderr.flush()
        
        import base64
        
        # Enhanced colormap for better visualization
        # Apply smooth gradients from dark blue (low) to bright cyan (high)
        prob_map_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Create smooth color transitions
        for i in range(H):
            for j in range(W):
                prob_val = pred_values[i, j]  # 0.0 to 1.0
                
                if prob_val < 0.1:
                    # Very low: Dark blue/black
                    prob_map_rgb[i, j] = [0, 0, int(prob_val * 10 * 50)]
                elif prob_val < 0.3:
                    # Low: Dark blue to blue
                    t = (prob_val - 0.1) / 0.2
                    prob_map_rgb[i, j] = [0, 0, int(50 + t * 100)]
                elif prob_val < 0.7:
                    # Medium: Blue to light blue
                    t = (prob_val - 0.3) / 0.4
                    prob_map_rgb[i, j] = [0, int(t * 150), int(150 + t * 105)]
                else:
                    # High: Light blue to bright cyan
                    t = (prob_val - 0.7) / 0.3
                    prob_map_rgb[i, j] = [int(t * 100), int(150 + t * 105), 255]
        
        # Convert RGB to BGR for OpenCV
        prob_map_bgr = cv2.cvtColor(prob_map_rgb, cv2.COLOR_RGB2BGR)
        
        # Add boundary contours to probability map for better visualization
        binary_thresh = (pred_values > threshold).astype(np.uint8)
        contours_viz, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Draw exact prediction boundaries in white
        cv2.drawContours(prob_map_bgr, contours_viz, -1, (255, 255, 255), thickness=1)
        
        # Draw threshold line contours in yellow (0.3 threshold)
        thresh_line = (pred_values > threshold).astype(np.uint8)
        thresh_contours, _ = cv2.findContours(thresh_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(prob_map_bgr, thresh_contours, -1, (0, 255, 255), thickness=2)
        
        # Encode as high-quality PNG
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]  # Less compression for better quality
        _, buffer = cv2.imencode('.png', prob_map_bgr, encode_params)
        prob_map_base64 = base64.b64encode(buffer).decode('utf-8')
        
        sys.stderr.write(f"[DEBUG] ✅ High-quality probability map with boundaries generated ({H}×{W} pixels)\n")
        sys.stderr.flush()

        result = {
            'success': True,
            'tile_index': int(tile_index),
            'mining_pixels': mining_pixels,
            'total_pixels': total_pixels,
            'mining_percentage': mining_percentage,
            'confidence': avg_confidence,
            'max_prediction': float(np.max(pred_values)),
            'mean_prediction': float(np.mean(pred_values)),
            'mask_shape': [H, W],
            'mine_blocks': mine_blocks,  # GeoJSON features with numbered blocks (Block 1, Block 2, ...)
            'num_mine_blocks': num_mine_blocks,
            'total_area_m2': total_area_m2,
            'threshold_used': threshold_used,
            'geojson': geojson_result,  # Full GeoJSON with metadata and visualization params
            'metadata': metadata,  # Processing metadata
            'visualization': visualization,  # Overlay parameters (transparency, colors)
            'probability_map_base64': prob_map_base64  # RAW prediction heatmap (blue scheme)
        }

        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        import traceback
        sys.stderr.write(f"[ERROR] {traceback.format_exc()}\n")
        sys.stderr.flush()
        try:
            tile_index = locals().get('tile_index', -1)
        except Exception:
            tile_index = -1
        print(json.dumps({'success': False, 'error': str(e), 'tile_index': tile_index}))
        sys.exit(1)


if __name__ == '__main__':
    main()

