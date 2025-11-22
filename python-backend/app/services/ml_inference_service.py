
# ================= FILE: ml_inference_service.py ==================
"""
ML Inference Service that launches the subprocess. This updated version
sends the image as a compressed npz blob (binary) to preserve dtype and avoid
precision/normalization drift caused by JSON serialization.

The service otherwise mirrors the notebook logic (normalization -> efficientnet preprocess -> patching -> same thresholding/post-processing).
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict
import numpy as np
import io

# Import model loader
from app.utils.model_loader import get_model_path


class MLInferenceService:
    def __init__(self):
        print(f"🤖 Initializing ML Inference Service...")
        
        # Get model path from model_loader (handles Kaggle download/caching)
        self.model_path = get_model_path()
        
        # Subprocess script location
        self.subprocess_script = os.path.join(Path(__file__).parent.absolute(), 'ml_inference_subprocess.py')

        print(f"   📦 Model path: {self.model_path}")
        print(f"   📄 Model exists: {os.path.exists(self.model_path)}")
        print(f"   🔧 Subprocess script: {self.subprocess_script}")
        print(f"   📄 Script exists: {os.path.exists(self.subprocess_script)}")

        if os.path.exists(self.model_path):
            file_size = os.path.getsize(self.model_path)
            print(f"   📊 Model file size: {file_size / (1024*1024):.1f} MB")
        else:
            print(f"   ⚠️  WARNING: Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if not os.path.exists(self.subprocess_script):
            print(f"   ⚠️  WARNING: Subprocess script not found!")
            raise FileNotFoundError(f"Subprocess script not found: {self.subprocess_script}")

        print(f"   ✅ TensorFlow will run in separate subprocess (avoids bus errors)")

        self.input_size = 256
        self.expected_bands = 6
        self.max_dimension = int(os.getenv('ML_MAX_MOSAIC_DIMENSION', '4096'))
        self.max_pixels = int(os.getenv('ML_MAX_MOSAIC_PIXELS', '12000000'))  # ~3464²
        print(f"   📐 Max mosaic dimension: {self.max_dimension}")
        print(f"   🧮 Max mosaic pixels: {self.max_pixels:,}")

    def validate_input_data(self, image_data: np.ndarray) -> tuple:
        if not isinstance(image_data, np.ndarray):
            return False, f"Image data must be numpy array, got {type(image_data)}"
        if image_data.ndim != 3:
            return False, f"Image must be 3D (H, W, C), got shape {image_data.shape}"
        H, W, C = image_data.shape
        if C != self.expected_bands:
            return False, f"Expected {self.expected_bands} bands, got {C} bands"
        if H < self.input_size or W < self.input_size:
            return False, f"Image too small: {H}x{W} (minimum {self.input_size}x{self.input_size})"
        if np.isnan(image_data).any():
            return False, "Image contains NaN values"
        if np.isinf(image_data).any():
            return False, "Image contains infinite values"
        return True, None

    def run_inference_on_tile(self, tile_data: dict) -> dict:
        tile_index = int(tile_data.get('tile_index', 0))
        print(f"   🔬 Running inference on tile {tile_index} via subprocess...")
        try:
            image_data = tile_data.get('image_data')
            if image_data is None:
                return {'success': False, 'error': "Missing 'image_data' in tile_data", 'tile_index': tile_index}

            is_valid, error_msg = self.validate_input_data(image_data)
            if not is_valid:
                print(f"   ❌ Input validation failed: {error_msg}")
                return {'success': False, 'error': f"Input validation failed: {error_msg}", 'tile_index': tile_index}

            print(f"   📊 Input shape: {image_data.shape}")
            print(f"   📊 Input range: [{image_data.min():.1f}, {image_data.max():.1f}]")
            print(f"   📊 Input dtype: {image_data.dtype}")

            if image_data.dtype != np.float32:
                print(f"   🔄 Converting from {image_data.dtype} to float32")
                image_data = image_data.astype(np.float32)

            original_height, original_width = image_data.shape[:2]
            scale_x = 1.0
            scale_y = 1.0

            def apply_resize(new_height: float, new_width: float, reason: str) -> None:
                nonlocal image_data, scale_x, scale_y
                current_height, current_width = image_data.shape[:2]
                target_height = max(self.input_size, int(round(new_height)))
                target_width = max(self.input_size, int(round(new_width)))
                target_height = max(1, target_height)
                target_width = max(1, target_width)

                if target_height == current_height and target_width == current_width:
                    return

                try:
                    import cv2
                    interpolation = cv2.INTER_AREA if target_height < current_height or target_width < current_width else cv2.INTER_LINEAR
                    image_data = cv2.resize(
                        image_data,
                        (target_width, target_height),
                        interpolation=interpolation
                    )
                except Exception:
                    from skimage.transform import resize
                    image_data = resize(
                        image_data,
                        (target_height, target_width, image_data.shape[2]),
                        preserve_range=True,
                        anti_aliasing=True
                    ).astype(np.float32)

                scale_x *= current_width / float(target_width)
                scale_y *= current_height / float(target_height)
                print(
                    f"   📏 {reason}: {current_height}x{current_width} -> {target_height}x{target_width} "
                    f"(scale_x={scale_x:.3f}, scale_y={scale_y:.3f})"
                )

            if max(original_height, original_width) > self.max_dimension:
                dominant = float(max(original_height, original_width))
                scale_factor = self.max_dimension / dominant
                apply_resize(original_height * scale_factor, original_width * scale_factor, "Downscaled to ML_MAX_MOSAIC_DIMENSION")

            height, width = image_data.shape[:2]
            pixel_count = height * width
            if pixel_count > self.max_pixels:
                pixel_scale = (self.max_pixels / float(pixel_count)) ** 0.5
                apply_resize(height * pixel_scale, width * pixel_scale, f"Downscaled to cap pixels at {self.max_pixels:,}")
                height, width = image_data.shape[:2]

            print(f"   🧮 Mosaic dimensions for inference: {height}x{width} ({height*width:,} px)")

            # Calculate Affine transform from tile bounds (CRITICAL for alignment)
            # This maps pixel coordinates to geographic coordinates (lat/lon)
            bounds = tile_data.get('bounds')
            georef_env = {}
            
            # PRIORITY 1: Use exact transform from tile if available (most accurate)
            if 'transform' in tile_data and tile_data['transform'] is not None:
                transform = tile_data['transform']  # This is already a rasterio.Affine object
                crs = tile_data.get('crs', 'EPSG:4326')
                try:
                    from rasterio.transform import Affine
                    if not isinstance(transform, Affine):
                        transform = Affine(*transform)
                except Exception:
                    transform = None
                
                if transform is not None and (scale_x != 1.0 or scale_y != 1.0):
                    transform = transform * Affine.scale(scale_x, scale_y)
                    print(
                        f"   📐 Adjusted transform for scaled mosaic: "
                        f"scale_x={scale_x:.3f}, scale_y={scale_y:.3f}"
                    )

                if transform is not None:
                    # Format as string: "a,b,c,d,e,f|CRS"
                    transform_str = (
                        f"{transform.a},{transform.b},{transform.c},"
                        f"{transform.d},{transform.e},{transform.f}|{crs}"
                    )
                    georef_env['TILE_GEOREF'] = transform_str

                    print(f"   🎯 Using EXACT transform from tile:")
                    print(f"   🗺️  Transform: |{transform.a:8.6f}, {transform.b:8.6f}, {transform.c:8.4f}|")
                    print(f"   🗺️            |{transform.d:8.6f}, {transform.e:8.6f}, {transform.f:8.4f}|")
                    print(f"   🗺️  CRS: {crs}")
                else:
                    print("   ⚠️  Unable to use provided transform after scaling")
                
            # FALLBACK: Calculate from bounds if exact transform not available  
            if ('transform' not in tile_data or tile_data.get('transform') is None or 'TILE_GEOREF' not in georef_env) and bounds and len(bounds) >= 4:
                # Bounds format: [[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat], ...]
                min_lon = min([b[0] for b in bounds])
                max_lon = max([b[0] for b in bounds])
                min_lat = min([b[1] for b in bounds])
                max_lat = max([b[1] for b in bounds])
                
                # Calculate Affine transform coefficients
                pixel_width = (max_lon - min_lon) / width
                pixel_height = -(max_lat - min_lat) / height  # Negative for north-up orientation
                
                # Create transform: pixel (0,0) maps to (min_lon, max_lat) = top-left corner
                from rasterio.transform import Affine
                transform = Affine(
                    pixel_width,  # a: x pixel size
                    0.0,          # b: row rotation
                    min_lon,      # c: top-left x (longitude)
                    0.0,          # d: column rotation
                    pixel_height, # e: y pixel size (negative)
                    max_lat       # f: top-left y (latitude)
                )
                
                # Format as string: "a,b,c,d,e,f|EPSG:4326"
                transform_str = f"{transform.a},{transform.b},{transform.c},{transform.d},{transform.e},{transform.f}|EPSG:4326"
                georef_env['TILE_GEOREF'] = transform_str
                
                print(f"   🗺️  Calculated georef from bounds: [{min_lon:.4f}, {min_lat:.4f}] to [{max_lon:.4f}, {max_lat:.4f}]")
                print(f"   🗺️  Transform: |{transform.a:8.6f}, {transform.b:8.6f}, {transform.c:8.4f}|")
                print(f"   🗺️            |{transform.d:8.6f}, {transform.e:8.6f}, {transform.f:8.4f}|")
                print(f"   🗺️  Pixel size: {pixel_width:.6f}° lon × {abs(pixel_height):.6f}° lat")
            elif 'TILE_GEOREF' not in georef_env:
                print(f"   ⚠️  No transform or bounds provided - polygons will be in pixel coordinates only")

            # Prepare binary npz payload to preserve dtype and reduce JSON drift
            bio = io.BytesIO()
            # Save under key 'image'
            np.savez_compressed(bio, image=image_data)
            npz_bytes = bio.getvalue()

            # Include tile_id and analysis_id in header for unique block IDs
            header_data = {
                'binary': True, 
                'tile_index': tile_index,
                'tile_id': tile_data.get('tile_id'),
                'analysis_id': tile_data.get('analysis_id')
            }
            header = json.dumps(header_data).encode('utf-8') + b"\n"
            payload = header + npz_bytes

            print(f"   📤 Sending {len(payload)} bytes (npz) to subprocess")

            # Merge georef environment with current environment
            env = os.environ.copy()
            env.update(georef_env)

            # Run subprocess with retry mechanism for segmentation faults
            max_retries = 2
            for attempt in range(max_retries + 1):
                if attempt > 0:
                    print(f"   🔄 Retry attempt {attempt}/{max_retries} for tile {tile_index}")
                    # Add small delay between retries to allow system recovery
                    import time
                    time.sleep(1)
                
                try:
                    result = subprocess.run(
                        [sys.executable, self.subprocess_script, self.model_path],
                        input=payload,
                        capture_output=True,
                        timeout=300,
                        env=env  # Pass environment with TILE_GEOREF
                    )
                    
                    # Check for segmentation fault (return code -11 on Unix systems)
                    if result.returncode == -11:
                        print(f"   ⚠️  Subprocess segmentation fault (attempt {attempt + 1})")
                        if attempt < max_retries:
                            continue  # Retry
                        else:
                            print(f"   ❌ Subprocess failed with segmentation fault after {max_retries + 1} attempts")
                            return {
                                'success': False, 
                                'error': f"Subprocess segmentation fault after {max_retries + 1} attempts", 
                                'tile_index': tile_index,
                                'retry_attempts': max_retries + 1
                            }
                    
                    # Other non-zero return codes
                    if result.returncode != 0:
                        print(f"   ❌ Subprocess failed with return code {result.returncode}")
                        stderr_preview = (result.stderr.decode('utf-8', errors='ignore') if isinstance(result.stderr, (bytes, bytearray)) else result.stderr)
                        print(f"   📋 Subprocess stderr (last 2000 chars): {stderr_preview[-2000:]}")
                        
                        # Don't retry for other types of errors
                        return {
                            'success': False, 
                            'error': f"Subprocess failed (code {result.returncode})", 
                            'tile_index': tile_index, 
                            'raw_stderr': stderr_preview
                        }
                    
                    # Success - break out of retry loop
                    break
                    
                except subprocess.TimeoutExpired:
                    print(f"   ⏰ Subprocess timeout (attempt {attempt + 1})")
                    if attempt < max_retries:
                        continue  # Retry
                    else:
                        return {
                            'success': False, 
                            'error': f"Subprocess timeout after {max_retries + 1} attempts", 
                            'tile_index': tile_index
                        }
                        
            # If we get here, subprocess succeeded

            # Print subprocess debug
            if result.stderr:
                stderr_txt = result.stderr.decode('utf-8', errors='ignore') if isinstance(result.stderr, (bytes, bytearray)) else result.stderr
                print(f"   📋 Subprocess stderr:\n{stderr_txt}")

            stdout_txt = result.stdout.decode('utf-8', errors='ignore') if isinstance(result.stdout, (bytes, bytearray)) else result.stdout
            stdout_clean = stdout_txt.strip()
            if not stdout_clean:
                return {'success': False, 'error': 'Subprocess produced no output', 'tile_index': tile_index, 'raw_stdout': stdout_txt}

            # Robust JSON parsing: some libraries may emit logs to stdout. Try direct parse first,
            # then attempt to extract the JSON object substring between the first '{' and last '}' if needed.
            try:
                inference_result = json.loads(stdout_clean)
            except json.JSONDecodeError:
                first_idx = stdout_txt.find('{')
                last_idx = stdout_txt.rfind('}')
                if first_idx != -1 and last_idx != -1 and last_idx > first_idx:
                    candidate = stdout_txt[first_idx:last_idx+1]
                    try:
                        inference_result = json.loads(candidate)
                        print(f"   ⚠️  Parsed JSON from noisy stdout (extraneous logs present). Trimmed stdout length {len(stdout_txt)} -> {len(candidate)}")
                    except Exception:
                        print("   ❌ Failed to parse JSON from subprocess stdout after trimming")
                        print(f"   🔎 Raw stdout (first 2000 chars): {stdout_txt[:2000]}")
                        return {'success': False, 'error': 'Failed to parse subprocess JSON output', 'tile_index': tile_index, 'raw_stdout': stdout_txt}
                else:
                    print("   ❌ Subprocess stdout did not contain a JSON object")
                    print(f"   🔎 Raw stdout (first 2000 chars): {stdout_txt[:2000]}")
                    return {'success': False, 'error': 'Subprocess stdout malformed (no JSON)', 'tile_index': tile_index, 'raw_stdout': stdout_txt}

            if inference_result.get('success'):
                mining_pct = inference_result.get('mining_percentage', 0)
                num_blocks = inference_result.get('num_mine_blocks', 0)
                confidence = inference_result.get('confidence', 0)
                print(f"   ✅ Success: {mining_pct:.2f}% mining, {num_blocks} blocks, {confidence:.1f}% confidence")
            else:
                print(f"   ❌ Inference failed: {inference_result.get('error', 'Unknown')}")

            return inference_result

        except subprocess.TimeoutExpired:
            print(f"   ❌ Inference timed out after 5 minutes")
            return {'success': False, 'error': 'Inference timed out (5 min limit)', 'tile_index': tile_index}
        except MemoryError:
            print(f"   ❌ Out of memory during inference")
            return {'success': False, 'error': 'Out of memory - image may be too large', 'tile_index': tile_index}
        except Exception as e:
            import traceback
            print(f"   ❌ Unexpected error: {e}")
            print(traceback.format_exc())
            return {'success': False, 'error': f"Unexpected error: {str(e)}", 'tile_index': tile_index}

    def get_model_info(self) -> dict:
        return {
            'model_path': self.model_path,
            'model_exists': os.path.exists(self.model_path),
            'script_path': self.subprocess_script,
            'script_exists': os.path.exists(self.subprocess_script),
            'input_size': self.input_size,
            'expected_bands': self.expected_bands,
            'model_size_mb': os.path.getsize(self.model_path) / (1024*1024) if os.path.exists(self.model_path) else 0
        }

    def preload_model(self) -> dict:
        """Ensure model artifacts are present before handling requests."""
        info = self.get_model_info()
        if not info['model_exists']:
            raise FileNotFoundError(f"Model file missing at {self.model_path}")
        return info


# Singleton helper
_ml_service = None

def get_ml_service() -> MLInferenceService:
    global _ml_service
    if _ml_service is None:
        _ml_service = MLInferenceService()
    return _ml_service


def cleanup_ml_service():
    global _ml_service
    _ml_service = None
