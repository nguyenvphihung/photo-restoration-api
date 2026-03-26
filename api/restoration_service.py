import os
import sys
import cv2
import httpx
from pathlib import Path
from typing import Dict, Any

# Worker endpoints
ZEROSCRATCHES_URL = "http://127.0.0.1:8001/process"
GFPGAN_URL        = "http://127.0.0.1:8002/enhance"
COLORIZATION_URL  = "http://127.0.0.1:8003/colorize"
ENHANCER_URL      = "http://127.0.0.1:8004/enhance"
CODEFORMER_URL    = "http://127.0.0.1:8005/enhance"

# Max input resolution (to prevent OOM on 4GB GPU)
MAX_INPUT_WIDTH  = 800
MAX_INPUT_HEIGHT = 800


def resize_if_needed(image_path: str,
                     max_width: int = MAX_INPUT_WIDTH,
                     max_height: int = MAX_INPUT_HEIGHT) -> tuple:
    """
    Resize image if it exceeds max dimensions to prevent GPU OOM.
    Returns: (resized_path, was_resized, original_size)
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return image_path, False, None

    # Handle RGBA (4-channel) images → convert to BGR
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # Re-save as 3-channel so downstream workers don't hit the same issue
        cv2.imwrite(image_path, img)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(image_path, img)

    h, w = img.shape[:2]
    original_size = (w, h)

    if w <= max_width and h <= max_height:
        return image_path, False, original_size

    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    print(f"[Pipeline] Resizing: {w}x{h} → {new_w}x{new_h} (scale {scale:.2f})")

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    resized_path = str(Path(image_path).parent / f"_resized_{Path(image_path).name}")
    cv2.imwrite(resized_path, resized)
    return resized_path, True, original_size


# ============================================================================
# PUBLIC PIPELINE ENTRY POINTS
# ============================================================================

def run_restoration_pipeline(input_image_path: str, **_) -> Dict[str, Any]:
    """Repair scratches + restore faces + sharpen (no colorization)."""
    return _execute_pipeline(input_image_path, run_restore=True, run_color=False, run_enhance=True)

def run_colorization_pipeline(input_image_path: str, **_) -> Dict[str, Any]:
    """Colorize only (no scratch removal, no face restore)."""
    return _execute_pipeline(input_image_path, run_restore=False, run_color=True, run_enhance=False)

def run_full_pipeline(input_image_path: str, **_) -> Dict[str, Any]:
    """Full restore + colorize + sharpen."""
    return _execute_pipeline(input_image_path, run_restore=True, run_color=True, run_enhance=True)


# ============================================================================
# UTILITIES
# ============================================================================
def estimate_max_face_ratio(image_path: str) -> float:
    """
    Returns the Area Ratio of the largest face detected using fast Haar Cascades.
    0.0 if no face detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load default OpenCV face cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return 0.0
        
    img_area = img.shape[0] * img.shape[1]
    max_face_area = max([w * h for (x, y, w, h) in faces])
    return max_face_area / img_area


# ============================================================================
# CORE PIPELINE
# ============================================================================

def _execute_pipeline(input_image_path: str,
                      run_restore: bool,
                      run_color: bool,
                      run_enhance: bool = False) -> Dict[str, Any]:
    """
    Correct step order:
      1. ZeroScratches  – remove scratches on the grayscale/faded original
      2. Colorization   – add color (internally converts to grayscale anyway,
                          so it MUST run before GFPGAN to not throw away face work)
      3. GFPGAN         – restore faces on the already-colorized image
                          (GFPGAN works on color images; its result is preserved
                          because nothing downstream converts to grayscale)
      4. Enhancer       – sharpen the final result

    WHY this order matters:
      colorization_worker.py does img.convert('L') before inference, meaning it
      ALWAYS discards color information from its input. If GFPGAN ran first, every
      face detail it restored would be thrown away in the very next step.
      Running colorization BEFORE GFPGAN ensures GFPGAN's output is the final
      high-quality face we actually keep.
    """
    try:
        input_path = Path(input_image_path)
        if not input_path.exists():
            return {'success': False, 'error': f'Input file not found: {input_image_path}'}

        project_root = Path(__file__).parent.parent.absolute()
        outputs_dir  = project_root / 'outputs'
        outputs_dir.mkdir(exist_ok=True)

        print(f"[Pipeline] Processing: {input_image_path}")
        process_path, was_resized, original_size = resize_if_needed(str(input_path))

        # Output paths for each step
        step1_path       = str(outputs_dir / 'step1_zeroscratches.jpg')
        step2_path       = str(outputs_dir / 'step2_colorized.jpg')
        step3_path       = str(outputs_dir / 'step3_face_restore.jpg')
        final_path       = str(outputs_dir / 'final_output.jpg')
        intermediate_path = None
        faces_detected   = 0

        current_img_path = str(Path(process_path).absolute())

        with httpx.Client(timeout=120.0) as client:

            # ------------------------------------------------------------------
            # STEP 1: ZeroScratches — remove scratches / noise from the original
            # ------------------------------------------------------------------
            if run_restore:
                print("[Pipeline] Step 1/3: ZeroScratches (scratch removal)...")
                try:
                    resp = client.post(ZEROSCRATCHES_URL, json={
                        "image_path": current_img_path,
                        "output_path": step1_path
                    })
                    resp.raise_for_status()
                    if not resp.json().get('success'):
                        return {'success': False, 'error': 'ZeroScratches worker failed'}
                    current_img_path = step1_path
                    intermediate_path = step1_path
                    print("[Pipeline] Step 1 done.")
                except Exception as exc:
                    return {'success': False, 'error': f'ZeroScratches failed: {exc}'}

            # ------------------------------------------------------------------
            # STEP 2: Colorization — MUST run before face restore.
            #   colorization_worker converts input to grayscale internally,
            #   so face restoration's (color + detail) would be lost
            #   if this step ran after face restore.
            # ------------------------------------------------------------------
            if run_color:
                # If restore follows, we write to an intermediate file
                out_path = step2_path if run_restore else final_path

                print("[Pipeline] Step 2/3: Colorization...")
                try:
                    resp = client.post(COLORIZATION_URL, json={
                        "image_path": current_img_path,
                        "output_path": out_path
                    })
                    resp.raise_for_status()
                    if not resp.json().get('success'):
                        return {'success': False, 'error': 'Colorization worker failed'}
                    current_img_path = out_path
                    if not intermediate_path:
                        intermediate_path = out_path
                    print("[Pipeline] Step 2 done.")
                except Exception as exc:
                    return {'success': False, 'error': f'Colorization failed: {exc}'}

            # ------------------------------------------------------------------
            # STEP 3: Adaptive Face Restore & Background Upsample
            #   - Portrait (Face >= 5% area) -> GFPGAN
            #   - Group / Far away (Face < 5%) -> CodeFormer
            #   (Both models internally use RealESRGAN to upscale the background)
            # ------------------------------------------------------------------
            if run_restore:
                # Smart Routing logic
                face_ratio = estimate_max_face_ratio(current_img_path)
                if face_ratio >= 0.05:
                    target_url = GFPGAN_URL
                    model_name = "GFPGAN"
                else:
                    target_url = CODEFORMER_URL
                    model_name = "CodeFormer"
                
                print(f"[Pipeline] Step 3/3: Face & Background Restoration...")
                print(f"  Max face ratio: {face_ratio*100:.1f}% -> Routing to {model_name}")
                
                try:
                    resp = client.post(target_url, json={
                        "image_path": current_img_path,
                        "output_path": final_path
                    })
                    resp.raise_for_status()
                    res = resp.json()
                    if not res.get('success'):
                        return {'success': False, 'error': f'{model_name} worker failed'}
                    
                    faces_detected   = res.get('faces_detected', 0)
                    current_img_path = final_path
                    if not intermediate_path:
                        intermediate_path = final_path
                    print(f"[Pipeline] Step 3 done. Faces detected: {faces_detected} by {model_name}")
                except Exception as exc:
                    return {'success': False, 'error': f'{model_name} failed: {exc}'}

        # ------------------------------------------------------------------
        # Read final image for size info
        # ------------------------------------------------------------------
        final_image = cv2.imread(current_img_path)
        if final_image is None:
            return {'success': False, 'error': 'Failed to read final output'}

        height, width = final_image.shape[:2]

        if was_resized and original_size:
            orig_width, orig_height = original_size
            try:
                Path(process_path).unlink()
            except Exception:
                pass
        else:
            src = cv2.imread(str(input_path))
            orig_height, orig_width = src.shape[:2]

        return {
            'success': True,
            'output_path': current_img_path,
            'intermediate_path': intermediate_path,
            'image_size': {
                'original':  {'width': orig_width,  'height': orig_height},
                'restored':  {'width': width,        'height': height},
                'upscale_factor': round(width / orig_width, 2)
            },
            'faces_detected': faces_detected
        }

    except Exception as e:
        import traceback
        print(f"[Pipeline] Unhandled error: {e}")
        print(traceback.format_exc())
        return {'success': False, 'error': f'Pipeline error: {str(e)}'}
