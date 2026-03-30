import cv2
import httpx
import os
from pathlib import Path
from typing import Any, Dict

# Worker endpoints
ZEROSCRATCHES_URL = os.getenv("ZEROSCRATCHES_URL", "http://127.0.0.1:8001/process")
COLORIZATION_URL = os.getenv("COLORIZATION_URL", "http://127.0.0.1:8002/colorize")
GFPGAN_URL = os.getenv("GFPGAN_URL", "http://127.0.0.1:8003/enhance")
CODEFORMER_URL = os.getenv("CODEFORMER_URL", "http://127.0.0.1:8004/enhance")
PIPELINE_HTTP_TIMEOUT_SECONDS = float(os.getenv("PIPELINE_HTTP_TIMEOUT_SECONDS", "60"))

# Max input resolution (to prevent OOM on 4GB GPU)
MAX_INPUT_WIDTH = 800
MAX_INPUT_HEIGHT = 800


def resize_if_needed(
    image_path: str,
    max_width: int = MAX_INPUT_WIDTH,
    max_height: int = MAX_INPUT_HEIGHT,
) -> tuple:
    """
    Resize image if it exceeds max dimensions to prevent GPU OOM.
    Returns: (resized_path, was_resized, original_size)
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return image_path, False, None

    # Normalize input to 3-channel BGR for all downstream workers.
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
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
    print(f"[Pipeline] Resizing: {w}x{h} -> {new_w}x{new_h} (scale {scale:.2f})")

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    resized_path = str(Path(image_path).parent / f"_resized_{Path(image_path).name}")
    cv2.imwrite(resized_path, resized)
    return resized_path, True, original_size


def run_restoration_pipeline(input_image_path: str, **_) -> Dict[str, Any]:
    """Repair scratches and restore faces without colorization."""
    return _execute_pipeline(input_image_path, run_restore=True, run_color=False)


def run_colorization_pipeline(input_image_path: str, **_) -> Dict[str, Any]:
    """Colorize only, without scratch removal or face restoration."""
    return _execute_pipeline(input_image_path, run_restore=False, run_color=True)


def run_full_pipeline(input_image_path: str, **_) -> Dict[str, Any]:
    """Run the production pipeline: restore, colorize, then face restore."""
    return _execute_pipeline(input_image_path, run_restore=True, run_color=True)


def estimate_max_face_ratio(image_path: str) -> float:
    """
    Return the area ratio of the largest detected face.
    Returns 0.0 if no face is detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        return 0.0

    img_area = img.shape[0] * img.shape[1]
    max_face_area = max(w * h for (_, _, w, h) in faces)
    return max_face_area / img_area


def _execute_pipeline(
    input_image_path: str,
    run_restore: bool,
    run_color: bool,
) -> Dict[str, Any]:
    """
    Pipeline order:
      1. ZeroScratches
      2. Colorization
      3. Adaptive face restore with GFPGAN or CodeFormer

    Colorization must run before face restoration because the colorization worker
    internally converts the input to grayscale.
    """
    try:
        input_path = Path(input_image_path)
        if not input_path.exists():
            return {"success": False, "error": f"Input file not found: {input_image_path}"}

        project_root = Path(__file__).parent.parent.absolute()
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        print(f"[Pipeline] Processing: {input_image_path}")
        process_path, was_resized, original_size = resize_if_needed(str(input_path))

        step1_path = str(outputs_dir / "step1_zeroscratches.jpg")
        step2_path = str(outputs_dir / "step2_colorized.jpg")
        final_path = str(outputs_dir / "final_output.jpg")
        intermediate_path = None
        faces_detected = 0
        current_img_path = str(Path(process_path).absolute())

        with httpx.Client(timeout=PIPELINE_HTTP_TIMEOUT_SECONDS) as client:
            if run_restore:
                print("[Pipeline] Step 1/3: ZeroScratches (scratch removal)...")
                try:
                    resp = client.post(
                        ZEROSCRATCHES_URL,
                        json={
                            "image_path": current_img_path,
                            "output_path": step1_path,
                        },
                    )
                    resp.raise_for_status()
                    if not resp.json().get("success"):
                        return {"success": False, "error": "ZeroScratches worker failed"}
                    current_img_path = step1_path
                    intermediate_path = step1_path
                    print("[Pipeline] Step 1 done.")
                except Exception as exc:
                    return {"success": False, "error": f"ZeroScratches failed: {exc}"}

            if run_color:
                out_path = step2_path if run_restore else final_path
                print("[Pipeline] Step 2/3: Colorization...")
                try:
                    resp = client.post(
                        COLORIZATION_URL,
                        json={
                            "image_path": current_img_path,
                            "output_path": out_path,
                        },
                    )
                    resp.raise_for_status()
                    if not resp.json().get("success"):
                        return {"success": False, "error": "Colorization worker failed"}
                    current_img_path = out_path
                    if not intermediate_path:
                        intermediate_path = out_path
                    print("[Pipeline] Step 2 done.")
                except Exception as exc:
                    return {"success": False, "error": f"Colorization failed: {exc}"}

            if run_restore:
                face_ratio = estimate_max_face_ratio(current_img_path)
                if face_ratio >= 0.05:
                    target_url = GFPGAN_URL
                    model_name = "GFPGAN"
                else:
                    target_url = CODEFORMER_URL
                    model_name = "CodeFormer"

                print("[Pipeline] Step 3/3: Face & Background Restoration...")
                print(f"  Max face ratio: {face_ratio * 100:.1f}% -> Routing to {model_name}")

                try:
                    resp = client.post(
                        target_url,
                        json={
                            "image_path": current_img_path,
                            "output_path": final_path,
                        },
                    )
                    resp.raise_for_status()
                    res = resp.json()
                    if not res.get("success"):
                        return {"success": False, "error": f"{model_name} worker failed"}

                    faces_detected = res.get("faces_detected", 0)
                    current_img_path = final_path
                    if not intermediate_path:
                        intermediate_path = final_path
                    print(f"[Pipeline] Step 3 done. Faces detected: {faces_detected} by {model_name}")
                except Exception as exc:
                    return {"success": False, "error": f"{model_name} failed: {exc}"}

        final_image = cv2.imread(current_img_path)
        if final_image is None:
            return {"success": False, "error": "Failed to read final output"}

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
            "success": True,
            "output_path": current_img_path,
            "intermediate_path": intermediate_path,
            "image_size": {
                "original": {"width": orig_width, "height": orig_height},
                "restored": {"width": width, "height": height},
                "upscale_factor": round(width / orig_width, 2),
            },
            "faces_detected": faces_detected,
        }

    except Exception as e:
        import traceback

        print(f"[Pipeline] Unhandled error: {e}")
        print(traceback.format_exc())
        return {"success": False, "error": f"Pipeline error: {str(e)}"}
