import os
import traceback
import torch
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gfpgan import GFPGANer

app = FastAPI(title="GFPGAN Worker")

restorer = None


def init_models():
    global restorer
    print("Loading GFPGAN Model into memory...")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    weights_dir  = os.path.join(project_root, 'experiments', 'pretrained_models')
    os.makedirs(weights_dir, exist_ok=True)

    gfpgan_path = os.path.join(weights_dir, 'GFPGANv1.4.pth')
    if not os.path.exists(gfpgan_path):
        raise RuntimeError(f"GFPGANv1.4.pth not found at: {gfpgan_path}")

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from basicsr.utils.download_util import load_file_from_url

    # Set up background upsampler
    model_name = "RealESRGAN_x2plus"
    model_url  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    bg_model_path = os.path.join(weights_dir, f"{model_name}.pth")
    if not os.path.exists(bg_model_path):
        load_file_from_url(url=model_url, model_dir=weights_dir, progress=True, file_name=None)
        
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2)
    # GTX 1650/1660 don't support fp16 properly → produces NaN → black output
    use_half = False
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        no_half_gpu_list = ['1650', '1660']
        if not any(g in gpu_name for g in no_half_gpu_list):
            use_half = True
        print(f"  GPU: {gpu_name}, half={use_half}")
    bg_upsampler = RealESRGANer(
        scale=2, model_path=bg_model_path, model=model,
        tile=400, tile_pad=40, pre_pad=0, half=use_half
    )

    restorer = GFPGANer(
        model_path=gfpgan_path,
        upscale=2,            # upscale 2x directly here
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=bg_upsampler,    # Upscale background too!
    )
    print("GFPGAN loaded. upscale=2, bg_upsampler=RealESRGAN (face+background restore).")


class EnhanceRequest(BaseModel):
    image_path: str
    output_path: str


@app.on_event("startup")
def startup_event():
    init_models()


@app.get("/health")
def health():
    return {"status": "ok", "model": "GFPGAN v1.4 (face-only, upscale=1)"}


@app.post("/enhance")
def enhance(request: EnhanceRequest):
    try:
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail="Input image not found")

        input_img = cv2.imread(request.image_path, cv2.IMREAD_UNCHANGED)
        if input_img is None:
            raise HTTPException(status_code=400, detail="Cannot read image")

        # Handle RGBA (4-channel) images → convert to BGR
        if input_img.ndim == 3 and input_img.shape[2] == 4:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGRA2BGR)
        elif input_img.ndim == 2:
            # Grayscale → BGR
            input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)

        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.7,
        )

        # ----------------------------------------------------------------
        # Validate: if bg_upsampler failed → black background
        # Two-pass fix: retry WITHOUT bg_upsampler to keep face restoration
        # ----------------------------------------------------------------
        use_img = input_img  # default fallback
        faces_detected = 0

        if restored_img is not None:
            mean_brightness = float(restored_img.mean())
            if mean_brightness < 5.0:
                print(f"  WARNING: bg_upsampler failed (mean={mean_brightness:.1f}). "
                      f"Retrying WITHOUT bg_upsampler...")
                # Temporarily disable bg_upsampler and set upscale=1
                old_bg = restorer.bg_upsampler
                old_upscale = restorer.upscale
                restorer.bg_upsampler = None
                restorer.upscale = 1
                try:
                    cropped_faces2, restored_faces2, restored_img2 = restorer.enhance(
                        input_img,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True,
                        weight=0.7,
                    )
                    if restored_img2 is not None and float(restored_img2.mean()) > 5.0:
                        use_img = restored_img2
                        faces_detected = len(cropped_faces2) if cropped_faces2 else 0
                        print(f"  Fallback OK. Faces: {faces_detected}")
                    else:
                        print("  Fallback also failed. Using original input.")
                except Exception as e2:
                    print(f"  Fallback enhance failed: {e2}. Using original input.")
                finally:
                    restorer.bg_upsampler = old_bg
                    restorer.upscale = old_upscale
            else:
                use_img        = restored_img
                faces_detected = len(cropped_faces) if cropped_faces else 0
                print(f"  GFPGAN done (with bg_upsampler). Faces: {faces_detected}, "
                      f"mean brightness: {mean_brightness:.1f}")
        else:
            print("  WARNING: GFPGAN returned None. Using input as fallback.")

        os.makedirs(os.path.dirname(request.output_path), exist_ok=True)
        cv2.imwrite(request.output_path, use_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return {
            "success": True,
            "output_path": request.output_path,
            "faces_detected": faces_detected,
        }

    except Exception as e:
        print(f"Error in GFPGAN worker: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
