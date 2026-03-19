import os
import traceback
import uvicorn
import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

app = FastAPI(title="Enhancer Worker")

upsampler = None


def init_enhancer():
    global upsampler
    print("Loading Real-ESRGAN Model into memory...")

    model_name = "RealESRGAN_x2plus"
    model_url  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    weights_dir  = os.path.join(project_root, "experiments", "pretrained_models")
    os.makedirs(weights_dir, exist_ok=True)
    model_path   = os.path.join(weights_dir, f"{model_name}.pth")

    if not os.path.exists(model_path):
        model_path = load_file_from_url(
            url=model_url, model_dir=weights_dir, progress=True, file_name=None
        )

    model    = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=2)
    use_half = torch.cuda.is_available()
    upsampler = RealESRGANer(
        scale=2, model_path=model_path, model=model,
        tile=400, tile_pad=10, pre_pad=0, half=use_half,
    )
    print(f"Real-ESRGAN loaded. fp16={'ON' if use_half else 'OFF (CPU)'}")


# ============================================================================
# SHARPENING — kênh Y trong YCrCb, bảo toàn màu
# ============================================================================

def estimate_blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def adaptive_sharpen_strength(blur_score: float) -> float:
    """
    blur_score thấp → ảnh mờ → strength cao
    blur_score cao  → ảnh nét → strength thấp
    Range [0.4, 1.2] — cao hơn phiên bản cũ vì GFPGAN không còn upscale 2x nữa
    """
    t = (float(np.clip(blur_score, 10.0, 500.0)) - 10.0) / (500.0 - 10.0)
    return float(np.clip(1.2 - t * (1.2 - 0.4), 0.4, 1.2))


def sharpen_luminance_ycrcb(bgr: np.ndarray,
                             strength: float,
                             blur_sigma: float = 1.2,
                             detail_sigma: float = 0.4) -> np.ndarray:
    """Unsharp masking chỉ trên kênh Y — Cr/Cb (màu) giữ nguyên tuyệt đối."""
    assert bgr.dtype == np.uint8
    ycrcb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    Y_dn   = cv2.bilateralFilter(Y, d=5, sigmaColor=20, sigmaSpace=20)
    Y_f    = Y_dn.astype(np.float32)
    ksize  = (0, 0)
    Y_blur = cv2.GaussianBlur(Y_f, ksize, sigmaX=blur_sigma)
    dmask  = Y_f - Y_blur
    if detail_sigma > 0.0:
        dmask = cv2.GaussianBlur(dmask, ksize, sigmaX=detail_sigma)

    Y_sharp = np.clip(Y.astype(np.float32) + strength * dmask, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([Y_sharp, Cr, Cb]), cv2.COLOR_YCrCb2BGR)


def is_valid_output(img: np.ndarray, min_mean: float = 5.0) -> bool:
    return float(img.mean()) >= min_mean


# ============================================================================
# API
# ============================================================================

class EnhanceRequest(BaseModel):
    image_path: str
    output_path: str
    sharpen_strength: float | None = None


@app.on_event("startup")
def startup_event():
    init_enhancer()


@app.get("/health")
def health():
    return {"status": "ok", "model": "RealESRGAN_x2plus + YCrCb-Sharpen"}


@app.post("/enhance")
def enhance(request: EnhanceRequest):
    blur_score = 0.0
    strength   = 0.0
    try:
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail="Input image not found")

        print(f"Enhancing: {request.image_path}")
        input_img = cv2.imread(request.image_path, cv2.IMREAD_COLOR)
        if input_img is None:
            raise HTTPException(status_code=400, detail="Cannot read image")

        original_h, original_w = input_img.shape[:2]

        # ----------------------------------------------------------------
        # BƯỚC 1: Real-ESRGAN upscale 2x → downscale về kích thước gốc
        #
        # Tại sao upscale rồi downscale lại?
        #   ESRGAN ở 2x tái tạo chi tiết tần số cao (cạnh, texture) tốt hơn
        #   nhiều so với sharpen thẳng trên ảnh mờ. Sau đó LANCZOS4 downscale
        #   nén những chi tiết đó vào đúng kích thước ban đầu → ảnh nét nhưng
        #   không bị phóng to so với input.
        #
        # Kích thước output = kích thước input (không thay đổi size)
        # ----------------------------------------------------------------
        try:
            upscaled, _ = upsampler.enhance(input_img, outscale=2)
        except Exception as e:
            print(f"  WARNING: ESRGAN failed ({e}). Falling back to input.")
            upscaled = None

        if upscaled is None or not is_valid_output(upscaled):
            print("  WARNING: ESRGAN output invalid. Using original.")
            working = input_img.copy()
        else:
            working = cv2.resize(
                upscaled, (original_w, original_h),
                interpolation=cv2.INTER_LANCZOS4,
            )
            print(f"  ESRGAN done → downscaled to {original_w}×{original_h}")

        # ----------------------------------------------------------------
        # BƯỚC 2: Adaptive sharpening trên kênh Y (bảo toàn màu)
        # ----------------------------------------------------------------
        gray       = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        blur_score = estimate_blur_score(gray)
        print(f"  Blur score: {blur_score:.2f}")

        if request.sharpen_strength is not None:
            strength = float(np.clip(request.sharpen_strength, 0.0, 2.0))
        else:
            strength = adaptive_sharpen_strength(blur_score)
        print(f"  Sharpen strength: {strength:.3f}")

        output_img = sharpen_luminance_ycrcb(working, strength=strength) if strength > 0.01 else working

        if not is_valid_output(output_img):
            print("  WARNING: Output invalid. Using working image.")
            output_img = working

        # ----------------------------------------------------------------
        # BƯỚC 3: Lưu kết quả
        # ----------------------------------------------------------------
        out_dir = os.path.dirname(request.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(request.output_path, output_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Saved: {request.output_path} (mean={output_img.mean():.1f})")

        return {
            "success": True,
            "output_path": request.output_path,
            "blur_score": round(blur_score, 2),
            "sharpen_strength_used": round(strength, 3),
        }

    except Exception as e:
        print(f"Error in Enhancer worker: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)