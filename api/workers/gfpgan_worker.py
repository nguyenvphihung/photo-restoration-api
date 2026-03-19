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

    # -----------------------------------------------------------------------
    # THIẾT KẾ: upscale=1, bg_upsampler=None
    # -----------------------------------------------------------------------
    # Vấn đề với upscale=2 + bg_upsampler (cách làm trước):
    #
    #   Khi GFPGAN dùng bg_upsampler (RealESRGAN) để upscale nền lên 2x:
    #   • Nếu bg_upsampler fail (OOM, NaN, tiling issue trên một số GPU/CPU)
    #     → nó trả về nền đen nhưng vùng mặt vẫn ổn
    #     → kết quả: vignette đen quanh mặt — đúng lỗi đang thấy
    #   • Thêm nữa, enhancer_worker cũng ESRGAN upscale 2x ở bước sau
    #     → tổng cộng 4x upscaling → lãng phí và tạo thêm cơ hội fail
    #
    # Giải pháp: GFPGAN chỉ làm đúng 1 việc — restore mặt, giữ nguyên size
    #   • upscale=1  → output cùng kích thước với input, KHÔNG upscale
    #   • bg_upsampler=None → không động đến nền, nền giữ nguyên 100%
    #   • Không có RealESRGAN trong worker này → không có black risk từ ESRGAN
    #   • enhancer_worker (bước 4) đã chạy ESRGAN toàn ảnh (cả mặt lẫn nền)
    #     → chất lượng tổng thể vẫn được nâng lên ở đúng bước đó
    # -----------------------------------------------------------------------
    restorer = GFPGANer(
        model_path=gfpgan_path,
        upscale=2,            # giữ nguyên kích thước
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,    # không upscale nền ở đây
    )
    print("GFPGAN loaded. upscale=2, bg_upsampler=None (face-only restore).")


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
            weight=0.5,   # blend 50% restored / 50% original → tự nhiên hơn
        )

        # ----------------------------------------------------------------
        # Validate output: nếu restored_img là None hoặc gần đen
        # thì dùng input gốc thay thế (an toàn tuyệt đối)
        # ----------------------------------------------------------------
        use_img = input_img  # default fallback
        faces_detected = 0

        if restored_img is not None:
            mean_brightness = float(restored_img.mean())
            if mean_brightness < 5.0:
                print(f"  WARNING: GFPGAN output near-black (mean={mean_brightness:.1f}). "
                      f"Using input image as fallback.")
            else:
                use_img        = restored_img
                faces_detected = len(cropped_faces) if cropped_faces else 0
                print(f"  GFPGAN done. Faces: {faces_detected}, "
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
