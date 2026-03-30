import os
import sys

# CRITICAL: sys.path manipulation must happen BEFORE importing basicsr/facelib 
# Otherwise, Python will load the pip-installed basicsr into sys.modules.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
codeformer_dir = os.path.join(project_root, 'api', 'CodeFormer')
sys.path.insert(0, codeformer_dir)

import traceback
import torch
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from basicsr.utils.download_util import load_file_from_url
from torchvision.transforms.functional import normalize

from basicsr.utils import imwrite, img2tensor, tensor2img
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY

app = FastAPI(title="CodeFormer Worker")

restorer = None
face_helper = None
bg_upsampler = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALLOW_MODEL_DOWNLOADS = os.getenv("ALLOW_MODEL_DOWNLOADS", "false").lower() in {"1", "true", "yes", "on"}

def init_models():
    global restorer, face_helper, bg_upsampler
    print(f"Loading CodeFormer Model into memory on {device}...")

    # Load architecture from CodeFormer/basicsr/archs
    from basicsr.archs import codeformer_arch  # This registers the CodeFormer architecture

    weights_dir = os.path.join(project_root, 'experiments', 'pretrained_models')
    os.makedirs(weights_dir, exist_ok=True)

    model_name = 'codeformer-v0.1.0'
    model_url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    model_path = os.path.join(weights_dir, f"{model_name}.pth")

    if not os.path.exists(model_path):
        if not ALLOW_MODEL_DOWNLOADS:
            raise RuntimeError(
                f"CodeFormer weights not found at: {model_path}. "
                "Offline mode is enabled; mount pretrained models before startup."
            )
        print(f"Downloading CodeFormer weights to {model_path}...")
        load_file_from_url(url=model_url, model_dir=weights_dir, progress=True, file_name=f"{model_name}.pth")

    net = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
        connect_list=['32', '64', '128', '256']
    ).to(device)
    
    ckpt = torch.load(model_path, map_location='cpu')
    net.load_state_dict(ckpt['params_ema'])
    net.eval()
    
    restorer = net
    
    # Initialize FaceRestoreHelper (from facexlib/facelib)
    face_helper = FaceRestoreHelper(
        upscale_factor=2,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device
    )
    
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    bg_model_name = "RealESRGAN_x2plus"
    bg_model_url  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    bg_model_path = os.path.join(weights_dir, f"{bg_model_name}.pth")

    if not os.path.exists(bg_model_path):
        if not ALLOW_MODEL_DOWNLOADS:
            raise RuntimeError(
                f"RealESRGAN_x2plus.pth not found at: {bg_model_path}. "
                "Offline mode is enabled; mount pretrained models before startup."
            )
        load_file_from_url(url=bg_model_url, model_dir=weights_dir, progress=True, file_name=None)

    bg_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    # GTX 1650/1660 don't support fp16 properly → produces NaN → black output
    use_half = False
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        no_half_gpu_list = ['1650', '1660']
        if not any(g in gpu_name for g in no_half_gpu_list):
            use_half = True
        print(f"  GPU: {gpu_name}, half={use_half}")
    bg_upsampler = RealESRGANer(
        scale=2, model_path=bg_model_path, model=bg_model,
        tile=400, tile_pad=40, pre_pad=0, half=use_half
    )
    
    print("CodeFormer loaded successfully (fidelity=0.7, upscale=2, bg_upsampler=RealESRGAN).")


class EnhanceRequest(BaseModel):
    image_path: str
    output_path: str


@app.on_event("startup")
def startup_event():
    init_models()


@app.get("/health")
def health():
    return {"status": "ok", "model": "CodeFormer (adaptive face restore)"}


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
            input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)

        img = input_img
        
        face_helper.clean_all()
        face_helper.read_image(img)
        
        # Get face landmarks for each face
        num_faces = face_helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5
        )
        
        print(f"  CodeFormer detected {num_faces} faces")
        
        # Align and warp each face
        face_helper.align_warp_face()

        # Face restoration for each cropped face
        for i, cropped_face in enumerate(face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    # w=0.7 is the fidelity weight (balance between quality and fidelity)
                    output_t = restorer(cropped_face_t, w=0.7, adain=True)[0]
                    restored_face = tensor2img(output_t, rgb2bgr=True, min_max=(-1, 1))
                del output_t
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  WARNING: CodeFormer inference failed for face {i}: {e}. Skipping this face.")
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face)

        if num_faces > 0:
            # Upsample the background before pasting faces back
            bg_img = None
            if bg_upsampler is not None:
                try:
                    bg_img = bg_upsampler.enhance(img, outscale=2)[0]
                except Exception as e:
                    print(f"  WARNING: CodeFormer bg_upsampler failed: {e}")

            face_helper.get_inverse_affine(None)
            # Paste faces back into the original image
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img)
        else:
            print("  Falling back to original image because no face detected.")
            restored_img = input_img
            
        use_img = restored_img
        
        # Validation as GFPGAN does
        mean_brightness = float(use_img.mean())
        if mean_brightness < 5.0:
            print(f"  WARNING: CodeFormer output near-black (mean={mean_brightness:.1f}). Using input image as fallback.")
            use_img = input_img

        os.makedirs(os.path.dirname(request.output_path), exist_ok=True)
        cv2.imwrite(request.output_path, use_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return {
            "success": True,
            "output_path": request.output_path,
            "faces_detected": num_faces,
        }

    except Exception as e:
        print(f"Error in CodeFormer worker: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
