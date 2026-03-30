import os
import sys
import uvicorn
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Colorization Worker")

# ============================================================================
# ML ARCHITECTURE (FROM infer_final.py)
# ============================================================================
class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        resnet = models.resnet18(weights=None)
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        return features

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        
        attention = F.softmax(torch.bmm(q, k) / (C ** 0.5), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class UNetDecoder(nn.Module):
    def __init__(self, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(512)
        
        self.up4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.dec4 = nn.Sequential(nn.Conv2d(256 + 256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dec3 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.up1 = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.final = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 2, 3, 1, 1), nn.Tanh())
        
    def forward(self, features):
        f0, f1, f2, f3, f4 = features
        if self.use_attention:
            f4 = self.attention(f4)
        
        x = self.up4(f4)
        x = self.dec4(torch.cat([x, f3], dim=1))
        
        x = self.up3(x)
        x = self.dec3(torch.cat([x, f2], dim=1))
        
        x = self.up2(x)
        x = self.dec2(torch.cat([x, f1], dim=1))
        
        x = self.up1(x)
        x = self.dec1(torch.cat([x, f0], dim=1))
        
        x = self.final(x)
        return x

class ColorizationModel(nn.Module):
    def __init__(self, pretrained=False, use_attention=True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        self.decoder = UNetDecoder(use_attention=use_attention)
        
    def forward(self, L):
        features = self.encoder(L)
        ab = self.decoder(features)
        return ab

# ============================================================================
# COLOR SPACE UTILITIES
# ============================================================================
def rgb_to_lab(rgb_tensor):
    rgb = rgb_tensor.clone()
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    x = x / 0.95047
    z = z / 1.08883
    
    epsilon = 0.008856
    kappa = 903.3
    
    fx = torch.where(x > epsilon, x.pow(1/3), (kappa * x + 16) / 116)
    fy = torch.where(y > epsilon, y.pow(1/3), (kappa * y + 16) / 116)
    fz = torch.where(z > epsilon, z.pow(1/3), (kappa * z + 16) / 116)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ch = 200 * (fy - fz)
    
    return torch.cat([L, a, b_ch], dim=1)

def lab_to_rgb(lab_tensor):
    L, a, b_ch = lab_tensor[:, 0:1], lab_tensor[:, 1:2], lab_tensor[:, 2:3]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b_ch / 200

    epsilon = 0.008856
    kappa = 903.3

    # Clamp fx^3, fz^3 to non-negative to avoid NaN from pow() on negatives
    fx3 = fx.pow(3).clamp(min=0)
    fz3 = fz.pow(3).clamp(min=0)

    x = torch.where(fx3 > epsilon, fx3, (116 * fx - 16) / kappa)
    y = torch.where(L > kappa * epsilon, ((L + 16) / 116).pow(3), L / kappa)
    z = torch.where(fz3 > epsilon, fz3, (116 * fz - 16) / kappa)

    # Clamp XYZ to non-negative (out-of-gamut values can be negative)
    x = x.clamp(min=0) * 0.95047
    y = y.clamp(min=0)
    z = z.clamp(min=0) * 1.08883

    r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314
    g = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 - y * 0.2040259 + z * 1.0572252

    rgb = torch.cat([r, g, b], dim=1)

    # CRITICAL: clamp rgb to non-negative before pow(1/2.4)
    # Negative values from out-of-gamut colors would produce NaN → all-black pixels
    rgb = rgb.clamp(min=0)

    mask = rgb > 0.0031308
    rgb_out = rgb.clone()
    rgb_out[mask] = 1.055 * rgb[mask].pow(1 / 2.4) - 0.055
    rgb_out[~mask] = 12.92 * rgb[~mask]

    return rgb_out.clamp(0, 1)

# ============================================================================
# FASTAPI WORKER SETUP
# ============================================================================
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_colorizer():
    global model
    print(f"Loading Colorization Model into memory on {device}...")
    
    # Locate user's model
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    model_path = os.path.join(project_root, 'colorization_model_final.pth')
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Colorization model not found at: {model_path}")
        
    model = ColorizationModel(pretrained=False, use_attention=True)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Colorization Model loaded successfully!")

class ColorizeRequest(BaseModel):
    image_path: str
    output_path: str

@app.on_event("startup")
def startup_event():
    init_colorizer()

@app.get("/health")
def health():
    return {"status": "ok", "model": "ColorizationModel"}

@app.post("/colorize")
def colorize(request: ColorizeRequest):
    try:
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail="Input image not found")
            
        print(f"Colorizing image: {request.image_path}")
        
        # Open and force to RGB first (strips alpha / palette),
        # then convert to grayscale RGB (to match training)
        img = Image.open(request.image_path).convert('RGB')
        img = img.convert('L').convert('RGB')
        original_size = img.size
        
        # Preprocess matching infer_final.py
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Convert to LAB format and normalize
        lab = rgb_to_lab(img_tensor)
        L = lab[:, 0:1] / 50.0 - 1.0
        
        # Inference
        with torch.no_grad():
            ab_pred = model(L)
            
        # Denormalize AB channels and merge with L
        L_denorm = (L + 1.0) * 50.0
        ab_pred_denorm = ab_pred * 128.0
        
        # Clamp to valid LAB ranges to prevent black output
        L_denorm = L_denorm.clamp(0, 100)
        ab_pred_denorm = ab_pred_denorm.clamp(-128, 127)
        
        lab_result = torch.cat([L_denorm, ab_pred_denorm], dim=1)
        rgb_result = lab_to_rgb(lab_result)
        
        # Return to full resolution
        result_tensor = rgb_result.squeeze(0).cpu()
        result_img = transforms.ToPILImage()(result_tensor)
        result_img = result_img.resize(original_size, Image.LANCZOS)
        
        # Save output
        os.makedirs(os.path.dirname(request.output_path), exist_ok=True)
        # Convert back to BGR for OpenCV saves later, or simply save with PIL
        result_img.save(request.output_path)
        
        return {
            "success": True, 
            "output_path": request.output_path
        }
    except Exception as e:
        import traceback
        print(f"Error in Colorization worker: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
