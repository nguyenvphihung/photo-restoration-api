import os
import uvicorn
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from zeroscratches import EraseScratches

app = FastAPI(title="ZeroScratches Worker")

# Initialize model globally to keep in memory
print("Loading ZeroScratches Model into memory...")
eraser = EraseScratches()
print("ZeroScratches Model loaded!")

class ProcessRequest(BaseModel):
    image_path: str
    output_path: str

@app.get("/health")
def health():
    return {"status": "ok", "model": "ZeroScratches"}

@app.post("/process")
def process(request: ProcessRequest):
    try:
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail="Input image not found")
        
        original_image = cv2.imread(request.image_path)
        if original_image is None:
            raise HTTPException(status_code=400, detail="Cannot read image")
            
        image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        # Inference
        result = eraser.erase(image_pil)
        
        if result is not None:
            scratch_removed = cv2.cvtColor(np.array(Image.fromarray(result)), cv2.COLOR_RGB2BGR)
            os.makedirs(os.path.dirname(request.output_path), exist_ok=True)
            cv2.imwrite(request.output_path, scratch_removed)
            return {"success": True, "output_path": request.output_path}
        else:
            # Fallback to copy
            os.makedirs(os.path.dirname(request.output_path), exist_ok=True)
            cv2.imwrite(request.output_path, original_image)
            return {"success": False, "message": "Failed to remove scratches, copied original", "output_path": request.output_path}
            
    except Exception as e:
        print(f"Error in ZeroScratches worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
