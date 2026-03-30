#!/usr/bin/env python3
"""
Photo Restoration FastAPI Server
Expose restore_photo.py pipeline via REST API

Run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import services
from cloudinary_service import get_storage_mode, init_cloudinary, upload_restoration_results

# Initialize Cloudinary
init_cloudinary()

API_DIR = Path(__file__).resolve().parent
STATIC_DIR = API_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="Photo Restoration API",
    description="AI-powered old photo restoration using ZeroScratches + GFPGAN",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# CORS - Allow Spring Boot and ReactJS
spring_boot_url = os.getenv('SPRING_BOOT_URL', 'http://localhost:8080')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        spring_boot_url,
        "http://localhost:3000",  # ReactJS dev
        "http://localhost:5173",  # Vite dev
        "*"  # Remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class RestoreResponse(BaseModel):
    task_id: str
    success: bool
    original_url: Optional[str] = None
    restored_url: Optional[str] = None
    intermediate_url: Optional[str] = None
    original_size: Optional[dict] = None
    restored_size: Optional[dict] = None
    upscale_factor: Optional[float] = None
    faces_detected: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    message: str
    storage: Optional[str] = None



@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        message="Photo Restoration API is running",
        storage=get_storage_mode(),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is operational",
        storage=get_storage_mode(),
    )

async def _process_image_upload(task_id: str, file: UploadFile, pipeline_func, temp_dir: Path):
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Must be image/*"
            )
        
        # Save uploaded file
        original_path = temp_dir / "original.jpg"
        content = await file.read()
        with open(original_path, 'wb') as f:
            f.write(content)
        
        print(f"[API] Task {task_id}: Received {file.filename} ({len(content)} bytes)")
        
        # Run specific pipeline
        print(f"[API] Task {task_id}: Starting processing pipeline...")
        pipeline_result = pipeline_func(str(original_path))
        
        if not pipeline_result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {pipeline_result.get('error')}"
            )
            
        print(f"[API] Task {task_id}: Processing complete")
        
        print(f"[API] Task {task_id}: Publishing result via {get_storage_mode()} storage...")
        upload_result = upload_restoration_results(
            task_id=task_id,
            original_path=str(original_path),
            restored_path=pipeline_result['output_path'],
            intermediate_path=pipeline_result.get('intermediate_path')
        )
        
        if not upload_result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Upload failed: {upload_result.get('error')}"
            )
            
        print(f"[API] Task {task_id}: Result publish complete")
        
        # Build response
        response = RestoreResponse(
            task_id=task_id,
            success=True,
            original_url=upload_result['original']['url'],
            restored_url=upload_result['restored']['url'],
            faces_detected=pipeline_result.get('faces_detected', 0)
        )
        
        if 'intermediate' in upload_result:
            response.intermediate_url = upload_result['intermediate']['url']
            
        if 'image_size' in pipeline_result:
            sizes = pipeline_result['image_size']
            response.original_size = sizes.get('original')
            response.restored_size = sizes.get('restored')
            response.upscale_factor = sizes.get('upscale_factor')
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Task {task_id}: Error - {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/restore", response_model=RestoreResponse)
async def restore_photo(file: UploadFile = File(...)):
    """API Endpoint 1: Repair Scratches and Faces"""
    task_id = str(uuid.uuid4())
    temp_dir = Path(f"temp/{task_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        from restoration_service import run_restoration_pipeline
        return await _process_image_upload(task_id, file, run_restoration_pipeline, temp_dir)
    finally:
        _cleanup_temp_files(temp_dir)

@app.post("/api/colorize", response_model=RestoreResponse)
async def colorize_photo(file: UploadFile = File(...)):
    """API Endpoint 2: Add Color to B&W Photos"""
    task_id = str(uuid.uuid4())
    temp_dir = Path(f"temp/{task_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        from restoration_service import run_colorization_pipeline
        return await _process_image_upload(task_id, file, run_colorization_pipeline, temp_dir)
    finally:
        _cleanup_temp_files(temp_dir)

@app.post("/api/restore-and-colorize", response_model=RestoreResponse)
async def restore_and_colorize_photo(file: UploadFile = File(...)):
    """API Endpoint 3: Complete Restoration + Colorization"""
    task_id = str(uuid.uuid4())
    temp_dir = Path(f"temp/{task_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        from restoration_service import run_full_pipeline
        return await _process_image_upload(task_id, file, run_full_pipeline, temp_dir)
    finally:
        _cleanup_temp_files(temp_dir)




def _cleanup_temp_files(temp_dir: Path):
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    outputs_dir = Path('outputs')
    if outputs_dir.exists():
        for file in outputs_dir.glob('*'):
            if file.is_file():
                try: file.unlink()
                except: pass



if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    
    print(f"Starting Photo Restoration API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
