"""
Storage service with Cloudinary upload and local static fallback.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import shutil

import cloudinary
import cloudinary.uploader
from PIL import Image


PLACEHOLDER_VALUES = {
    "",
    "your_cloud_name",
    "your_cloud_name_here",
    "your_api_key",
    "your_api_key_here",
    "your_api_secret",
    "your_api_secret_here",
}


def _normalized_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    return "" if value in PLACEHOLDER_VALUES else value


def is_cloudinary_configured() -> bool:
    return all(
        [
            _normalized_env("CLOUDINARY_CLOUD_NAME"),
            _normalized_env("CLOUDINARY_API_KEY"),
            _normalized_env("CLOUDINARY_API_SECRET"),
        ]
    )


def get_storage_mode() -> str:
    return "cloudinary" if is_cloudinary_configured() else "local"


def init_cloudinary() -> bool:
    """Initialize Cloudinary when credentials are present."""
    if not is_cloudinary_configured():
        print("[Storage] Cloudinary config missing or placeholder values detected. Using local fallback.")
        return False

    cloudinary.config(
        cloud_name=_normalized_env("CLOUDINARY_CLOUD_NAME"),
        api_key=_normalized_env("CLOUDINARY_API_KEY"),
        api_secret=_normalized_env("CLOUDINARY_API_SECRET"),
    )
    print("[Storage] Cloudinary initialized.")
    return True


def _public_base_url() -> str:
    return os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _local_results_dir() -> Path:
    api_dir = Path(__file__).resolve().parent
    relative_dir = os.getenv("LOCAL_RESULTS_DIR", "static/results")
    target_dir = (api_dir / relative_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _image_metadata(image_path: str) -> Dict[str, Any]:
    image_file = Path(image_path)
    width = None
    height = None
    image_format = image_file.suffix.lstrip(".").lower() or None

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image_format = (image.format or image_format or "").lower() or None
    except Exception:
        pass

    return {
        "width": width,
        "height": height,
        "format": image_format,
        "bytes": image_file.stat().st_size,
    }


def upload_image(
    image_path: str,
    folder: str = "restored_photos",
    public_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload a single image to Cloudinary."""
    try:
        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        upload_result = cloudinary.uploader.upload(
            image_path,
            folder=folder,
            public_id=public_id,
            overwrite=True,
            resource_type="image",
            quality="auto:best",
            fetch_format="auto",
        )

        return {
            "success": True,
            "url": upload_result.get("url"),
            "secure_url": upload_result.get("secure_url"),
            "public_id": upload_result.get("public_id"),
            "width": upload_result.get("width"),
            "height": upload_result.get("height"),
            "format": upload_result.get("format"),
            "bytes": upload_result.get("bytes"),
        }
    except Exception as e:
        return {"success": False, "error": f"Cloudinary upload failed: {str(e)}"}


def _store_local_image(task_id: str, image_path: str, name: str) -> Dict[str, Any]:
    if not Path(image_path).exists():
        return {"success": False, "error": f"Image not found: {image_path}"}

    task_dir = _local_results_dir() / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    source = Path(image_path)
    extension = source.suffix.lower() or ".jpg"
    target = task_dir / f"{name}{extension}"
    shutil.copy2(source, target)

    metadata = _image_metadata(str(target))
    return {
        "success": True,
        "url": f"{_public_base_url()}/static/results/{task_id}/{target.name}",
        "width": metadata["width"],
        "height": metadata["height"],
        "format": metadata["format"],
        "bytes": metadata["bytes"],
    }


def _upload_restoration_results_to_cloudinary(
    task_id: str,
    original_path: str,
    restored_path: str,
    intermediate_path: Optional[str] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    print("[Cloudinary] Uploading original...")
    original_result = upload_image(
        original_path,
        folder=f"restored_photos/{task_id}",
        public_id="original",
    )
    if not original_result["success"]:
        return {"success": False, "error": f"Original upload failed: {original_result.get('error')}"}

    results["original"] = {
        "url": original_result["secure_url"],
        "width": original_result.get("width"),
        "height": original_result.get("height"),
    }

    print("[Cloudinary] Uploading restored...")
    restored_result = upload_image(
        restored_path,
        folder=f"restored_photos/{task_id}",
        public_id="restored",
    )
    if not restored_result["success"]:
        return {"success": False, "error": f"Restored upload failed: {restored_result.get('error')}"}

    results["restored"] = {
        "url": restored_result["secure_url"],
        "width": restored_result.get("width"),
        "height": restored_result.get("height"),
    }

    if intermediate_path and Path(intermediate_path).exists():
        print("[Cloudinary] Uploading intermediate...")
        intermediate_result = upload_image(
            intermediate_path,
            folder=f"restored_photos/{task_id}",
            public_id="intermediate",
        )
        if intermediate_result["success"]:
            results["intermediate"] = {
                "url": intermediate_result["secure_url"],
                "width": intermediate_result.get("width"),
                "height": intermediate_result.get("height"),
            }

    return {"success": True, "task_id": task_id, "storage": "cloudinary", **results}


def _store_restoration_results_locally(
    task_id: str,
    original_path: str,
    restored_path: str,
    intermediate_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("[Storage] Saving results locally...")

    original_result = _store_local_image(task_id, original_path, "original")
    if not original_result["success"]:
        return {"success": False, "error": original_result["error"]}

    restored_result = _store_local_image(task_id, restored_path, "restored")
    if not restored_result["success"]:
        return {"success": False, "error": restored_result["error"]}

    results: Dict[str, Any] = {
        "success": True,
        "task_id": task_id,
        "storage": "local",
        "original": {
            "url": original_result["url"],
            "width": original_result.get("width"),
            "height": original_result.get("height"),
        },
        "restored": {
            "url": restored_result["url"],
            "width": restored_result.get("width"),
            "height": restored_result.get("height"),
        },
    }

    if intermediate_path and Path(intermediate_path).exists():
        intermediate_result = _store_local_image(task_id, intermediate_path, "intermediate")
        if intermediate_result["success"]:
            results["intermediate"] = {
                "url": intermediate_result["url"],
                "width": intermediate_result.get("width"),
                "height": intermediate_result.get("height"),
            }

    return results


def upload_restoration_results(
    task_id: str,
    original_path: str,
    restored_path: str,
    intermediate_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload restoration results to Cloudinary when available.
    Fall back to local static files if Cloudinary is disabled or upload fails.
    """
    try:
        if is_cloudinary_configured():
            cloud_result = _upload_restoration_results_to_cloudinary(
                task_id=task_id,
                original_path=original_path,
                restored_path=restored_path,
                intermediate_path=intermediate_path,
            )
            if cloud_result["success"]:
                return cloud_result
            print(f"[Storage] Cloudinary failed. Falling back to local storage. Reason: {cloud_result.get('error')}")
        else:
            print("[Storage] Cloudinary disabled. Using local static fallback.")

        return _store_restoration_results_locally(
            task_id=task_id,
            original_path=original_path,
            restored_path=restored_path,
            intermediate_path=intermediate_path,
        )
    except Exception as e:
        return {"success": False, "error": f"Storage upload failed: {str(e)}"}
