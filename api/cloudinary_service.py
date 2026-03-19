"""
Cloudinary Service
Handles image uploads to Cloudinary
"""

import cloudinary
import cloudinary.uploader
from typing import Dict, Any, Optional
import os
from pathlib import Path


def init_cloudinary():
    """Initialize Cloudinary with credentials from environment"""
    cloudinary.config(
        cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
        api_key=os.getenv('CLOUDINARY_API_KEY'),
        api_secret=os.getenv('CLOUDINARY_API_SECRET')
    )


def upload_image(
    image_path: str,
    folder: str = 'restored_photos',
    public_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upload image to Cloudinary
    
    Args:
        image_path: Local path to image
        folder: Cloudinary folder name
        public_id: Optional custom public ID
    
    Returns:
        dict with success, url, secure_url, public_id
    """
    try:
        if not Path(image_path).exists():
            return {
                'success': False,
                'error': f'Image not found: {image_path}'
            }
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            image_path,
            folder=folder,
            public_id=public_id,
            overwrite=True,
            resource_type='image',
            quality='auto:best',
            fetch_format='auto'
        )
        
        return {
            'success': True,
            'url': upload_result.get('url'),
            'secure_url': upload_result.get('secure_url'),
            'public_id': upload_result.get('public_id'),
            'width': upload_result.get('width'),
            'height': upload_result.get('height'),
            'format': upload_result.get('format'),
            'bytes': upload_result.get('bytes')
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Cloudinary upload failed: {str(e)}'
        }


def upload_restoration_results(
    task_id: str,
    original_path: str,
    restored_path: str,
    intermediate_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upload all restoration results to Cloudinary
    
    Args:
        task_id: Unique task ID
        original_path: Path to original image
        restored_path: Path to restored image
        intermediate_path: Optional path to intermediate result
    
    Returns:
        dict with URLs for all uploaded images
    """
    try:
        results = {}
        
        # Upload original
        print(f"[Cloudinary] Uploading original...")
        original_result = upload_image(
            original_path,
            folder=f'restored_photos/{task_id}',
            public_id='original'
        )
        
        if not original_result['success']:
            return {
                'success': False,
                'error': f"Original upload failed: {original_result.get('error')}"
            }
        
        results['original'] = {
            'url': original_result['secure_url'],
            'width': original_result.get('width'),
            'height': original_result.get('height')
        }
        
        # Upload restored
        print(f"[Cloudinary] Uploading restored...")
        restored_result = upload_image(
            restored_path,
            folder=f'restored_photos/{task_id}',
            public_id='restored'
        )
        
        if not restored_result['success']:
            return {
                'success': False,
                'error': f"Restored upload failed: {restored_result.get('error')}"
            }
        
        results['restored'] = {
            'url': restored_result['secure_url'],
            'width': restored_result.get('width'),
            'height': restored_result.get('height')
        }
        
        # Upload intermediate if provided
        if intermediate_path and Path(intermediate_path).exists():
            print(f"[Cloudinary] Uploading intermediate...")
            intermediate_result = upload_image(
                intermediate_path,
                folder=f'restored_photos/{task_id}',
                public_id='intermediate'
            )
            
            if intermediate_result['success']:
                results['intermediate'] = {
                    'url': intermediate_result['secure_url'],
                    'width': intermediate_result.get('width'),
                    'height': intermediate_result.get('height')
                }
        
        return {
            'success': True,
            'task_id': task_id,
            **results
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Upload batch failed: {str(e)}'
        }
