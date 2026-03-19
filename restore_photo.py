#!/usr/bin/env python3
"""
Simplified Photo Restoration Pipeline
ZeroScratches -> GFPGAN (No Denoise)

Usage: python restore_photo.py [image_path]
Example: python restore_photo.py img1.jpg
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import subprocess
import matplotlib.pyplot as plt

def print_header():
    """Print pipeline header"""
    print("Simplified Photo Restoration Pipeline")
    print("=" * 60)
    print("Pipeline: ZeroScratches -> GFPGAN (No Denoise)")
    print("[+] Preserves original details")
    print("[+] Fast processing")
    print("[+] High-quality results")
    print("=" * 60)

def check_environments():
    """Check if required environments exist"""
    try:
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        envs = result.stdout
        
        has_rs = 'rs-clean' in envs
        has_gfpgan = 'gfpgan-clean' in envs
        
        if not has_rs or not has_gfpgan:
            print("[-] Missing required environments!")
            print("Please run setup first:")
            if not has_rs:
                print("  conda create -n rs-clean python=3.10 -y")
                print("  conda activate rs-clean")
                print("  pip install pillow opencv-python numpy matplotlib zeroscratches")
                print("  pip install 'numpy==1.26.4' --force-reinstall")
            if not has_gfpgan:
                print("  conda create -n gfpgan-clean python=3.9 -y") 
                print("  conda activate gfpgan-clean")
                print("  pip install torch==1.13.1 torchvision==0.14.1 opencv-python pillow numpy==1.24.3 basicsr==1.4.2 facexlib gfpgan")
            return False
        
        return True
    except Exception as e:
        print(f"[-] Error checking environments: {e}")
        return False

def run_zeroscratches(image_path):
    """Step 1: Run ZeroScratches scratch removal"""
    print("\n[*] Step 1: ZeroScratches (Scratch Removal)")
    
    temp_script = """
import cv2
import numpy as np
from PIL import Image
from zeroscratches import EraseScratches
import os
import sys

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    image_path = sys.argv[1]
    print(f'Processing: {image_path}')
    
    # Load image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print('Error: Could not load image')
        sys.exit(1)
    
    # Convert to PIL
    image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    
    # Remove scratches
    eraser = EraseScratches()
    result = eraser.erase(image_pil)
    
    if result is not None:
        # Convert back to OpenCV format
        scratch_removed = cv2.cvtColor(np.array(Image.fromarray(result)), cv2.COLOR_RGB2BGR)
        
        # Save result
        os.makedirs('outputs', exist_ok=True)
        output_path = 'outputs/step1_zeroscratches.jpg'
        cv2.imwrite(output_path, scratch_removed)
        
        print(f'ZeroScratches completed successfully')
        print(f'Saved: {output_path}')
        print(f'Size: {scratch_removed.shape[1]}x{scratch_removed.shape[0]}')
    else:
        print('ZeroScratches failed - copying original')
        os.makedirs('outputs', exist_ok=True)
        cv2.imwrite('outputs/step1_zeroscratches.jpg', original_image)
        
except Exception as e:
    print(f'Error in ZeroScratches: {e}')
    sys.exit(1)
"""
    
    # Write temp script
    with open('temp_zeroscratches.py', 'w', encoding='utf-8') as f:
        f.write(temp_script)
    
    try:
        # Run in rs-clean environment using powershell command chaining
        cmd = f'powershell -Command "conda activate rs-clean; python temp_zeroscratches.py \'{image_path}\'"'
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        
        # Check if output exists
        if os.path.exists('outputs/step1_zeroscratches.jpg'):
            print("[+] ZeroScratches completed successfully!")
            return True
        else:
            print("[-] ZeroScratches failed - no output file")
            return False
            
    except subprocess.TimeoutExpired:
        print("[-] ZeroScratches timed out")
        return False
    except Exception as e:
        print(f"[-] Error running ZeroScratches: {e}")
        return False
    finally:
        # Cleanup temp script
        if os.path.exists('temp_zeroscratches.py'):
            os.remove('temp_zeroscratches.py')

def run_gfpgan():
    """Step 2: Run GFPGAN face restoration"""
    print("\n[*] Step 2: GFPGAN (Face Restoration)")
    
    temp_script = """
import cv2
from gfpgan import GFPGANer
from basicsr.utils.download_util import load_file_from_url
import os

try:
    print('Initializing GFPGAN...')
    
    # Setup model path
    model_path = 'experiments/pretrained_models/GFPGANv1.4.pth'
    if not os.path.exists(model_path):
        print('Downloading GFPGAN model...')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        load_file_from_url(url=model_url, model_dir='experiments/pretrained_models', progress=True, file_name=None)
    
    # Initialize GFPGAN
    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch='clean', 
        channel_multiplier=2,
        bg_upsampler=None
    )
    
    print('GFPGAN initialized successfully')
    
    # Process ZeroScratches result
    input_path = 'outputs/step1_zeroscratches.jpg'
    if not os.path.exists(input_path):
        print(f'Error: Input file not found: {input_path}')
        exit(1)
    
    print(f'Processing: {input_path}')
    input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    
    # Enhance with GFPGAN
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=False,
        only_center_face=False, 
        paste_back=True
    )
    
    # Save final result
    output_path = 'outputs/final_restored.jpg'
    cv2.imwrite(output_path, restored_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print(f'GFPGAN completed successfully')
    print(f'Found {len(cropped_faces)} face(s)')
    print(f'Final result: {output_path}')
    print(f'Size: {restored_img.shape[1]}x{restored_img.shape[0]}')
    
except Exception as e:
    print(f'Error in GFPGAN: {e}')
    exit(1)
"""
    
    # Write temp script  
    with open('temp_gfpgan.py', 'w', encoding='utf-8') as f:
        f.write(temp_script)
    
    try:
        # Run in gfpgan-clean environment
        cmd = 'powershell -Command "conda activate gfpgan-clean; python temp_gfpgan.py"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        
        # Check if output exists
        if os.path.exists('outputs/final_restored.jpg'):
            print("[+] GFPGAN completed successfully!")
            return True
        else:
            print("[-] GFPGAN failed - no output file")
            return False 
            
    except subprocess.TimeoutExpired:
        print("[-] GFPGAN timed out")
        return False
    except Exception as e:
        print(f"[-] Error running GFPGAN: {e}")
        return False
    finally:
        # Cleanup temp script
        if os.path.exists('temp_gfpgan.py'):
            os.remove('temp_gfpgan.py')

def show_results(original_path, display_gui=True):
    """Display comparison results"""
    print("\n[*] Results Comparison")
    
    try:
        # Load images
        original = cv2.imread(original_path)
        final = cv2.imread('outputs/final_restored.jpg')
        
        if original is None or final is None:
            print("[-] Could not load images for comparison")
            return
        
        # Print comparison info
        print(f"\n[*] Comparison Results:")
        print(f"  Original: {original.shape[1]}x{original.shape[0]} pixels")
        print(f"  Restored: {final.shape[1]}x{final.shape[0]} pixels")
        print(f"  Upscale factor: {final.shape[1]/original.shape[1]:.1f}x")
        
        # Only show GUI if requested (skip for batch processing)
        if not display_gui:
            print("[*] Skipping GUI display (batch mode)")
            return
        
        # Convert to RGB for display
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        
        # Create comparison plot
        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_rgb)
        plt.title("Original Image", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(final_rgb)
        plt.title("Restored Image\n(ZeroScratches + GFPGAN)", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle("Photo Restoration Results", fontsize=18, fontweight='bold', y=0.95)
        plt.show()
        
    except Exception as e:
        print(f"[-] Error displaying results: {e}")

def main():
    """Main function"""
    print_header()
    
    # Get image path and display mode
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "img2.jpg"
    
    # Check for --no-gui flag
    display_gui = '--no-gui' not in sys.argv
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"[-] Image not found: {image_path}")
        print("Please provide a valid image path")
        return
    
    print(f"[*] Processing: {image_path}")
    
    # Check environments
    if not check_environments():
        return
    
    print("[+] Environments ready: rs, gfpgan")
    
    # Run pipeline
    success = True
    
    # Step 1: ZeroScratches
    if not run_zeroscratches(image_path):
        print("[-] Pipeline failed at ZeroScratches step")
        success = False
    
    # Step 2: GFPGAN
    if success and not run_gfpgan():
        print("[-] Pipeline failed at GFPGAN step")
        success = False
    
    if success:
        print("\n" + "=" * 60)
        print("[+] Pipeline completed successfully!")
        print("=" * 60)
        
        # Show results (with or without GUI)
        show_results(image_path, display_gui)
        
        print(f"\n[*] Output files:")
        print(f"  Step 1 (ZeroScratches): outputs/step1_zeroscratches.jpg")
        print(f"  Final result: outputs/final_restored.jpg")
        
    else:
        print("\n" + "=" * 60)
        print("[-] Pipeline failed!")
        print("=" * 60)

if __name__ == "__main__":
    main()