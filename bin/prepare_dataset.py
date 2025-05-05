#!/usr/bin/env python3

import os
import argparse
import glob
import shutil
import cv2
import numpy as np
from tqdm import tqdm

def create_directory_structure(base_dir):
    """Create the required directory structure for training"""
    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "visual_test"), exist_ok=True)
    print(f"Created directory structure in {base_dir}")

def check_dataset(base_dir, img_suffix=".png", mask_suffix="_mask"):
    """Check if dataset has proper image-mask pairs"""
    issues = False
    for subset in ["train", "val", "visual_test"]:
        subset_dir = os.path.join(base_dir, subset)
        if not os.path.exists(subset_dir):
            print(f"ERROR: Directory {subset_dir} does not exist!")
            issues = True
            continue
            
        images = [f for f in os.listdir(subset_dir) if f.endswith(img_suffix) and not f.endswith(f"{mask_suffix}{img_suffix}")]
        masks = [f.replace(img_suffix, f"{mask_suffix}{img_suffix}") for f in images]
        
        missing_masks = [m for m in masks if not os.path.exists(os.path.join(subset_dir, m))]
        
        print(f"{subset}: Found {len(images)} images and {len(masks) - len(missing_masks)} matching masks")
        if missing_masks:
            print(f"WARNING: {len(missing_masks)} masks are missing in {subset}!")
            for missing in missing_masks[:5]:
                print(f"  - Missing: {missing}")
            if len(missing_masks) > 5:
                print(f"  - ... and {len(missing_masks) - 5} more")
            issues = True
    
    return not issues

def fix_filenames(base_dir, img_suffix=".png", mask_suffix="_mask"):
    """Fix common naming issues with mask files"""
    for subset in ["train", "val", "visual_test"]:
        subset_dir = os.path.join(base_dir, subset)
        if not os.path.exists(subset_dir):
            continue

        # Find all mask files that don't follow the naming convention
        possible_mask_files = glob.glob(os.path.join(subset_dir, f"*mask*{img_suffix}"))
        renamed = 0
        
        for mask_path in possible_mask_files:
            mask_filename = os.path.basename(mask_path)
            if mask_suffix in mask_filename:
                continue  # Already has correct naming
                
            # Try to find the corresponding image
            possible_img_name = mask_filename.replace("mask", "").replace("_", "")
            possible_img_path = os.path.join(subset_dir, possible_img_name)
            
            if os.path.exists(possible_img_path):
                # Found matching image, rename mask to follow convention
                new_mask_name = possible_img_name.replace(img_suffix, f"{mask_suffix}{img_suffix}")
                new_mask_path = os.path.join(subset_dir, new_mask_name)
                shutil.move(mask_path, new_mask_path)
                renamed += 1
                print(f"Renamed: {mask_filename} → {new_mask_name}")
        
        if renamed > 0:
            print(f"Fixed {renamed} mask filenames in {subset}")

def create_sample_data(base_dir, count=5, img_suffix=".png", mask_suffix="_mask"):
    """Create sample image-mask pairs for testing"""
    for subset in ["train", "val", "visual_test"]:
        subset_dir = os.path.join(base_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        
        for i in range(count):
            # Create a sample image (black with a white rectangle)
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
            
            # Create a corresponding mask (black with a white area to be inpainted)
            mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.rectangle(mask, (100, 100), (150, 150), 255, -1)
            
            # Save the files
            img_path = os.path.join(subset_dir, f"sample_{i+1}{img_suffix}")
            mask_path = os.path.join(subset_dir, f"sample_{i+1}{mask_suffix}{img_suffix}")
            
            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)
        
        print(f"Created {count} sample image-mask pairs in {subset_dir}")

def main():
    parser = argparse.ArgumentParser(description="Prepare and verify dataset for LaMa inpainting")
    parser.add_argument("dataset_dir", help="Path to dataset directory")
    parser.add_argument("--img-suffix", default=".png", help="Image filename suffix (default: .png)")
    parser.add_argument("--mask-suffix", default="_mask", help="Mask filename suffix before extension (default: _mask)")
    parser.add_argument("--create-dirs", action="store_true", help="Create directory structure if it doesn't exist")
    parser.add_argument("--fix-names", action="store_true", help="Try to fix common mask naming issues")
    parser.add_argument("--create-samples", action="store_true", help="Create sample image-mask pairs for testing")
    parser.add_argument("--sample-count", type=int, default=5, help="Number of sample pairs to create per subset")
    
    args = parser.parse_args()
    
    if args.create_dirs:
        create_directory_structure(args.dataset_dir)
    
    if args.fix_names:
        fix_filenames(args.dataset_dir, args.img_suffix, args.mask_suffix)
    
    if args.create_samples:
        create_sample_data(args.dataset_dir, args.sample_count, args.img_suffix, args.mask_suffix)
    
    # Always check the dataset at the end
    is_valid = check_dataset(args.dataset_dir, args.img_suffix, args.mask_suffix)
    
    if is_valid:
        print("\nSUCCESS: Dataset looks properly set up!")
    else:
        print("\nWARNING: Dataset has issues! See above for details.")
    
    # Print example usage for training
    print("\nExample usage for training:")
    print(f"python bin/train.py model.lama.config=lama location=my_dataset data=abl-04-256-mh-dist")

if __name__ == "__main__":
    main()
