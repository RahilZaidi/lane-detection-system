#!/usr/bin/env python3
# preprocess_tusimple.py

import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import shutil

def parse_tusimple_json(json_path):
    """Parse TuSimple JSON annotation files."""
    annotations = []
    with open(json_path, 'r') as f:
        for line in f:
            annotations.append(json.loads(line))
    return annotations

def create_lane_mask_from_tusimple(annotation, img_shape):
    """Create a binary lane mask from TuSimple annotation."""
    lanes = annotation['lanes']
    h_samples = annotation['h_samples']
    
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    
    # Draw each lane
    for lane in lanes:
        points = []
        for i in range(len(h_samples)):
            if lane[i] != -2:  # -2 means no lane marking at this height
                y = h_samples[i]
                x = lane[i]
                points.append((int(x), int(y)))
        
        # Draw points
        if len(points) > 1:
            # Draw polylines for continuous lanes
            points = np.array(points)
            cv2.polylines(mask, [points], False, 255, 5)
    
    # Dilate to make lanes more visible
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

def process_tusimple_dataset(data_dir, output_dir):
    """Process the TuSimple dataset to create training data."""
    # First, clean the output directory if it exists
    if os.path.exists(output_dir):
        print(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    # Find all JSON annotation files
    json_files = glob.glob(os.path.join(data_dir, 'label_data_*.json'))
    
    if not json_files:
        print(f"No annotation files found in {data_dir}")
        return
    
    annotations = []
    for json_file in json_files:
        print(f"Loading annotations from {json_file}")
        annotations.extend(parse_tusimple_json(json_file))
    
    print(f"Processing {len(annotations)} annotations...")
    
    # Process each annotation
    for idx, anno in tqdm(enumerate(annotations)):
        # Get the raw file path
        raw_file = anno['raw_file']
        
        # Try different possible paths
        possible_paths = [
            os.path.join(data_dir, raw_file),
            os.path.join(data_dir, "clips", raw_file),
            raw_file if os.path.isabs(raw_file) else os.path.join(data_dir, "clips", raw_file)
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            print(f"Warning: Could not find image for {raw_file}")
            continue
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Create mask
        mask = create_lane_mask_from_tusimple(anno, img.shape[:2])
        
        # Important: Use the same naming convention for image and mask
        # The mask filename must be exactly the same as the image filename
        # but with .png extension instead of .jpg
        img_filename = f"{idx:05d}.jpg"
        mask_filename = f"{idx:05d}.png"  # Same base name, different extension
        
        cv2.imwrite(os.path.join(output_dir, 'images', img_filename), img)
        cv2.imwrite(os.path.join(output_dir, 'masks', mask_filename), mask)
    
    print(f"Processed {len(annotations)} images. Saved to {output_dir}")

if __name__ == "__main__":
    TUSIMPLE_DIR = "tusimple_dataset"
    OUTPUT_DIR = "processed_dataset"
    
    process_tusimple_dataset(TUSIMPLE_DIR, OUTPUT_DIR)
    
    print("Preprocessing complete!")