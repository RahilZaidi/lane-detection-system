# create_test_set_v2.py
import os
import random
import shutil
import argparse
import json
import cv2
import numpy as np
from tqdm import tqdm
import glob

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

def parse_tusimple_json(json_path):
    """Parse TuSimple JSON annotation files."""
    annotations = []
    with open(json_path, 'r') as f:
        for line in f:
            annotations.append(json.loads(line))
    return annotations

def create_test_set(source_dir, dest_dir, num_samples):
    """Create a test dataset from TuSimple data."""
    # Create destination directories
    os.makedirs(os.path.join(dest_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'ground_truth'), exist_ok=True)
    
    # Find JSON annotation files - look in the parent directory of the clips folder
    base_dir = os.path.dirname(os.path.dirname(source_dir))  # Go up two levels from clips/0601
    print(f"Looking for JSON files in: {base_dir}")
    json_files = glob.glob(os.path.join(base_dir, "*.json"))
    
    if not json_files:
        print("No JSON annotation files found! Trying alternative locations...")
        # Try another common location
        base_dir = os.path.dirname(source_dir)  # Just one level up
        json_files = glob.glob(os.path.join(base_dir, "*.json"))
    
    if not json_files:
        # As a last resort, search the entire tusimple_dataset directory
        print("Searching entire dataset directory for JSON files...")
        base_dir = os.path.join(base_dir, "..")  # Go to dataset root
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
    
    if not json_files:
        print("No JSON annotation files found anywhere! Please check your dataset structure.")
        return
    
    print(f"Found {len(json_files)} JSON files:")
    for json_file in json_files:
        print(f"  - {json_file}")
    
    # Parse annotations
    all_annotations = []
    for json_file in json_files:
        try:
            annotations = parse_tusimple_json(json_file)
            print(f"  Loaded {len(annotations)} annotations from {json_file}")
            all_annotations.extend(annotations)
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
    
    print(f"Loaded total of {len(all_annotations)} annotations")
    
    # Look for the specific directory in the raw_file paths
    source_dir_marker = '0601'  # Just use the folder name as a marker
    source_annotations = []
    for anno in all_annotations:
        if source_dir_marker in anno['raw_file']:
            source_annotations.append(anno)
    
    print(f"Found {len(source_annotations)} annotations from source directory")
    
    if len(source_annotations) == 0:
        print("No annotations found for the specified source directory!")
        # Print some sample raw_file paths to help debug
        print("Sample raw_file paths:")
        for i, anno in enumerate(all_annotations[:5]):
            print(f"  {i+1}: {anno['raw_file']}")
        return
    
    # Select random subset
    if len(source_annotations) > num_samples:
        selected_annotations = random.sample(source_annotations, num_samples)
    else:
        selected_annotations = source_annotations
        print(f"Warning: Requested {num_samples} samples but only found {len(source_annotations)}")
    
    # Process selected samples
    successful_count = 0
    for idx, anno in enumerate(tqdm(selected_annotations, desc="Creating test set")):
        # Get image path using the raw_file from annotation
        raw_file = anno['raw_file']
        
        # Try different possible paths
        possible_paths = [
            os.path.join(base_dir, raw_file),  # Full path from dataset root
            os.path.join(base_dir, "clips", raw_file),  # If raw_file doesn't include "clips/"
            raw_file if os.path.isabs(raw_file) else os.path.join(base_dir, raw_file)  # Handle absolute paths
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            print(f"Image not found for annotation: {raw_file}")
            # Try to extract just the filename and search for it
            filename = os.path.basename(raw_file)
            for root, _, files in os.walk(source_dir):
                if filename in files:
                    img_path = os.path.join(root, filename)
                    print(f"Found alternative path: {img_path}")
                    break
            
            if img_path is None:
                continue
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        
        # Create mask
        mask = create_lane_mask_from_tusimple(anno, img.shape[:2])
        
        # Save image and mask
        dest_img_path = os.path.join(dest_dir, 'images', f"test_{idx:04d}.jpg")
        dest_mask_path = os.path.join(dest_dir, 'ground_truth', f"test_{idx:04d}.png")
        
        cv2.imwrite(dest_img_path, img)
        cv2.imwrite(dest_mask_path, mask)
        successful_count += 1
    
    print(f"Created test set with {successful_count} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create test dataset from TuSimple')
    parser.add_argument('--source', required=True, help='Source directory with TuSimple clips')
    parser.add_argument('--dest', required=True, help='Destination directory for test data')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to include')
    
    args = parser.parse_args()
    create_test_set(args.source, args.dest, args.samples)