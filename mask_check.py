# mask_check.py
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import glob

def display_images_and_masks(dataset_dir, num_samples=5):
    image_files = sorted(glob.glob(os.path.join(dataset_dir, 'images', '*.jpg')))[:num_samples]
    
    for img_path in image_files:
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get corresponding mask
        mask_path = os.path.join(dataset_dir, 'masks', 
                              os.path.basename(img_path).replace('.jpg', '.png'))
        
        if not os.path.exists(mask_path):
            print(f"Mask not found for {img_path}")
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Show the images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax2.imshow(mask, cmap='gray')
        ax2.set_title(f'Mask (white pixel count: {np.sum(mask > 0)})')
        plt.show()
        
        print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
        print(f"Mask min: {mask.min()}, max: {mask.max()}, unique values: {np.unique(mask)}")
        print(f"Percentage of lane pixels: {np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100:.2f}%")
        print("-" * 50)

if __name__ == "__main__":
    display_images_and_masks("processed_dataset")