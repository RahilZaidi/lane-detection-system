# data/preprocessing.py

import cv2
import numpy as np
import os
import glob
import random
from sklearn.model_selection import train_test_split

def load_image(image_path):
    """
    Load an image from the given path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        image: Loaded image in BGR format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return image

def resize_image(image, target_height, target_width):
    """
    Resize an image to the target dimensions.
    
    Args:
        image: Input image
        target_height: Target height
        target_width: Target width
        
    Returns:
        resized_image: Resized image
    """
    return cv2.resize(image, (target_width, target_height))

def normalize_image(image):
    """
    Normalize image pixel values to [0, 1].
    
    Args:
        image: Input image
        
    Returns:
        normalized_image: Image with pixel values in [0, 1]
    """
    return image.astype(np.float32) / 255.0

def augment_image(image, mask=None):
    """
    Apply data augmentation to an image and its mask (if provided).
    
    Args:
        image: Input image
        mask: Corresponding segmentation mask (optional)
        
    Returns:
        augmented_image: Augmented image
        augmented_mask: Augmented mask (if mask was provided)
    """
    # Random brightness adjustment
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.8, 1.2)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        if mask is not None:
            mask = cv2.flip(mask, 1)
    
    # Random shadow
    if random.random() > 0.5:
        rows, cols, _ = image.shape
        top_y = np.random.rand() * rows
        top_x = 0
        bot_x = cols
        bot_y = np.random.rand() * rows
        
        image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        shadow_mask = np.zeros_like(image)
        x_m = np.mgrid[0:rows, 0:cols][0]
        y_m = np.mgrid[0:rows, 0:cols][1]
        
        shadow_mask[((x_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (y_m - top_y)) >= 0] = 1
        
        if random.random() > 0.5:
            shadow_mask = 1 - shadow_mask
            
        shadow_factor = random.uniform(0.6, 0.9)
        
        image_hls[:, :, 1][shadow_mask == 0] = image_hls[:, :, 1][shadow_mask == 0] * shadow_factor
        image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2BGR)
    
    if mask is not None:
        return image, mask
    else:
        return image

def create_train_val_split(images_dir, masks_dir=None, train_ratio=0.8, random_state=42):
    """
    Create train/validation split from images and masks directories.
    
    Args:
        images_dir: Directory containing image files
        masks_dir: Directory containing mask files (optional)
        train_ratio: Ratio of training samples (default 0.8)
        random_state: Random seed for reproducibility
        
    Returns:
        train_images: List of paths to training images
        val_images: List of paths to validation images
        train_masks: List of paths to training masks (if masks_dir provided)
        val_masks: List of paths to validation masks (if masks_dir provided)
    """
    from sklearn.model_selection import train_test_split
    
    # Get all image paths
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) + 
                         glob.glob(os.path.join(images_dir, "*.png")))
    
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    
    # Split into train and validation sets
    train_images, val_images = train_test_split(
        image_paths, train_size=train_ratio, random_state=random_state
    )
    
    # If mask directory is provided, create corresponding mask lists
    if masks_dir is not None:
        # Initialize mask lists
        train_masks = []
        val_masks = []
        
        for img_path in train_images:
            # Get base filename without extension
            img_filename = os.path.basename(img_path)
            img_basename = os.path.splitext(img_filename)[0]
            
            # Look for mask with same base name but .png extension
            mask_filename = f"{img_basename}.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            
            if os.path.exists(mask_path):
                train_masks.append(mask_path)
            else:
                print(f"Warning: No mask found for {img_path}, searched for {mask_path}")
                # Try alternative naming schemes
                alt_mask_path = os.path.join(masks_dir, img_filename.replace('.jpg', '.png'))
                if os.path.exists(alt_mask_path):
                    train_masks.append(alt_mask_path)
                else:
                    raise ValueError(f"Mask not found for {img_path}, tried {mask_path} and {alt_mask_path}")
        
        for img_path in val_images:
            img_filename = os.path.basename(img_path)
            img_basename = os.path.splitext(img_filename)[0]
            
            mask_filename = f"{img_basename}.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            
            if os.path.exists(mask_path):
                val_masks.append(mask_path)
            else:
                print(f"Warning: No mask found for {img_path}, searched for {mask_path}")
                # Try alternative naming schemes
                alt_mask_path = os.path.join(masks_dir, img_filename.replace('.jpg', '.png'))
                if os.path.exists(alt_mask_path):
                    val_masks.append(alt_mask_path)
                else:
                    raise ValueError(f"Mask not found for {img_path}, tried {mask_path} and {alt_mask_path}")
        
        return train_images, val_images, train_masks, val_masks
    
    return train_images, val_images
def preprocess_for_cv_detector(image, config):
    """
    Preprocess an image for the traditional CV-based lane detector.
    
    Args:
        image: Input image
        config: Configuration object
        
    Returns:
        processed_image: Processed image ready for the CV detector
    """
    # Resize image to the configured dimensions
    resized_image = resize_image(image, config.img_height, config.img_width)
    
    return resized_image

def preprocess_for_dl_detector(image, config, mask=None, training=True):
    """
    Preprocess an image (and mask if provided) for the deep learning lane detector.
    
    Args:
        image: Input image
        config: Configuration object
        mask: Segmentation mask (optional)
        training: Whether preprocessing for training (True) or inference (False)
        
    Returns:
        processed_image: Processed image ready for the DL detector
        processed_mask: Processed mask (if mask was provided)
    """
    # Resize to the input shape expected by the DL model
    resized_image = resize_image(image, config.input_shape[0], config.input_shape[1])
    
    # Normalize pixel values to [0, 1]
    normalized_image = normalize_image(resized_image)
    
    # Apply data augmentation during training
    if training and mask is not None:
        resized_mask = resize_image(mask, config.input_shape[0], config.input_shape[1])
        augmented_image, augmented_mask = augment_image(normalized_image, resized_mask)
        return augmented_image, augmented_mask
    elif training:
        augmented_image = augment_image(normalized_image)
        return augmented_image
    
    if mask is not None:
        resized_mask = resize_image(mask, config.input_shape[0], config.input_shape[1])
        return normalized_image, resized_mask
    
    return normalized_image