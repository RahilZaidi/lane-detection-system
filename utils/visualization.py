# utils/visualization.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os

def display_image(image, figsize=(12, 8), title=None, cmap=None):
    """
    Display an image using matplotlib.
    
    Args:
        image: Image to display
        figsize: Figure size (width, height) in inches
        title: Title to display above the image
        cmap: Colormap to use (default None)
    """
    figure(figsize=figsize)
    
    # Convert BGR to RGB if image has 3 channels
    if len(image.shape) == 3 and image.shape[2] == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_img = image
    
    plt.imshow(display_img, cmap=cmap)
    
    if title:
        plt.title(title)
    
    plt.axis('on')
    plt.tight_layout()
    plt.show()

def display_multiple_images(images, titles=None, figsize=(15, 10), rows=1, cols=None, cmap=None):
    """
    Display multiple images in a grid layout.
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        figsize: Figure size (width, height) in inches
        rows: Number of rows in the grid
        cols: Number of columns in the grid (calculated from rows if None)
        cmap: Colormap to use (default None)
    """
    if cols is None:
        cols = int(np.ceil(len(images) / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows * cols == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        # Convert BGR to RGB if image has 3 channels
        if len(img.shape) == 3 and img.shape[2] == 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            display_img = img
        
        ax.imshow(display_img, cmap=cmap)
        
        if titles and i < len(titles):
            ax.set_title(titles[i])
        
        ax.axis('on')
    
    # Hide any unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def display_lane_detection_results(original_image, processed_image, lane_info=None, figsize=(15, 10)):
    """
    Display the original image alongside the processed image with lane detection results.
    
    Args:
        original_image: Original input image
        processed_image: Image with lane detection visualization
        lane_info: Dictionary containing lane information (curvature, position, etc.)
        figsize: Figure size (width, height) in inches
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Convert BGR to RGB
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    ax1.imshow(original_rgb)
    ax1.set_title('Original Image')
    ax1.axis('on')
    
    ax2.imshow(processed_rgb)
    ax2.set_title('Lane Detection Result')
    ax2.axis('on')
    
    # Add lane information as text if provided
    if lane_info is not None:
        info_text = []
        
        if lane_info.get('left_curvature') is not None:
            info_text.append(f"Left Curvature: {lane_info['left_curvature']:.2f} m")
        
        if lane_info.get('right_curvature') is not None:
            info_text.append(f"Right Curvature: {lane_info['right_curvature']:.2f} m")
        
        if lane_info.get('vehicle_position') is not None:
            direction = "right" if lane_info['vehicle_position'] > 0 else "left"
            info_text.append(f"Vehicle Position: {abs(lane_info['vehicle_position']):.2f} m {direction} of center")
        
        if lane_info.get('lane_width') is not None:
            info_text.append(f"Lane Width: {lane_info['lane_width']:.2f} m")
        
        info_str = "\n".join(info_text)
        fig.text(0.5, 0.01, info_str, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def create_video_from_frames(frames, output_path, fps=30):
    """
    Create a video from a list of frames.
    
    Args:
        frames: List of image frames (must all have the same dimensions)
        output_path: Path to save the output video
        fps: Frames per second
    """
    if not frames:
        raise ValueError("No frames provided")
    
    # Get dimensions from the first frame
    height, width = frames[0].shape[:2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to the video
    for frame in frames:
        out.write(frame)
    
    # Release the VideoWriter
    out.release()
    
    print(f"Video saved to {output_path}")

def add_text_to_image(image, text, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, 
                     font_scale=1, color=(255, 255, 255), thickness=2):
    """
    Add text to an image.
    
    Args:
        image: Input image
        text: Text to add
        position: Position (x, y) to place the text
        font: Font type
        font_scale: Font scale factor
        color: Text color (B, G, R)
        thickness: Line thickness
        
    Returns:
        result_image: Image with text added
    """
    result_image = image.copy()
    cv2.putText(result_image, text, position, font, font_scale, color, thickness)
    return result_image

def visualize_lane_detection_pipeline(image, edges, roi_mask, hough_lines, final_result):
    """
    Visualize each step of the lane detection pipeline.
    
    Args:
        image: Original input image
        edges: Output of edge detection
        roi_mask: Region of interest mask
        hough_lines: Image with Hough lines drawn
        final_result: Final lane detection result
    """
    # Create color versions of grayscale images for visualization
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    roi_mask_color = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    
    # Create a list of images and titles
    images = [image, edges_color, roi_mask_color, hough_lines, final_result]
    titles = ['Original Image', 'Edge Detection', 'Region of Interest', 'Hough Lines', 'Final Result']
    
    # Display all images
    display_multiple_images(images, titles, figsize=(20, 10), rows=2, cols=3)