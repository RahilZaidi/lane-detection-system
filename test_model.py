# test_model.py
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.dl_detector import DLLaneDetector
from config import Config

def test_lane_detection(model_path, image_path):
    # Load configuration
    config = Config()
    
    # Create detector instance
    detector = DLLaneDetector(config)
    
    # Load model
    detector.model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load and process image
    image = cv2.imread(image_path)
    result_image, lane_info = detector.detect_lanes(image)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Result with lane detection
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Lane Detection')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('lane_detection_result.png')
    plt.show()
    
    # Print lane information
    print("Lane Information:")
    for key, value in lane_info.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    return result_image, lane_info

if __name__ == "__main__":
    # Path to your saved model
    model_path = "/Users/rahilzaidi/Documents/LaneDetetctor/models/unet_20250404-225646.keras"
    
    # Test on a sample image
    test_image = "/Users/rahilzaidi/Documents/LaneDetetctor/test_data/images/9.jpg"  # Change to an actual image path
    test_lane_detection(model_path, test_image)