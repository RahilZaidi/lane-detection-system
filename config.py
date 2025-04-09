# config.py

import numpy as np

class Config:
    """Configuration parameters for the lane detection system."""
    
    def __init__(self):
        # Image parameters
        self.img_height = 720
        self.img_width = 1280
        
        # Dataset parameters
        self.data_dir = "data/raw"
        self.train_split = 0.8
        self.batch_size = 8
        
        # Traditional CV detector parameters
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.rho = 1
        self.theta = np.pi / 180
        self.hough_threshold = 20
        self.min_line_length = 20
        self.max_line_gap = 300
        
        # Region of interest - trapezoid shape
        # (bottom left, top left, top right, bottom right)
        self.roi_vertices = np.array([
            [200, self.img_height],
            [self.img_width // 2 - 100, self.img_height // 2 + 50],
            [self.img_width // 2 + 100, self.img_height // 2 + 50],
            [self.img_width - 200, self.img_height]
        ], dtype=np.int32)
        
        # Deep learning model parameters
        self.model_type = 'unet'  # Options: 'unet', 'deeplabv3', 'segnet'
        self.input_shape = (256, 512, 3)  # (height, width, channels)
        self.num_classes = 1  # Background and lane
        
        # Training parameters
        self.learning_rate = 0.001
        self.epochs = 50
        self.early_stopping_patience = 10
        
        # Inference parameters
        self.confidence_threshold = 0.5
        
        # Post-processing parameters
        self.lane_smoothing_window = 5  # Number of frames for temporal smoothing