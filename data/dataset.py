# data/dataset.py

import os
import numpy as np
import cv2
import tensorflow as tf
from . import preprocessing
from sklearn.model_selection import train_test_split

class LaneDataset:
    """Dataset handler for lane detection."""
    
    def __init__(self, config):
        """
        Initialize the lane dataset.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.batch_size = config.batch_size
        
    def load_tusimple_dataset(self, data_dir):
        """
        Load and prepare the TuSimple dataset.
        
        Args:
            data_dir: Directory containing the TuSimple dataset
            
        Returns:
            train_dataset: TensorFlow dataset for training
            val_dataset: TensorFlow dataset for validation
        """
        # Parse annotation files
        json_files = [
            os.path.join(data_dir, 'label_data_0313.json'),
            os.path.join(data_dir, 'label_data_0531.json'),
            os.path.join(data_dir, 'label_data_0601.json')
        ]
        
        annotations = []
        for json_file in json_files:
            if os.path.exists(json_file):
                annotations.extend(parse_tusimple_json(json_file))
        
        print(f"Loaded {len(annotations)} annotations from TuSimple dataset")
        
        # Extract image paths and create masks
        image_paths = []
        masks = []
        
        for anno in annotations:
            # Get the last frame of each clip
            raw_file = anno['raw_file']
            # Convert relative path to absolute path
            clip_path = os.path.join(data_dir, raw_file)
            # Get the directory containing the clip frames
            clip_dir = os.path.dirname(clip_path)
            # Get all frames in the clip
            frame_files = sorted(glob.glob(os.path.join(clip_dir, '*.jpg')))
            if not frame_files:
                continue
            
            # Get the last frame (which has annotation)
            last_frame = frame_files[-1]
            image_paths.append(last_frame)
            
            # Create mask for this frame
            img = cv2.imread(last_frame)
            if img is None:
                continue
            
            mask = create_lane_mask_from_tusimple(anno, img.shape[:2])
            mask_path = last_frame.replace('.jpg', '_mask.png')
            cv2.imwrite(mask_path, mask)
            masks.append(mask_path)
        
        # Split into train and validation sets
        train_indices, val_indices = train_test_split(
            range(len(image_paths)), 
            test_size=1-self.config.train_split,
            random_state=42
        )
        
        train_images = [image_paths[i] for i in train_indices]
        train_masks = [masks[i] for i in train_indices]
        val_images = [image_paths[i] for i in val_indices]
        val_masks = [masks[i] for i in val_indices]
        
        # Create TensorFlow datasets
        train_dataset = self._create_tf_dataset(train_images, train_masks, training=True)
        val_dataset = self._create_tf_dataset(val_images, val_masks, training=False)
    
        return train_dataset, val_dataset
    
    def load_custom_dataset(self, images_dir, masks_dir=None):
        """
        Load and prepare a custom dataset.
        
        Args:
            images_dir: Directory containing image files
            masks_dir: Directory containing mask files (optional)
            
        Returns:
            dataset: TensorFlow dataset
        """
        if masks_dir:
            # For supervised learning with segmentation masks
            train_images, val_images, train_masks, val_masks = preprocessing.create_train_val_split(
                images_dir, masks_dir, self.config.train_split
            )
            
            train_dataset = self._create_tf_dataset(train_images, train_masks, training=True)
            val_dataset = self._create_tf_dataset(val_images, val_masks, training=False)
            
            return train_dataset, val_dataset
        else:
            # For unsupervised or traditional CV approaches
            image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            dataset = tf.data.Dataset.from_tensor_slices(image_paths)
            dataset = dataset.map(self._load_and_preprocess_image)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            
            return dataset
    
    def load_video(self, video_path):
        """
        Load frames from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            frames: List of video frames
        """
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame for CV detector
            processed_frame = preprocessing.preprocess_for_cv_detector(frame, self.config)
            frames.append(processed_frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"Failed to extract frames from {video_path}")
        
        return frames
    
    def _create_tf_dataset(self, image_paths, mask_paths=None, training=True):
        """
        Create a TensorFlow dataset from image and mask paths.
        
        Args:
            image_paths: List of paths to images
            mask_paths: List of paths to masks (optional)
            training: Whether this is a training dataset
            
        Returns:
            dataset: TensorFlow dataset
        """
        if mask_paths:
            # For segmentation with masks
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
            if training:
                dataset = dataset.shuffle(buffer_size=len(image_paths))
            dataset = dataset.map(self._load_and_preprocess_image_mask, 
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            # Without masks
            dataset = tf.data.Dataset.from_tensor_slices(image_paths)
            if training:
                dataset = dataset.shuffle(buffer_size=len(image_paths))
            dataset = dataset.map(self._load_and_preprocess_image,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset
    
    def _load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image for the deep learning model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            processed_image: Preprocessed image tensor
        """
        # Load image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        
        # Resize to model input shape
        img = tf.image.resize(img, (self.config.input_shape[0], self.config.input_shape[1]))
        
        return img
    
    def _load_and_preprocess_image_mask(self, image_path, mask_path):
        """
        Load and preprocess an image and its corresponding mask.
        
        Args:
            image_path: Path to the image file
            mask_path: Path to the mask file
            
        Returns:
            processed_image: Preprocessed image tensor
            processed_mask: Preprocessed mask tensor
        """
        # Load image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        
        # Load mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        
        # Binary mask for lane/non-lane
        mask = tf.cast(mask > 0, tf.float32)
        
        # Resize to model input shape
        img = tf.image.resize(img, (self.config.input_shape[0], self.config.input_shape[1]))
        mask = tf.image.resize(mask, (self.config.input_shape[0], self.config.input_shape[1]))
        
        return img, mask

    def parse_tusimple_json(json_path):
        """
        Parse TuSimple JSON annotation files.
        
        Args:
            json_path: Path to JSON annotation file
            
        Returns:
            annotations: List of annotation dictionaries
        """
        annotations = []
        with open(json_path, 'r') as f:
            for line in f:
                annotations.append(json.loads(line))
        return annotations
    
    def create_lane_mask_from_tusimple(annotation, img_shape):
        """
        Create a binary lane mask from TuSimple annotation.
        
        Args:
            annotation: TuSimple annotation dictionary
            img_shape: Shape of the image (height, width)
            
        Returns:
            mask: Binary lane mask
        """
        lanes = annotation['lanes']
        h_samples = annotation['h_samples']
        
        mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
        
        # Draw each lane
        for lane in lanes:
            for i in range(len(h_samples)):
                if lane[i] != -2:  # -2 means no lane marking at this height
                    y = h_samples[i]
                    x = lane[i]
                    # Draw a small line segment or circle at this point
                    cv2.circle(mask, (int(x), int(y)), 5, 255, -1)
        
        # Connect points to form continuous lanes
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        
        return mask