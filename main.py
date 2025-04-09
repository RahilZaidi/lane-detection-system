# main.py

import os
import sys
import argparse
import cv2
import time
import numpy as np

from config import Config
from models.cv_detector import CVLaneDetector
from data.dataset import LaneDataset
from utils.visualization import (
    display_image, display_lane_detection_results, 
    create_video_from_frames, add_text_to_image
)

def process_image(image_path, config, output_dir=None):
    """
    Process a single image for lane detection.
    
    Args:
        image_path: Path to the input image
        config: Configuration object
        output_dir: Directory to save output images (optional)
        
    Returns:
        result_image: Image with detected lanes
        lane_info: Dictionary containing lane information
    """
    # Create lane detector
    detector = CVLaneDetector(config)
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Detect lanes
    start_time = time.time()
    result_image, lane_info = detector.detect_lanes(image)
    end_time = time.time()
    
    # Add processing time to lane info
    lane_info['processing_time'] = end_time - start_time
    
    # Add processing time to the image
    time_text = f"Processing time: {lane_info['processing_time']:.3f} s"
    result_image = add_text_to_image(result_image, time_text, position=(50, 50))
    
    # Save output if directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, result_image)
        print(f"Output saved to {output_path}")
    
    return result_image, lane_info

def process_video(video_path, config, output_dir=None):
    """
    Process a video for lane detection.
    
    Args:
        video_path: Path to the input video
        config: Configuration object
        output_dir: Directory to save output video and frames (optional)
        
    Returns:
        output_path: Path to the output video
    """
    # Create lane detector
    detector = CVLaneDetector(config)
    
    # Create dataset object to load video frames
    dataset = LaneDataset(config)
    frames = dataset.load_video(video_path)
    
    # Process each frame
    result_frames = []
    processing_times = []
    
    print(f"Processing {len(frames)} frames...")
    for i, frame in enumerate(frames):
        # Display progress
        if i % 10 == 0:
            sys.stdout.write(f"\rProcessing frame {i+1}/{len(frames)}")
            sys.stdout.flush()
        
        # Detect lanes
        start_time = time.time()
        result_frame, lane_info = detector.detect_lanes(frame)
        end_time = time.time()
        
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        # Add frame info
        time_text = f"Frame: {i+1}/{len(frames)} | Time: {processing_time:.3f} s"
        result_frame = add_text_to_image(result_frame, time_text, position=(50, 50))
        
        if lane_info.get('left_curvature') and lane_info.get('right_curvature'):
            curve_text = f"Curvature: {(lane_info['left_curvature'] + lane_info['right_curvature'])/2:.2f} m"
            result_frame = add_text_to_image(result_frame, curve_text, position=(50, 100))
        
        if lane_info.get('vehicle_position'):
            pos_text = f"Position: {lane_info['vehicle_position']:.2f} m"
            result_frame = add_text_to_image(result_frame, pos_text, position=(50, 150))
        
        result_frames.append(result_frame)
    
    print(f"\nFinished processing {len(frames)} frames.")
    print(f"Average processing time: {np.mean(processing_times):.3f} s per frame")
    
    # Save output video
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save output video
        video_name = os.path.basename(video_path)
        video_name_no_ext = os.path.splitext(video_name)[0]
        output_path = os.path.join(output_dir, f"{video_name_no_ext}_lane_detection.mp4")
        
        # Get original video fps
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        create_video_from_frames(result_frames, output_path, fps=fps)
        
        # Save sample frames
        sample_indices = np.linspace(0, len(result_frames)-1, 5, dtype=int)
        for i, idx in enumerate(sample_indices):
            frame_path = os.path.join(output_dir, f"{video_name_no_ext}_frame_{idx}.jpg")
            cv2.imwrite(frame_path, result_frames[idx])
            print(f"Sample frame saved to {frame_path}")
        
        return output_path
    
    return None

def batch_process_images(input_dir, config, output_dir=None):
    """
    Process all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        config: Configuration object
        output_dir: Directory to save output images (optional)
    """
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    # Create lane detector
    detector = CVLaneDetector(config)
    
    # Process each image
    for i, image_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {image_file}")
        
        image_path = os.path.join(input_dir, image_file)
        
        try:
            process_image(image_path, config, output_dir)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Lane Detection System")
    parser.add_argument('--mode', choices=['image', 'video', 'batch'], required=True,
                        help="Processing mode: 'image', 'video', or 'batch'")
    parser.add_argument('--input', required=True,
                        help="Path to input image, video, or directory")
    parser.add_argument('--output_dir', default='output',
                        help="Directory to save output files")
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process based on mode
    if args.mode == 'image':
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} is not a file")
            return
        
        try:
            result_image, lane_info = process_image(args.input, config, args.output_dir)
            
            # Display results
            image = cv2.imread(args.input)
            display_lane_detection_results(image, result_image, lane_info)
            
            print("Lane Information:")
            for key, value in lane_info.items():
                if value is not None:
                    print(f"  {key}: {value}")
        
        except Exception as e:
            print(f"Error processing image: {e}")
    
    elif args.mode == 'video':
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} is not a file")
            return
        
        try:
            output_path = process_video(args.input, config, args.output_dir)
            if output_path:
                print(f"Processed video saved to {output_path}")
        
        except Exception as e:
            print(f"Error processing video: {e}")
    
    elif args.mode == 'batch':
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            return
        
        batch_process_images(args.input, config, args.output_dir)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()