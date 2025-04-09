# evaluate_model.py
import cv2
import numpy as np
import tensorflow as tf
import os
import time
from sklearn.metrics import precision_recall_fscore_support
from models.dl_detector import DLLaneDetector
from config import Config

def calculate_metrics(pred_masks, true_masks, threshold=0.5):
    """Calculate performance metrics."""
    # Flatten the masks
    pred_flat = (pred_masks > threshold).flatten()
    true_flat = (true_masks > 0).flatten()
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_flat, pred_flat, average='binary'
    )
    
    # Calculate IoU
    intersection = np.logical_and(pred_flat, true_flat).sum()
    union = np.logical_or(pred_flat, true_flat).sum()
    iou = intersection / union if union > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou
    }

def evaluate_model(model_path, test_dir):
    """Evaluate the model on test data."""
    # Load configuration
    config = Config()
    
    # Create detector instance
    detector = DLLaneDetector(config)
    
    # Load model
    custom_objects = {
        'weighted_binary_crossentropy': detector.weighted_binary_crossentropy 
        if hasattr(detector, 'weighted_binary_crossentropy') else None
    }
    detector.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Get test images and ground truth masks
    image_dir = os.path.join(test_dir, 'images')
    mask_dir = os.path.join(test_dir, 'ground_truth')
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    
    # Metrics storage
    metrics = {
        'precision': [],
        'recall': [],
        'f1_score': [],
        'iou': [],
        'processing_time': []
    }
    
    # Process each image
    for img_file in image_files:
        # Load image
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image: {img_path}")
            continue
            
        # Load ground truth mask
        mask_path = os.path.join(mask_dir, img_file.replace('.jpg', '.png'))
        if not os.path.exists(mask_path):
            print(f"No mask found for: {img_path}")
            continue
            
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Measure processing time
        start_time = time.time()
        
        # Make prediction
        preprocessed = cv2.resize(image, (detector.input_shape[1], detector.input_shape[0]))
        preprocessed = preprocessed.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(preprocessed, axis=0)
        prediction = detector.model.predict(input_tensor)[0]
        
        # Resize prediction to match ground truth
        pred_mask = cv2.resize(prediction, (true_mask.shape[1], true_mask.shape[0]))
        
        # Measure processing time
        end_time = time.time()
        metrics['processing_time'].append(end_time - start_time)
        
        # Calculate performance metrics
        batch_metrics = calculate_metrics(pred_mask, true_mask)
        for key, value in batch_metrics.items():
            metrics[key].append(value)
    
    # Calculate average metrics
    avg_metrics = {
        'precision': np.mean(metrics['precision']),
        'recall': np.mean(metrics['recall']),
        'f1_score': np.mean(metrics['f1_score']),
        'iou': np.mean(metrics['iou']),
        'processing_time': np.mean(metrics['processing_time'])
    }
    
    return avg_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate lane detection model')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--test_dir', required=True, help='Directory containing test data')
    parser.add_argument('--output', default='evaluation_results.txt', help='Output file for results')
    args = parser.parse_args()
    
    results = evaluate_model(args.model, args.test_dir)
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"IoU: {results['iou']:.4f}")
    print(f"Average Processing Time: {results['processing_time']*1000:.2f} ms")
    
    # Save results to file
    with open(args.output, 'w') as f:
        f.write("Lane Detection Model Evaluation Results\n")
        f.write("=====================================\n\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1_score']:.4f}\n")
        f.write(f"IoU: {results['iou']:.4f}\n")
        f.write(f"Average Processing Time: {results['processing_time']*1000:.2f} ms\n")