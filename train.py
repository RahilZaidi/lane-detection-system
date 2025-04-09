# train.py

import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from datetime import datetime

from config import Config
from models.dl_detector import DLLaneDetector
from data.dataset import LaneDataset
from utils.visualization import display_multiple_images

def train_model(config, data_dir, model_dir, log_dir, epochs=None, batch_size=None):
    """
    Train a deep learning lane detection model.
    
    Args:
        config: Configuration object
        data_dir: Directory containing the training data
        model_dir: Directory to save the trained model
        log_dir: Directory to save training logs
        epochs: Number of training epochs (uses config value if None)
        batch_size: Batch size (uses config value if None)
    """
    # Override config values if provided
    if batch_size:
        config.batch_size = batch_size
    
    if epochs:
        config.epochs = epochs
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create dataset handler
    dataset_handler = LaneDataset(config)
    
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    
    try:
        # First try to load as TuSimple dataset
        train_dataset, val_dataset = dataset_handler.load_tusimple_dataset(data_dir)
        print("Loaded TuSimple dataset format.")
    except Exception as e:
        print(f"Could not load as TuSimple dataset: {e}")
        print("Trying to load as custom dataset...")
        
        # Try to find images and masks directories
        images_dir = os.path.join(data_dir, 'images')
        masks_dir = os.path.join(data_dir, 'masks')
        
        if os.path.exists(images_dir) and os.path.exists(masks_dir):
            train_dataset, val_dataset = dataset_handler.load_custom_dataset(images_dir, masks_dir)
            print("Loaded custom dataset with segmentation masks.")
        else:
            raise ValueError(f"Could not find proper dataset structure in {data_dir}")
    
    # Create model
    print(f"Creating {config.model_type} model...")
    detector = DLLaneDetector(config)
    
    # Create TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(log_dir, f"{config.model_type}_{timestamp}"),
        histogram_freq=1,
        update_freq='epoch'
    )
    
    # Create model checkpoint callback
    checkpoint_path = os.path.join(model_dir, f"{config.model_type}_{timestamp}.keras")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        save_weights_only=False,
        verbose=1
    )
    
    # Create early stopping callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Create learning rate scheduler callback
    lr_scheduler_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Combine callbacks
    callbacks = [
        tensorboard_callback,
        checkpoint_callback,
        early_stopping_callback,
        lr_scheduler_callback
    ]
    
    # Train the model
    print(f"Starting training for {config.epochs} epochs...")
    history = detector.train(train_dataset, val_dataset, epochs=config.epochs, callbacks=callbacks)
    
    # Save the final model
    final_model_path = os.path.join(model_dir, f"{config.model_type}_final_{timestamp}.keras")
    detector.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot training history
    plot_training_history(history, log_dir, config.model_type, timestamp)
    
    # Evaluate on validation set
    print("Evaluating model on validation data...")
    evaluation = detector.model.evaluate(val_dataset)
    
    print("Validation Results:")
    for metric_name, value in zip(detector.model.metrics_names, evaluation):
        print(f"  {metric_name}: {value}")
    
    # Save evaluation results
    with open(os.path.join(log_dir, f"{config.model_type}_{timestamp}_evaluation.txt"), 'w') as f:
        f.write(f"Model: {config.model_type}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Validation Results:\n")
        for metric_name, value in zip(detector.model.metrics_names, evaluation):
            f.write(f"  {metric_name}: {value}\n")
    
    # Visualize some predictions
    print("Generating visualizations...")
    visualize_predictions(detector, val_dataset, log_dir, config.model_type, timestamp)
    
    return detector, history

def plot_training_history(history, log_dir, model_type, timestamp):
    """
    Plot and save the training history.
    
    Args:
        history: Training history object
        log_dir: Directory to save the plots
        model_type: Type of model used
        timestamp: Timestamp string for filenames
    """
    # Create figure for accuracy
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(log_dir, f"{model_type}_{timestamp}_history.png"))
    plt.close()
    
    # If IoU metric is available, plot it separately
    if 'iou' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['iou'])
        plt.plot(history.history['val_iou'])
        plt.title('Model IoU')
        plt.ylabel('IoU')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(log_dir, f"{model_type}_{timestamp}_iou.png"))
        plt.close()

def visualize_predictions(detector, dataset, log_dir, model_type, timestamp):
    """
    Visualize and save model predictions on sample images.
    
    Args:
        detector: Trained lane detector
        dataset: Validation dataset
        log_dir: Directory to save visualizations
        model_type: Type of model used
        timestamp: Timestamp string for filenames
    """
    # Create directory for visualizations
    vis_dir = os.path.join(log_dir, f"{model_type}_{timestamp}_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get a batch of validation data
    for images, masks in dataset.take(1):
        # Select a few images to visualize
        num_samples = min(4, len(images))
        selected_indices = np.random.choice(len(images), num_samples, replace=False)
        
        for i, idx in enumerate(selected_indices):
            # Get the image and ground truth mask
            image = images[idx].numpy()
            mask = masks[idx].numpy()
            
            # Convert image from [0,1] to [0,255] for visualization
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Make prediction
            prediction = detector.model.predict(np.expand_dims(image, axis=0))[0]
            
            # Threshold prediction to get binary mask
            binary_prediction = (prediction > detector.config.confidence_threshold).astype(np.uint8)
            
            # Create overlay of prediction on image
            overlay = image.copy()
            overlay[binary_prediction > 0] = [0, 1, 0]  # Green for predicted lane
            blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            
            # Create visualization with multiple images
            display_images = [
                image,                  # Original image
                mask,                   # Ground truth mask
                binary_prediction,      # Predicted mask
                blended                 # Overlay
            ]
            
            titles = [
                'Original Image',
                'Ground Truth',
                'Prediction',
                'Overlay'
            ]
            
            # Display and save
            fig = plt.figure(figsize=(16, 4))
            for j, (img, title) in enumerate(zip(display_images, titles)):
                plt.subplot(1, 4, j+1)
                
                # Handle different channel configurations
                if len(img.shape) == 2 or img.shape[2] == 1:
                    plt.imshow(img, cmap='gray')
                else:
                    plt.imshow(img)
                
                plt.title(title)
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"sample_{i}.png"))
            plt.close()
            
            # Save the overlay image separately
            overlay_path = os.path.join(vis_dir, f"overlay_{i}.png")
            cv2.imwrite(overlay_path, (blended * 255).astype(np.uint8)[:, :, ::-1])  # Convert RGB to BGR for OpenCV

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Lane Detection Model")
    parser.add_argument('--data_dir', required=True,
                        help="Directory containing the dataset")
    parser.add_argument('--model_dir', default='models',
                        help="Directory to save the trained model")
    parser.add_argument('--log_dir', default='logs',
                        help="Directory to save training logs")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Batch size for training")
    parser.add_argument('--model_type', choices=['unet', 'deeplabv3'], default=None,
                        help="Type of model to use")
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Override model type if specified
    if args.model_type:
        config.model_type = args.model_type
    
    # Train the model
    train_model(
        config=config,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()