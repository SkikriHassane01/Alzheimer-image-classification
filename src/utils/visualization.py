import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from src.utils.logger import get_logger

logger = get_logger("6_visualization")

def plot_training_history(history, save_path=None):
    """
    Plot training history (accuracy and loss) and optionally save the figure.
    
    Args:
        history: Training history object from model.fit()
        save_path: Path to save the plot, if None, the plot is just displayed
    """
    plt.figure(figsize=(14, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved training history plot to {save_path}")
    
    plt.close()

def plot_confusion_matrix(cm, class_labels, save_path=None, normalize=False):
    """
    Plot confusion matrix and optionally save the figure.
    
    Args:
        cm: Confusion matrix array
        class_labels: List of class labels for axes
        save_path: Path to save the plot, if None, the plot is just displayed
        normalize: Whether to normalize the confusion matrix
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, values_format='.0f' if not normalize else '.2f')
    
    # Use disp.ax_ to set ticks (fix tick label mismatch error)
    labels = disp.display_labels if disp.display_labels is not None else list(range(cm.shape[0]))
    disp.ax_.set_xticks(range(len(labels)))  # override default ticks
    disp.ax_.set_xticklabels(labels)
    disp.ax_.set_yticks(range(len(labels)))
    disp.ax_.set_yticklabels(labels)
    
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved confusion matrix plot to {save_path}")
    
    plt.close()

def visualize_batch(images, labels, class_labels=None, save_path=None, max_images=16):
    """
    Visualize a batch of images with their labels.
    
    Args:
        images: Batch of images
        labels: Labels (one-hot encoded or class indices)
        class_labels: List mapping indices to class names
        save_path: Path to save the visualization
        max_images: Maximum number of images to display
    """
    # Convert one-hot encoded labels to indices if needed
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    
    # Limit the number of images
    n_images = min(len(images), max_images)
    images = images[:n_images]
    labels = labels[:n_images]
    
    # Determine grid layout
    grid_size = int(np.ceil(np.sqrt(n_images)))
    plt.figure(figsize=(12, 12))
    
    for i in range(n_images):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Handle different image formats
        img = images[i]
        if img.shape[-1] == 1:  # Grayscale
            plt.imshow(img.squeeze(), cmap='gray')
        else:  # Color
            plt.imshow(img)
        
        # Add label as title
        if class_labels is not None:
            title = class_labels[labels[i]]
        else:
            title = f"Class {labels[i]}"
        plt.title(title, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved batch visualization to {save_path}")
    
    plt.close()

def plot_model_predictions(images, true_labels, pred_labels, class_names, save_path=None, max_images=25):
    """
    Plot a grid of images with their true and predicted labels.
    
    Args:
        images: Array of images
        true_labels: Array of true labels (indices)
        pred_labels: Array of predicted labels (indices)
        class_names: List of class names
        save_path: Path to save the visualization
        max_images: Maximum number of images to display
    """
    # Limit number of images
    n_images = min(len(images), max_images)
    images = images[:n_images]
    true_labels = true_labels[:n_images]
    pred_labels = pred_labels[:n_images]
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    plt.figure(figsize=(15, 15))
    
    for i in range(n_images):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Display the image
        img = images[i]
        if img.shape[-1] == 1:  # Grayscale
            plt.imshow(img.squeeze(), cmap='gray')
        else:  # Color
            plt.imshow(img)
        
        # Create title with true and predicted labels
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        
        # Color code the prediction (green for correct, red for wrong)
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        
        plt.title(f"True: {true_class}\nPred: {pred_class}", 
                 color=color, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved prediction visualization to {save_path}")
    
    plt.close()
