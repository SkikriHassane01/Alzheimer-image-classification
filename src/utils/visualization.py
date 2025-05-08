import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from src.utils.logger import get_logger

logger = get_logger("visualization")

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
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved confusion matrix plot to {save_path}")
    
    plt.close()
