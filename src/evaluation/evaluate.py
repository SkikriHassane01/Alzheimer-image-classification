import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef

from src.utils.logger import get_logger
from src.utils.visualization import plot_confusion_matrix, plot_training_history

logger = get_logger("model_evaluation")

class ModelEvaluator:
    """Handles model evaluation and metrics calculation."""

    def __init__(self, model, test_generator, class_indices=None, class_descriptions=None):
        self.model = model
        self.test_generator = test_generator
        
        # If not provided, try to get from the generator
        self.class_indices = class_indices or getattr(test_generator, 'class_indices', {})
        
        # Create reverse mapping (idx -> label)
        self.idx_to_label = {v: k for k, v in self.class_indices.items()} if self.class_indices else {}
        self.class_descriptions = class_descriptions or {}
    
    def evaluate(self):
        """Evaluate model performance on the test set."""
        logger.info("Evaluating model on test data...")
        
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def predict_and_analyze(self, save_dir="results"):
        """Make predictions and perform detailed analysis."""
        logger.info("Making predictions for detailed analysis...")
        
        # Reset the generator to ensure we start from the beginning
        self.test_generator.reset()
        
        # Make predictions
        y_pred_prob = self.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = self.test_generator.classes[:len(y_pred)]  # Ensure same length
        
        # Get class names with descriptions for better readability
        class_names = [self.idx_to_label.get(idx, f"Class {idx}") for idx in range(len(self.class_indices))]
        class_names_with_desc = [
            f"{label} ({self.class_descriptions.get(label, 'Unknown')})" 
            for label in class_names
        ]
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            cm, 
            class_labels=class_names_with_desc, 
            save_path=f"{save_dir}/confusion_matrix.png"
        )
        
        # Generate and log classification report
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=class_names_with_desc, 
            zero_division=0
        )
        logger.info(f"Classification Report:\n{report}")
        
        # Calculate Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        logger.info(f"Matthews Correlation Coefficient: {mcc:.4f}")
        
        # Return metrics for further use if needed
        metrics = {
            'accuracy': (y_true == y_pred).mean(),
            'mcc': mcc,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return metrics
