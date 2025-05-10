import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
import tensorflow as tf
from PIL import Image
import pandas as pd

from src.utils.logger import get_logger
from src.utils.visualization import plot_confusion_matrix, plot_training_history

logger = get_logger("7_model_evaluation")

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
        
        # Debug settings
        self.debug = True
        self.debug_samples = 20  # Number of samples to analyze in detail
    
    def evaluate(self):
        """Evaluate model performance on the test set."""
        logger.info("Evaluating model on test data...")
        
        # Debug info about test generator
        logger.info(f"Test generator information:")
        logger.info(f"  - Class indices: {self.class_indices}")
        logger.info(f"  - Number of samples: {len(self.test_generator.filenames)}")
        logger.info(f"  - Batch size: {self.test_generator.batch_size}")
        logger.info(f"  - Image shape: {self.test_generator.image_shape}")
        logger.info(f"  - Class counts: {np.bincount(self.test_generator.classes)}")
        
        # Ensure the model is compiled before evaluation
        if not getattr(self.model, 'optimizer', None):
            logger.info("Model not compiled. Compiling with default settings for evaluation...")
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # For debugging, inspect model architecture
        if self.debug:
            logger.info("Model architecture:")
            self.model.summary(print_fn=logger.info)
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def predict_and_analyze(self, save_dir="results"):
        """Make predictions and perform detailed analysis."""
        logger.info("Making predictions for detailed analysis...")
        os.makedirs(save_dir, exist_ok=True)
        
        # Reset the generator to ensure we start from the beginning
        self.test_generator.reset()
        
        # For debugging, inspect the first batch of images
        if self.debug:
            try:
                # Get a batch of images and check their values
                x_batch, y_batch = next(self.test_generator)
                logger.info(f"First batch shape: {x_batch.shape}")
                logger.info(f"Labels shape: {y_batch.shape}")
                logger.info(f"Image value range: min={np.min(x_batch)}, max={np.max(x_batch)}")
                logger.info(f"Sample true labels (first 5): {np.argmax(y_batch[:5], axis=1)}")
                
                # Save a few sample images for inspection
                debug_dir = os.path.join(save_dir, "debug_samples")
                os.makedirs(debug_dir, exist_ok=True)
                
                for i in range(min(5, len(x_batch))):
                    plt.figure(figsize=(5, 5))
                    img = x_batch[i]
                    # If image is single channel (grayscale)
                    if img.shape[-1] == 1:
                        plt.imshow(img.squeeze(), cmap='gray')
                    else:
                        plt.imshow(img)
                    true_label = self.idx_to_label.get(np.argmax(y_batch[i]), "Unknown")
                    plt.title(f"True label: {true_label}")
                    plt.axis('off')
                    plt.savefig(os.path.join(debug_dir, f"sample_{i}.png"))
                    plt.close()
            except Exception as e:
                logger.error(f"Error during debug image inspection: {e}")
        
        # Make predictions
        try:
            # Collect all test filenames and true labels for detailed analysis
            all_files = self.test_generator.filepaths
            all_true_labels = self.test_generator.classes
            logger.info(f"Starting predictions on {len(all_files)} test samples")
            
            # Standard prediction on the generator
            y_pred_prob = self.model.predict(self.test_generator, verbose=1)
            y_pred = np.argmax(y_pred_prob, axis=1)
            
            # Check if prediction length matches expected test size
            if len(y_pred) != len(all_true_labels):
                logger.warning(f"Warning: Prediction length ({len(y_pred)}) doesn't match test set length ({len(all_true_labels)})")
                # Adjust to the shorter length to avoid index errors
                n_samples = min(len(y_pred), len(all_true_labels))
                y_pred = y_pred[:n_samples]
                y_true = all_true_labels[:n_samples]
                all_files = all_files[:n_samples]
            else:
                y_true = all_true_labels
            
            # Log detailed statistics
            logger.info(f"Predictions complete. Shape: {y_pred_prob.shape}")
            logger.info(f"True label distribution: {np.bincount(y_true)}")
            logger.info(f"Predicted label distribution: {np.bincount(y_pred)}")
            
            # Calculate accuracy
            accuracy = (y_true == y_pred).mean()
            logger.info(f"Overall accuracy: {accuracy:.4f}")
            
            # Create detailed prediction analysis dataframe
            results_df = pd.DataFrame({
                'filepath': all_files,
                'true_label_idx': y_true,
                'true_label': [self.idx_to_label.get(idx, f"Unknown_{idx}") for idx in y_true],
                'pred_label_idx': y_pred,
                'pred_label': [self.idx_to_label.get(idx, f"Unknown_{idx}") for idx in y_pred],
                'is_correct': y_true == y_pred
            })
            
            # Add confidence scores for each class
            for idx, label in self.idx_to_label.items():
                results_df[f'conf_{label}'] = y_pred_prob[:, idx]
            
            # Save detailed results
            results_df.to_csv(os.path.join(save_dir, 'detailed_predictions.csv'), index=False)
            logger.info(f"Detailed prediction results saved to {os.path.join(save_dir, 'detailed_predictions.csv')}")
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            class_names = [self.idx_to_label.get(i, f"Class_{i}") for i in range(len(self.class_indices))]
            
            # Plot and save confusion matrix
            plt.figure(figsize=(10, 8))
            plot_confusion_matrix(cm, class_names)
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
            plt.close()
            
            # Generate classification report
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(save_dir, 'classification_report.csv'))
            
            # Calculate Matthews Correlation Coefficient
            mcc = matthews_corrcoef(y_true, y_pred)
            logger.info(f"Matthews Correlation Coefficient: {mcc:.4f}")
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report,
                'mcc': mcc,
                'results_df': results_df
            }
            
        except Exception as e:
            logger.error(f"Error during prediction analysis: {str(e)}")
            raise
        finally:
            logger.info("Prediction analysis completed")
