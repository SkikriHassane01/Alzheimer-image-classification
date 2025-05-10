import os
import yaml
from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam

from src.utils.logger import get_logger
from src.utils.visualization import plot_training_history, visualize_batch

logger = get_logger("5_model_trainer")

class ModelTrainer:
    """Class for handling model training with appropriate configurations."""

    def __init__(self, config_path='configs/train_config.yaml', output_dir='models/checkpoints'):
        """
        Initialize the model trainer.
        
        Args:
            config_path: Path to the training configuration file
            output_dir: Directory to save model checkpoints
        """
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default training settings.")
            self.config = {
                'learning_rate': 1e-4,
                'fine_tune_lr': 1e-5,
                'epochs': 20,
                'batch_size': 32,
                'early_stop_patience': 10,
                'reduce_lr_patience': 5,
                'reduce_lr_factor': 0.2,
                'min_lr': 1e-7,
            }
        
        # Setup output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Training parameters
        self.learning_rate = float(self.config.get('learning_rate', 1e-4))
        self.fine_tune_lr = float(self.config.get('fine_tune_lr', 1e-5))
        self.epochs = self.config.get('epochs', 20)
        self.early_stop_patience = self.config.get('early_stop_patience', 10)
        self.reduce_lr_patience = self.config.get('reduce_lr_patience', 5)
        self.reduce_lr_factor = self.config.get('reduce_lr_factor', 0.2)
        self.min_lr = float(self.config.get('min_lr', 1e-7))
        
        # Results directory
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Add debug flag
        self.debug = True
        
        logger.info(f"ModelTrainer initialized with learning rate: {self.learning_rate}")
    
    def _create_callbacks(self, model_name, stage="training"):
        """
        Create training callbacks.
        
        Args:
            model_name: Name of the model (e.g., "DenseNet201")
            stage: Training stage (e.g., "training" or "fine_tuning")
            
        Returns:
            list: List of keras callbacks
        """
        checkpoint_path = os.path.join(
            self.output_dir, 
            f"{model_name}_{stage}_{self.timestamp}_best.keras"
        )
        
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                patience=self.early_stop_patience,
                monitor='val_loss',
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                factor=self.reduce_lr_factor,
                patience=self.reduce_lr_patience,
                min_lr=self.min_lr,
                monitor='val_loss',
                verbose=1
            )
        ]
        
        # Add TensorBoard callback for better debugging
        log_dir = os.path.join('logs', f"{model_name}_{stage}_{self.timestamp}")
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,  # Log histogram every epoch
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0  # Disable profiling to reduce memory usage
        )
        
        callbacks.append(tensorboard_callback)
        logger.info(f"TensorBoard logs will be saved to {log_dir}")
        
        return callbacks
    
    def prepare_for_fine_tuning(self, model):
        """
        Prepare model for fine-tuning by unfreezing selected layers.
        
        Args:
            model: The keras model to prepare for fine-tuning
            
        Returns:
            model: The prepared model
        """
        logger.info("Preparing model for fine-tuning...")
        
        # Unfreeze specific layers for fine-tuning
        for layer in model.layers:
            # For the base model (packaged inside our model)
            if isinstance(layer, tf.keras.models.Model):
                for base_layer in layer.layers:
                    if any(x in base_layer.name for x in ['conv4', 'conv5', 'pool']):
                        base_layer.trainable = True
                        logger.info(f"Unfreezing layer: {base_layer.name}")
                    else:
                        base_layer.trainable = False
        
        # Recompile the model with a lower learning rate for fine-tuning
        model.compile(
            optimizer=Adam(learning_rate=self.fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Show the model layers and trainable status
        for layer in model.layers:
            if isinstance(layer, tf.keras.models.Model):
                logger.info(f"Base model has {len(layer.layers)} layers")
                logger.info(f"Trainable weights: {len(layer.trainable_weights)}")
                logger.info(f"Non-trainable weights: {len(layer.non_trainable_weights)}")
        
        return model
    
    def train(self, model, train_generator, val_generator, fine_tuning=False):
        """
        Train the model with the given data generators.
        
        Args:
            model: The keras model to train
            train_generator: Training data generator
            val_generator: Validation data generator
            fine_tuning: Whether this is a fine-tuning run
            
        Returns:
            tuple: (trained_model, history, model_path)
        """
        # Determine model name and training stage
        model_name = getattr(model, 'name', 'model')
        stage = "fine_tuning" if fine_tuning else "initial"
        logger.info(f"Starting {stage} training for {model_name}...")
        
        # Debug information about generators
        if self.debug:
            logger.info("DEBUG: Training generator information:")
            logger.info(f"  - Batch size: {train_generator.batch_size}")
            logger.info(f"  - Image shape: {train_generator.image_shape}")
            logger.info(f"  - Number of samples: {len(train_generator.filenames)}")
            logger.info(f"  - Class indices: {train_generator.class_indices}")
            logger.info(f"  - Class counts: {np.bincount(train_generator.classes)}")
            
            # Visualize a sample batch to verify data
            debug_dir = os.path.join(self.results_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            try:
                # Get a batch of images
                x_batch, y_batch = next(train_generator)
                # Check for NaNs or infinity values
                has_nan = np.isnan(x_batch).any()
                has_inf = np.isinf(x_batch).any()
                logger.info(f"DEBUG: Batch contains NaN values: {has_nan}")
                logger.info(f"DEBUG: Batch contains Inf values: {has_inf}")
                
                # Visualize a batch
                idx_to_label = {v: k for k, v in train_generator.class_indices.items()}
                class_labels = [idx_to_label.get(i, f"Unknown_{i}") for i in range(len(idx_to_label))]
                
                # Check if one-hot encoded or class indices
                if len(y_batch.shape) > 1 and y_batch.shape[1] > 1:  # One-hot encoded
                    labels_idx = np.argmax(y_batch, axis=1)
                else:  # Class indices
                    labels_idx = y_batch
                
                # Log some sample values
                logger.info(f"DEBUG: Batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")
                logger.info(f"DEBUG: Image value range: min={np.min(x_batch)}, max={np.max(x_batch)}")
                logger.info(f"DEBUG: Sample labels (first 5): {labels_idx[:5]}")
                
                # Save batch visualization
                visualize_batch(
                    x_batch, 
                    labels_idx, 
                    class_labels=class_labels, 
                    save_path=os.path.join(debug_dir, f"train_batch_sample_{stage}.png")
                )
            except Exception as e:
                logger.error(f"Error during batch visualization: {e}")
        
        # If not fine-tuning, compile with initial learning rate
        if not fine_tuning:
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Log model summary
        model.summary(print_fn=logger.info)
        
        # Get callbacks
        callbacks = self._create_callbacks(model_name, stage)
        
        # Train the model
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(
            self.output_dir, 
            f"{model_name}_{stage}_final_{self.timestamp}.keras"
        )
        model.save(final_model_path)
        logger.info(f"Saved final {stage} model to {final_model_path}")
        
        # Plot and save training history
        plot_path = os.path.join(self.results_dir, f"training_history_{stage}_{self.timestamp}.png")
        plot_training_history(history, save_path=plot_path)
        
        # Additional debug validation after training
        if self.debug:
            logger.info("Running post-training validation check...")
            val_loss, val_acc = model.evaluate(val_generator, verbose=1)
            logger.info(f"Post-training validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            
            # Verify model predictions on validation data
            try:
                val_generator.reset()
                x_val, y_val = next(val_generator)
                y_pred = model.predict(x_val)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_val, axis=1)
                
                logger.info(f"DEBUG: Sample validation prediction check:")
                logger.info(f"  - True classes: {y_true_classes[:5]}")
                logger.info(f"  - Predicted classes: {y_pred_classes[:5]}")
                logger.info(f"  - Sample probabilities: {y_pred[0]}")
            except Exception as e:
                logger.error(f"Error during validation prediction check: {e}")
        
        return model, history, final_model_path
