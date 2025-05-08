import os
import yaml
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from src.utils.logger import get_logger
from src.utils.visualization import plot_training_history

logger = get_logger("model_trainer")

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
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.fine_tune_lr = self.config.get('fine_tune_lr', 1e-5)
        self.epochs = self.config.get('epochs', 20)
        self.early_stop_patience = self.config.get('early_stop_patience', 10)
        self.reduce_lr_patience = self.config.get('reduce_lr_patience', 5)
        self.reduce_lr_factor = self.config.get('reduce_lr_factor', 0.2)
        self.min_lr = self.config.get('min_lr', 1e-7)
        
        # Results directory
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
        
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
        # Determine model name and stage
        model_name = getattr(model, 'name', 'model')
        stage = "fine_tuning" if fine_tuning else "initial"
        
        logger.info(f"Starting {stage} training for {model_name}...")
        
        # If not fine-tuning, compile with initial learning rate
        if not fine_tuning:
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
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
        
        return model, history, final_model_path
