import os
import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.utils.logger import get_logger
from src.models.transfer_models import build_densenet169_model, build_densenet201_model

logger = get_logger("model_factory")

class ModelFactory:
    """Factory class to build or load models based on configuration."""

    def __init__(self, config_path='configs/model_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.input_shape = tuple(self.config['input_shape'])
        self.num_classes = self.config['num_classes']
        self.backbone = self.config['backbone']
        self.model_registry = {
            'DenseNet169': build_densenet169_model,
            'DenseNet201': build_densenet201_model
        }
    
    def build_model(self):
        """Build a new model based on configuration."""
        if self.backbone not in self.model_registry:
            logger.error(f"Backbone {self.backbone} not supported. Using DenseNet201 instead.")
            self.backbone = "DenseNet201"
        
        logger.info(f"Building {self.backbone} model with input shape {self.input_shape} and {self.num_classes} classes")
        model_fn = self.model_registry[self.backbone]
        model = model_fn(input_shape=self.input_shape, num_classes=self.num_classes)
        
        return model
    
    def load_model(self, checkpoint_paths=None):
        """
        Load model from checkpoint if available, otherwise build a new model.
        Returns both the model and a boolean indicating if the model was loaded from checkpoint.
        """
        if checkpoint_paths is None:
            checkpoint_paths = []
            
        loaded_model = None
        loaded_from_checkpoint = False
        
        # Try to load the complete model
        for model_path in checkpoint_paths:
            if not os.path.exists(model_path):
                logger.warning(f"Checkpoint path does not exist: {model_path}")
                continue
                
            try:
                loaded_model = load_model(model_path)
                logger.info(f"Successfully loaded model from {model_path}")
                loaded_from_checkpoint = True
                break
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
        
        # If direct loading failed, try rebuilding and loading weights
        if not loaded_from_checkpoint:
            logger.info("Building new model architecture...")
            loaded_model = self.build_model()
            
            # Try to load weights
            for path in checkpoint_paths:
                if not os.path.exists(path):
                    continue
                    
                try:
                    loaded_model.load_weights(path)
                    logger.info(f"Successfully loaded weights from {path}")
                    loaded_from_checkpoint = True
                    break
                except Exception as e:
                    logger.error(f"Failed to load weights from {path}: {e}")
        
        if not loaded_from_checkpoint:
            logger.warning("Could not load any saved weights. Starting from ImageNet weights.")
            
        return loaded_model, loaded_from_checkpoint
