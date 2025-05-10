import os
import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
from tensorflow.keras.applications import DenseNet169, DenseNet201, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Lambda

logger = logging.getLogger("ModelFactory")

class ModelFactory:
    """Factory class to build or load models based on configuration."""

    def __init__(self, config_path='configs/model_config.yaml'):
        self.logger = logger
        self.config = self._load_config(config_path)
        self.backbone = self.config.get('backbone', 'DenseNet201')
        self.input_shape = tuple(self.config.get('input_shape', [224, 224, 3]))
        self.num_classes = self.config.get('num_classes', 4)
    
    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Failed to load model configuration: {e}")
            return {}
    
    def build_model(self):
        """Build a model based on the configuration."""
        self.logger.info(f"Building model with backbone: {self.backbone}")
        
        # Get original input shape info
        input_height, input_width, input_channels = self.input_shape
        
        # Create input tensor based on original shape
        input_tensor = Input(shape=self.input_shape)
        
        # Handle grayscale input - convert to 3 channels if needed
        if input_channels == 1:
            self.logger.info("Converting grayscale input to RGB for transfer learning compatibility")
            x = Lambda(lambda x: tf.repeat(x, repeats=3, axis=-1), name="grayscale_to_rgb")(input_tensor)
        else:
            x = input_tensor
        
        # Select the backbone architecture
        if self.backbone == 'DenseNet169':
            base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(input_height, input_width, 3))
        elif self.backbone == 'DenseNet201':
            base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(input_height, input_width, 3))
        elif self.backbone == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(input_height, input_width, 3))
        else:
            self.logger.warning(f"Unknown backbone: {self.backbone}, using DenseNet201 instead")
            base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(input_height, input_width, 3))
            self.backbone = 'DenseNet201'
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Connect the input to the base model
        x = base_model(x, training=False)
        
        # Add classification layers
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the complete model
        model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name=f"{self.backbone}_model")
        
        self.logger.info(f"Successfully built {self.backbone} model")
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
                self.logger.warning(f"Checkpoint path does not exist: {model_path}")
                continue
                
            try:
                loaded_model = load_model(model_path)
                self.logger.info(f"Successfully loaded model from {model_path}")
                loaded_from_checkpoint = True
                break
            except Exception as e:
                self.logger.error(f"Failed to load model from {model_path}: {e}")
        
        # If direct loading failed, try rebuilding and loading weights
        if not loaded_from_checkpoint:
            self.logger.info("Building new model architecture...")
            loaded_model = self.build_model()
            
            # Try to load weights
            for path in checkpoint_paths:
                if not os.path.exists(path):
                    continue
                    
                try:
                    loaded_model.load_weights(path)
                    self.logger.info(f"Successfully loaded weights from {path}")
                    loaded_from_checkpoint = True
                    break
                except Exception as e:
                    self.logger.error(f"Failed to load weights from {path}: {e}")
        
        if not loaded_from_checkpoint:
            self.logger.warning("Could not load any saved weights. Starting from ImageNet weights.")
            
        return loaded_model, loaded_from_checkpoint
