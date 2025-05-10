import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet201, DenseNet169

def build_densenet201_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Builds the DenseNet201 model using ImageNet weights without the top layers,
    adds custom classification layers, and selectively freezes layers.
    """
    # Load pre-trained DenseNet201 without top layers
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Initially freeze layers until reaching later convolution blocks
    trainable = False
    for layer in base_model.layers:
        if 'conv5' in layer.name or 'pool5' in layer.name:
            trainable = True
        layer.trainable = trainable

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_densenet169_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Builds the DenseNet169 model using ImageNet weights without the top layers,
    adds custom classification layers, and selectively freezes layers.
    """
    # Load pre-trained DenseNet169 without top layers
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Initially freeze layers until reaching later convolution blocks
    trainable = False
    for layer in base_model.layers:
        if 'conv5' in layer.name or 'pool5' in layer.name:
            trainable = True
        layer.trainable = trainable

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_resnet50_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Build and return a ResNet50 model with the given input shape and number of classes.
    
    Args:
        input_shape: Tuple of integers defining the input shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        A compiled Keras model
    """
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    
    # Create the base pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model