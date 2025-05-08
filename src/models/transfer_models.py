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