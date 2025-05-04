# step 1: project configuration

## install pipx and Install Poetry with pipx

pipx allows for isolated installation of Python applications.
Poetry is our dependency management tool

```
pip install pipx
pipx ensurepath

pipx install poetry
poetry --version

poetry init
// Add Project Dependencies
# Core dependencies
poetry add tensorflow scikit-learn PyYAML matplotlib Pillow

# Development dependencies
poetry add pytest ipython --group dev
```

## Create Virtual Environment

Create virtual environment and install dependencies

```
poetry install
poetry lock
poetry env activate


# Check installed packages
poetry show

# Add new packages
poetry add package-name

# Update dependencies
poetry update

# Export requirements.txt (if needed)
poetry export -f requirements.txt --output requirements.txt
```

# Step 2: Configuration Management

## Creating Configuration Files

Configuration files help separate code from parameters, making the project more maintainable and flexible.

1. Create a `configs` directory for YAML configuration files including data_config and model_config

# Step 3: Data Loading Implementation

In this step, we implement a data loader to process MRI images for our Alzheimer's classification model.

## Data Organization

Organize your ADNI dataset in the following structure:
```
Total: 86437, Val: 17287, Test: 8643, Train: 60507
```

# Step 4: Data Augmentation Implementation

In this step, we implement data augmentation to increase the diversity of the training data by applying random transformations to images. These augmentations help improve model robustness by simulating variations in real-world conditions.

The augmentation module performs a series of random transformations:
- **Rotation:** Randomly rotates the image by a multiple of 90° 
 
- **Brightness Adjustment:** Randomly adjusts the brightness of the image

- **Contrast, Saturation, and Horizontal Flip:** Further augmentations include adjusting image contrast, saturation, and applying random horizontal flips, as specified in the configuration file.

Configuration for these augmentations is managed through a YAML file (`configs/data_config.yaml`), which allows easy modification of probabilities and intensity of each augmentation.

These steps ensure that the model is exposed to a variety of image conditions during training, thereby enhancing its generalization capabilities.

# Step 5: Data Preprocessing Implementation

The preprocessing module (located at `src/data/preprocessing.py`) handles image loading and preprocessing tasks. Key functionalities include:

- **Preprocessor Class**:  
  This class encapsulates the image preprocessing pipeline. Its main methods are:
  - `load_image(file_path)`: Loads an image using OpenCV, and logs an error if the image cannot be loaded.
  - `resize_image(image)`: Resizes the image to target dimensions specified during initialization (default or from config).
  - `normalize_image(image)`: Normalizes image pixel values to the range [0, 1].
  - `preprocess_image(file_path)`: A helper that combines the above steps to load, resize, and normalize an image.