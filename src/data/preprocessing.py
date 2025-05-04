import os
import cv2
import yaml
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("3_preprocessing")

def load_data_config(config_path="configs/data_config.yaml"):
    """
    Loads the data configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        logger.error("Configuration file not found at %s. Using default settings.", config_path)
        return {}
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        logger.error("Error loading configuration file: %s", e)
        return {}
    
DATA_CONFIG = load_data_config()

class Preprocessor:
    """
    A class used to preprocess images.

    Attributes:
        img_size (tuple): The target image size (width, height).
    """
    def __init__(self, img_size=None):
        if img_size is None:
            width = DATA_CONFIG.get("img_width", 224)
            height = DATA_CONFIG.get("img_height", 224)
            img_size = (width, height)
            logger.info("Using default image size from config: %s", img_size)
        self.img_size = img_size

    def load_image(self, file_path: str) -> np.ndarray:
        """
        Loads an image from the given file path.
        Returns:
            np.ndarray: The loaded image array, or None if loading fails.
        """
        logger.info("Loading image from: %s", file_path)
        image = cv2.imread(file_path)
        if image is None:
            logger.error("Failed to load image from: %s", file_path)
        return image

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes the image to the target size.

        Args:
            image (np.ndarray): Image array.

        Returns:
            np.ndarray: Resized image.
        """
        logger.info("Resizing image to dimensions: %s", self.img_size)
        return cv2.resize(image, self.img_size)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizes the image array.
        Returns:
            np.ndarray: Normalized image array.
        """
        logger.info("Normalizing image pixel values to [0, 1]")
        return image.astype(np.float32) / 255.0

    def preprocess_image(self, file_path: str) -> np.ndarray:
        """
        Preprocesses an image: loads, resizes, and normalizes.
        Returns:
            np.ndarray: Preprocessed image array, or None if image load fails.
        """
        image = self.load_image(file_path)
        if image is None:
            return None
        image = self.resize_image(image)
        image = self.normalize_image(image)
        return image

if __name__ == "__main__":

    # Example usage
    preprocessor = Preprocessor()
    image_path = r"Data\test\OAS1_0028_MR1_mpr-1_100.jpg"
    preprocessed_image = preprocessor.preprocess_image(image_path)
    if preprocessed_image is not None:
        logger.info("Preprocessed image shape: %s", preprocessed_image.shape)
    else:
        logger.error("Failed to preprocess image.")
        