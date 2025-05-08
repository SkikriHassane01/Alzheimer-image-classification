"""
load_dataset() scans a directory structure containing image
files and prepares them for machine learning by extracting paths,
labels, and organization information.
"""
import os
import glob
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from src.utils.logger import get_logger

logger = get_logger("1_data_loader")

def load_yaml_config(file_path):
    """
    Loads a YAML configuration file.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {file_path} successfully.")
            return config
    except Exception as e:
        logger.error(f"Configuration file {file_path} not found. Error: {e}")
        return {}

def load_dataset(dataset_dir, new_class_structure, class_descriptions):
    """
    Loads image paths and labels from the dataset directory.
    Returns a DataFrame with columns: path, label, original_label, split.
    """
    image_data = []
    if os.path.exists(os.path.join(dataset_dir, 'train')) or os.path.exists(os.path.join(dataset_dir, 'test')):
        splits = ['train', 'test']
    else:
        splits = ['all']
    for split in splits:
        base_path = dataset_dir if split == 'all' else os.path.join(dataset_dir, split)
        if not os.path.exists(base_path):
            logger.warning(f"Warning: {base_path} does not exist!")
            continue
        logger.info(f"Processing {split} set...")
        for new_class, original_class in new_class_structure.items():
            class_path = os.path.join(base_path, new_class)
            if not os.path.exists(class_path):
                logger.warning(f"Warning: {class_path} does not exist!")
                continue
            image_paths = glob.glob(os.path.join(class_path, '*.jpg')) + \
                          glob.glob(os.path.join(class_path, '*.png')) + \
                          glob.glob(os.path.join(class_path, '*.jpeg')) + \
                          glob.glob(os.path.join(class_path, '*.tif')) + \
                          glob.glob(os.path.join(class_path, '*.tiff'))
            for img_path in image_paths:
                image_data.append({
                    'path': img_path,
                    'label': new_class,
                    'original_label': original_class,
                    'split': split
                })
            logger.info(f"Added {len(image_paths)} images from class {new_class} -> {original_class} ({split})")
    df = pd.DataFrame(image_data)
    if df.empty:
        logger.error("ERROR: No images found. Please check dataset structure and paths.")
        return df
    else:
        logger.info(f"Successfully loaded {len(df)} images.")
        df['original_class_description'] = df['original_label'].map(class_descriptions)
        return df

if __name__ == "__main__":
    try:
        config = load_yaml_config("configs/data_config.yaml")
        dataset_dir = config.get("dataset_dir", "Data/raw")
        new_class_structure = config.get("new_class_structure",)
        class_descriptions = config.get("class_descriptions")
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        dataset_dir = "Data/raw"
        new_class_structure = {
            "Non Demented": "CN",
            "Mild Dementia": "LMCI",
            "Moderate Dementia": "AD",
            "Very Mild Dementia": "EMCI"
        }
        class_descriptions = {
            "AD": "Alzheimer's Disease",
            "CN": "Cognitive Normal",
            "EMCI": "Early Mild Cognitive Impairment",
            "LMCI": "Late Mild Cognitive Impairment"
        }
    df = load_dataset(dataset_dir, new_class_structure, class_descriptions)
    if not df.empty:
        logger.info(f"DataFrame head:\n{df.head()}")
        # Save the dataframe into a CSV file
        os.makedirs("Data/processed", exist_ok=True)
        csv_path = os.path.join("Data/processed", "dataset.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"DataFrame saved to {csv_path}")
    else:
        logger.error("No images loaded; please verify your dataset structure.")