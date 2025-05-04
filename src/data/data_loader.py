"""
Loads ADNI MRI images into TensorFlow datasets and splits 
into train/validation/test sets.
"""

import tensorflow as tf
import yaml
from src.utils.logger import get_logger

logger = get_logger("1_data_loader")

def load_data_config():
    logger.info('Loading data_config.yaml')
    with open('configs/data_config.yaml') as f:
        return yaml.safe_load(f)


def get_datasets():
    cfg = load_data_config()
    logger.info('Creating TF dataset from %s', cfg['data_dir'])
    ds = tf.keras.utils.image_dataset_from_directory(
        cfg['data_dir'], labels='inferred', label_mode='categorical',
        color_mode='grayscale', batch_size=None,
        image_size=(cfg['img_height'], cfg['img_width']), shuffle=True)

    total = ds.cardinality().numpy()
    n_val = int(cfg['validation_split'] * total)
    n_test = int(cfg['test_split'] * total)
    logger.info('Total: %d, Val: %d, Test: %d, Train: %d', total, n_val, n_test, total - n_val - n_test)

    val_ds = ds.take(n_val)
    test_ds = ds.skip(n_val).take(n_test)
    train_ds = ds.skip(n_val + n_test)
    logger.info('Datasets prepared')
    return train_ds, val_ds, test_ds

# if __name__ == "__main__":
#     train_ds, val_ds, test_ds = get_datasets()
#     print('Train dataset: %s', train_ds)
#     print('Validation dataset: %s', val_ds)
#     print('Test dataset: %s', test_ds)