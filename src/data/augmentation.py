import tensorflow as tf
import yaml
from src.utils.logger import get_logger

logger = get_logger("2_data_augmentation")

def load_augmentation_config():
    """
    Load augmentation configuration settings from the YAML file.
    """
    logger.info("Loading augmentation config from YAML file...")
    with open("configs/data_config.yaml") as f:
        cfg = yaml.safe_load(f)['augmentation']
    logger.info("Augmentation config loaded successfully :)")
    return cfg

def augment(image, label):
    """
    Apply data augmentation to the input image based on configuration parameters.

    The function performs a series of augmentation operations:
      - Rotation: Rotates the image by a random multiple of 90 degrees.
      - Brightness Adjustment: Alters the brightness by a random delta.
      - Contrast Adjustment: Modifies the image contrast within specified bounds.
      - Saturation Adjustment: Changes the image saturation between given limits.
      - Horizontal Flip: Randomly flips the image left-to-right.

    Each operation is applied probabilistically based on the corresponding configuration values.
    """
    cfg = load_augmentation_config()

    # If augmentation is disabled in the configuration, skip further processing.
    if not cfg.get("enable", False):
        logger.info("Augmentation is disabled in config, skipping augmentation steps.")
        return image, label

    # =========================
    # Rotation Augmentation
    # =========================
    # rotate the image by a random multiple of 90 degrees.
    if tf.random.uniform([]) < cfg['rotation_prob']:
        # Generates a random integer value between 1 and 3 (inclusive).
        rotations = tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32)
        
        # Rotates the image by 90 degrees a random number of times.
        image = tf.image.rot90(image, k=rotations)
        logger.debug('Rotation augmentation applied with {} rotations.'.format(rotations.numpy()))

    # =========================
    # Brightness Adjustment
    # =========================
    # adjust the image brightness.
    if tf.random.uniform([]) < cfg['brightness_prob']:
        image = tf.image.random_brightness(image, max_delta=cfg['brightness_max_delta'])
        logger.debug('Brightness augmentation applied.')

    # =========================
    # Contrast Adjustment
    # =========================
    if tf.random.uniform([]) < cfg['contrast_prob']:
        # Randomly adjust contrast with a factor between 'contrast_lower' and 'contrast_upper'.
        image = tf.image.random_contrast(image, lower=cfg['contrast_lower'], upper=cfg['contrast_upper'])
        logger.debug('Contrast augmentation applied.')

    # =========================
    # Saturation Adjustment
    # =========================
    if tf.random.uniform([]) < cfg['saturation_prob']:
        image = tf.image.random_saturation(image, lower=cfg['saturation_lower'], upper=cfg['saturation_upper'])
        logger.debug('Saturation augmentation applied.')

    # =========================
    # Horizontal Flip
    # =========================
    if tf.random.uniform([]) < cfg['flip_prob']:
        image = tf.image.random_flip_left_right(image)
        logger.debug('Horizontal flip augmentation applied.')

    return image, label