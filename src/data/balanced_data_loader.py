import os
import uuid
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
import yaml
import matplotlib.pyplot as plt
from src.utils.logger import get_logger

logger = get_logger("2_balancing_data_and_processing")


def balance_dataset(
    df: pd.DataFrame,
    path_col: str,
    label_col: str,
    target_samples_per_class: int = 20000,
    augment: bool = True,
    augment_params: dict = None,
    output_dir: str = 'Data/processed/augmented',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Balance the dataset by downsampling majority classes and oversampling minority classes.
    For minority classes, generate augmented images using ImageDataGenerator.

    Returns a DataFrame with columns:
      - path_col: file path to image
      - label_col: class label
      - original_label: encoded label index
      - original_class_description: original label string
    """
    np.random.seed(random_state)
    df_copy = df.copy().reset_index(drop=True)
    le = LabelEncoder()
    df_copy['original_label'] = le.fit_transform(df_copy[label_col])
    df_copy['original_class_description'] = df_copy[label_col]

    if augment:
        params = augment_params or {
            'rotation_range': 30,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'brightness_range': [0.7, 1.3],
            'zoom_range': 0.2,
            'horizontal_flip': True
        }
        aug_gen = ImageDataGenerator(**params)
        os.makedirs(output_dir, exist_ok=True)

    balanced_parts = []
    for cls in le.classes_:
        cls_df = df_copy[df_copy[label_col] == cls]
        count = len(cls_df)
        if count >= target_samples_per_class:
            part = cls_df.sample(n=target_samples_per_class, random_state=random_state)
            logger.info(f"Downsampled '{cls}': {count} -> {target_samples_per_class}")
        else:
            part = cls_df.copy()
            needed = target_samples_per_class - count
            logger.info(f"Augmenting '{cls}': {count} -> {target_samples_per_class} (need {needed})")
            if augment:
                cls_out_dir = os.path.join(output_dir, cls)
                os.makedirs(cls_out_dir, exist_ok=True)
                samples = cls_df.to_dict('records')
                for i in range(needed):
                    sample = np.random.choice(samples)
                    img_path = sample[path_col]
                    try:
                        img = load_img(img_path)
                    except Exception as e:
                        logger.warning(f"Failed loading {img_path}: {e}")
                        continue
                    arr = img_to_array(img)
                    arr = arr.reshape((1,) + arr.shape)
                    batch = next(aug_gen.flow(arr, batch_size=1))
                    new_img = batch[0].astype('uint8')
                    fname = f"{uuid.uuid4().hex}.png"
                    out_path = os.path.join(cls_out_dir, fname)
                    save_img(out_path, new_img)
                    row = sample.copy()
                    row[path_col] = out_path
                    row_df = pd.DataFrame([row])
                    part = pd.concat([part, row_df], ignore_index=True)
            else:
                extra = cls_df.sample(n=needed, replace=True, random_state=random_state)
                part = pd.concat([part, extra], ignore_index=True)
        balanced_parts.append(part)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    out_csv = 'Data/processed/balanced_dataset.csv'
    balanced_df.to_csv(out_csv, index=False)
    logger.info(f"Saved balanced dataset: {len(balanced_df)} rows to {out_csv}")
    return balanced_df


def create_data_generators(
    csv_path: str,
    path_col: str,
    label_col: str,
    img_size: int,
    batch_size: int,
    label_order: list = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    random_state: int = 42,
    debug: bool = True
):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n = len(df)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    df['split'] = ['train'] * n_train + ['val'] * n_val + ['test'] * (n - n_train - n_val)

    # Debug: Log class distribution in each split
    if debug:
        logger.info(f"Dataset loaded with {n} total samples")
        for split in ['train', 'val', 'test']:
            split_df = df[df['split'] == split]
            logger.info(f"{split.capitalize()} set: {len(split_df)} samples")
            class_counts = split_df[label_col].value_counts().to_dict()
            logger.info(f"{split.capitalize()} class distribution: {class_counts}")
        
        # Check for missing files
        missing_files = [f for f in df[path_col] if not os.path.exists(f)]
        if missing_files:
            logger.warning(f"Found {len(missing_files)} missing files in dataset!")
            logger.warning(f"First 5 missing files: {missing_files[:5]}")

    datagens = {
        'train': ImageDataGenerator(
            rescale=1/255.0,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        ),
        'val': ImageDataGenerator(rescale=1/255.0),
        'test': ImageDataGenerator(rescale=1/255.0)
    }
    
    # Debug: Log preprocessing parameters
    if debug:
        logger.info(f"Image preprocessing configuration:")
        logger.info(f"  - Target image size: {img_size}x{img_size}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Train augmentation: rotation, shift, shear, zoom, flip")
        logger.info(f"  - Train/Val/Test rescaling: 1/255.0")
    
    gens = {}
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        logger.info(f"Creating {split} generator: {len(split_df)} samples")
        
        # If label_order isn't specified but we have classes in the dataframe,
        # use them to ensure consistent ordering
        if label_order is None and label_col in df.columns:
            label_order = sorted(df[label_col].unique())
            logger.info(f"Using derived label order: {label_order}")
        
        gens[split] = datagens[split].flow_from_dataframe(
            dataframe=split_df,
            x_col=path_col,
            y_col=label_col,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=(split == 'train'),
            classes=label_order,
            color_mode="grayscale"  # force grayscale loading
        )
        
        # Debug: Check class indices mapping
        if debug:
            logger.info(f"{split} generator class indices: {gens[split].class_indices}")
            
            # Verify the first batch
            try:
                x_batch, y_batch = next(gens[split])
                logger.info(f"{split} first batch: X shape={x_batch.shape}, Y shape={y_batch.shape}")
                logger.info(f"{split} X min={np.min(x_batch)}, max={np.max(x_batch)}")
                
                # Save a few images to verify they're loading correctly
                if split == 'train':
                    debug_dir = os.path.join("debug", "data_samples")
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    for i in range(min(5, len(x_batch))):
                        plt.figure(figsize=(5, 5))
                        img = x_batch[i]
                        if img.shape[2] == 1:  # Grayscale
                            plt.imshow(img.squeeze(), cmap='gray')
                        else:
                            plt.imshow(img)
                        
                        # Get label
                        label_idx = np.argmax(y_batch[i])
                        label_name = next((k for k, v in gens[split].class_indices.items() 
                                          if v == label_idx), f"Unknown-{label_idx}")
                        
                        plt.title(f"Class: {label_name} (idx: {label_idx})")
                        plt.axis('off')
                        plt.savefig(os.path.join(debug_dir, f"{split}_sample_{i}.png"))
                        plt.close()
            except Exception as e:
                logger.error(f"Error checking {split} generator: {e}")
    
    # Additional verification to ensure class indices are consistent
    if debug:
        class_indices_list = [gen.class_indices for gen in gens.values()]
        if len(set(frozenset(ci.items()) for ci in class_indices_list)) > 1:
            logger.warning("Class indices are not consistent across generators!")
            for split, indices in zip(['train', 'val', 'test'], class_indices_list):
                logger.warning(f"{split} indices: {indices}")
        else:
            logger.info("Class indices are consistent across all generators")
    
    # Return the three generators separately instead of returning the dictionary
    return gens['train'], gens['val'], gens['test']
