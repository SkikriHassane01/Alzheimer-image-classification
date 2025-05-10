import os
import yaml
import argparse
import tensorflow as tf
from datetime import datetime
import numpy as np

def setup_device():
    """
    Check for available devices (CPU/GPU), configure them properly,
    and return device information.
    
    Returns:
        str: Information about the device being used
    """
    # Check for GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    # Setup based on available hardware
    if gpus:
        try:
            # Set memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Return info about the first GPU
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            device_name = f"GPU: {gpu_details.get('device_name', 'Unknown GPU')}"
            device_type = "GPU"
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
            device_name = "CPU (GPU configuration failed)"
            device_type = "CPU"
    else:
        device_name = "CPU (No GPUs found)"
        device_type = "CPU"
    
    # Set visible devices based on what's available
    if device_type == "GPU":
        tf.config.set_visible_devices(gpus, 'GPU')
    
    return f"Training using: {device_name}"

# Import project modules
from src.utils.logger import get_logger
from src.models.model_factory import ModelFactory
from src.training.model_trainer import ModelTrainer
from src.evaluation.evaluate import ModelEvaluator
from src.data.balanced_data_loader import create_data_generators, balance_dataset
from src.data.data_loader import load_dataset, load_yaml_config
import pandas as pd

def main(args):
    # Initialize logger
    logger = get_logger("main")
    logger.info("=" * 50)
    logger.info("ALZHEIMER'S DISEASE CLASSIFICATION PIPELINE")
    logger.info("=" * 50)
    
    # Setup and check device (CPU/GPU)
    device_info = setup_device()
    logger.info(device_info)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
    
    # Load configurations
    logger.info("Loading configurations...")
    try:
        with open('configs/data_config.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        logger.info("Data configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load data configuration: {e}")
        return
    
    # STEP 0: Load raw data if requested
    if args.data_loader:
        logger.info("STEP 0: Loading raw dataset...")
        try:
            config = load_yaml_config("configs/data_config.yaml")
            dataset_dir = config.get("dataset_dir", "Data/raw")
            new_class_structure = config.get("new_class_structure")
            class_descriptions = config.get("class_descriptions")
            
            df = load_dataset(dataset_dir, new_class_structure, class_descriptions)
            if not df.empty:
                logger.info(f"Raw dataset loaded successfully with {len(df)} images")
                # Save the dataframe into a CSV file
                os.makedirs("Data/processed", exist_ok=True)
                csv_path = os.path.join("Data/processed", "dataset.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"Raw dataset saved to {csv_path}")
            else:
                logger.error("No images loaded; please verify your dataset structure.")
                if not os.path.exists('Data/processed/dataset.csv'):
                    return
        except Exception as e:
            logger.error(f"Error in data loading step: {e}")
            if not os.path.exists('Data/processed/dataset.csv'):
                return
    
    # STEP 1: Load and prepare data
    logger.info("STEP 1: Loading and preparing data...")
    csv_path = os.path.join('Data/processed', 'balanced_dataset.csv')
    
    if not os.path.exists(csv_path):
        logger.info("Balanced dataset not found. Balancing the data...")
        raw_csv = 'Data/processed/dataset.csv'
        df_raw = pd.read_csv(raw_csv)
        balanced_df = balance_dataset(
            df_raw,
            path_col='path',
            label_col='label',
            target_samples_per_class=20000,
            augment=True
        )
    logger.info(f"Using dataset from: {csv_path}")
    
    # Create data generators with debugging enabled
    try:
        train_gen, val_gen, test_gen = create_data_generators(
            csv_path=csv_path,
            path_col='path',
            label_col='label',
            img_size=data_config.get('img_height', 224),
            batch_size=data_config.get('batch_size', 32),
            train_frac=0.7,
            val_frac=0.15,
            random_state=42,
            debug=args.debug,  # Enable debugging
            label_order=["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]  # explicitly specify all 4 classes
        )
        logger.info(f"Data generators created successfully: {len(train_gen.filenames)} training, " 
                  f"{len(val_gen.filenames)} validation, {len(test_gen.filenames)} test samples")
        
        # Get class mapping
        class_indices = train_gen.class_indices
        idx_to_label = {v: k for k, v in class_indices.items()}
        class_descriptions = data_config.get('class_descriptions', {})
        
        # Verify class indices consistency between generators
        train_indices = train_gen.class_indices
        val_indices = val_gen.class_indices
        test_indices = test_gen.class_indices
        
        if train_indices != val_indices or train_indices != test_indices:
            logger.warning("WARNING: Class indices differ between generators!")
            logger.warning(f"Train indices: {train_indices}")
            logger.warning(f"Val indices: {val_indices}")
            logger.warning(f"Test indices: {test_indices}")
        
    except Exception as e:
        logger.error(f"Failed to create data generators: {e}")
        return
    
    # STEP 2: Build or load model
    logger.info("\nSTEP 2: Building or loading model...")
    model_factory = ModelFactory('configs/model_config.yaml')
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model, loaded = model_factory.load_model([args.checkpoint])
        if not loaded:
            logger.warning("Failed to load checkpoint, building new model")
    else:
        logger.info("Building new model...")
        model = model_factory.build_model()
    
    # STEP 3: Train model
    if not args.evaluate_only:
        logger.info("\nSTEP 3: Training model...")
        trainer = ModelTrainer(
            config_path='configs/train_config.yaml',
            output_dir=os.path.join(args.output_dir, 'checkpoints')
        )
        
        # Initial training phase
        model, history, model_path = trainer.train(
            model=model,
            train_generator=train_gen,
            val_generator=val_gen,
            fine_tuning=False
        )
        
        # Fine-tuning phase if requested
        if args.fine_tune:
            logger.info("\nSTEP 3b: Fine-tuning model...")
            model = trainer.prepare_for_fine_tuning(model)
            model, ft_history, ft_model_path = trainer.train(
                model=model,
                train_generator=train_gen,
                val_generator=val_gen,
                fine_tuning=True
            )
            model_path = ft_model_path
    else:
        logger.info("Skipping training, evaluation only mode")
        model_path = args.checkpoint
    
    # STEP 4: Evaluate model
    logger.info("\nSTEP 4: Evaluating model...")
    
    # Debug check: verify model input shape matches data shape
    input_shape = model.input_shape[1:]  # Exclude batch dimension
    data_shape = train_gen.image_shape
    logger.info(f"Model input shape: {input_shape}, Data shape: {data_shape}")
    if input_shape != data_shape:
        logger.warning(f"WARNING: Model input shape {input_shape} doesn't match data shape {data_shape}")
    
    # Save a summary of the model architecture for debugging
    model_summary_path = os.path.join(args.output_dir, 'model_summary.txt')
    with open(model_summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    logger.info(f"Saved model summary to {model_summary_path}")
    
    evaluator = ModelEvaluator(
        model=model,
        test_generator=test_gen,
        class_indices=class_indices,
        class_descriptions=class_descriptions
    )
    
    # Basic evaluation
    test_loss, test_accuracy = evaluator.evaluate()
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed analysis with debugging
    metrics = evaluator.predict_and_analyze(save_dir=os.path.join(args.output_dir, 'results'))
    
    # STEP 5: Save results summary
    logger.info("\nSTEP 5: Saving results summary...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.output_dir, f"results_summary_{timestamp}.txt")
    
    with open(summary_path, 'w') as f:
        f.write(f"Alzheimer's Disease Classification Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_factory.backbone}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Matthews Correlation Coefficient: {metrics['mcc']:.4f}\n\n")
        f.write("Classification Report:\n")
        # convert the dict to a table-like string before writing
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        f.write(report_df.to_string())
    
    logger.info(f"Results summary saved to: {summary_path}")
    logger.info("\nPipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alzheimer's Disease Classification Pipeline")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint to load')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Skip training and only evaluate model')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Perform fine-tuning after initial training')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional logging and visualization')
    parser.add_argument('--data-loader', action='store_true',
                        help='Run initial data loading step before other pipeline stages')
    
    args = parser.parse_args()
    main(args)
