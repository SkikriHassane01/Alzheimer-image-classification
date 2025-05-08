import os
import yaml
import argparse
import tensorflow as tf
from datetime import datetime

# Set memory growth to avoid GPU memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Import project modules
from src.utils.logger import get_logger
from src.models.model_factory import ModelFactory
from src.training.model_trainer import ModelTrainer
from src.evaluation.evaluate import ModelEvaluator
from src.data.balanced_data_loader import create_data_generators, balance_dataset
import pandas as pd
def main(args):
    # Initialize logger
    logger = get_logger("main")
    logger.info("=" * 50)
    logger.info("ALZHEIMER'S DISEASE CLASSIFICATION PIPELINE")
    logger.info("=" * 50)
    
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
    
    # Create data generators
    try:
        train_gen, val_gen, test_gen = create_data_generators(
            csv_path=csv_path,
            path_col='path',
            label_col='label',
            img_size=data_config.get('img_height', 224),
            batch_size=data_config.get('batch_size', 32),
            train_frac=0.7,
            val_frac=0.15,
            random_state=42
        )
        logger.info(f"Data generators created successfully: {len(train_gen.filenames)} training, " 
                  f"{len(val_gen.filenames)} validation, {len(test_gen.filenames)} test samples")
        
        # Get class mapping
        class_indices = train_gen.class_indices
        idx_to_label = {v: k for k, v in class_indices.items()}
        class_descriptions = data_config.get('class_descriptions', {})
        
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
    evaluator = ModelEvaluator(
        model=model,
        test_generator=test_gen,
        class_indices=class_indices,
        class_descriptions=class_descriptions
    )
    
    # Basic evaluation
    test_loss, test_accuracy = evaluator.evaluate()
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed analysis
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
        f.write(metrics['classification_report'])
    
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
    
    args = parser.parse_args()
    main(args)
