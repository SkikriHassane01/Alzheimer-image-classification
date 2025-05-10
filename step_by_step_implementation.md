# Project Overall Structure

The project is organized into several key directories and files:

- **configs/**  
  Contains configuration files such as `train_config.yaml`, `model_config.yaml`, and `data_config.yaml` that store hyperparameters, model settings, and data parameters.

- **Data/processed/**  
  Contains processed datasets (e.g. `balanced_dataset.csv`) used for training, validation, and testing.

- **logs/**  
  Stores log files generated during training and evaluation.

- **src/**  
  Contains the source code divided into several submodules:
  - **src/training/**:  
    Contains the `model_trainer.py` script that handles model training and fine-tuning.
  - **src/evaluation/**:  
    Contains the `evaluate.py` script that evaluates the model and creates detailed analysis.
  - **src/utils/**:  
    Provides helper functions such as `visualization.py` (for plotting results and confusion matrix) and `logger.py` (for logging).
  - **src/models/**:  
    Contains modules like `model_factory.py` for constructing the model.
  - **src/data/**:  
    Provides data handling, e.g. `balanced_data_loader.py` for generating balanced data generators.

- **main.py**  
  The main pipeline script that glues all components together – setting up devices, loading data/configs, building or loading the model, training, evaluating, and saving results.

---

# Step-by-Step Implementation Details

## 1. Data Handling and Configuration

- **Purpose:**  
  Prepare the dataset and load configuration parameters.  
- **Files Involved:**  
  `configs/data_config.yaml`, `Data/processed/dataset.csv`, and `src/data/balanced_data_loader.py`.

- **Explanation:**  
  - *balanced_data_loader.py*:  
    This script reads the raw CSV file, balances the classes by sampling or augmentation, and produces a balanced CSV output. It returns training, validation, and test generators that the training script reads to load images in batches as defined by `data_config.yaml`.

## 2. Model Training Script

- **File:** `src/training/model_trainer.py`
- **Purpose:**  
  Handle model training by reading configurations, setting callbacks, training the model, and (if necessary) fine-tuning the model.
- **Key Functions:**

  - **`__init__`**  
    *Initializes the trainer by loading training configurations and setting up directories.*  
    - Loads learning rate, epochs, batch size, and other hyperparameters.
    - Sets up output and results directories for checkpoints and training history plots.

  - **`_create_callbacks`**  
    *Creates Keras callbacks to monitor training progress.*  
    - Sets a checkpoint callback to save the best model.
    - Sets early stopping and learning rate reduction callbacks based on validation metrics.

  - **`prepare_for_fine_tuning`**  
    *Prepares the pre-trained model for fine-tuning.*  
    - Unfreezes specific layers based on layer names.
    - Recompiles the model with a lower learning rate suitable for fine-tuning.

  - **`train`**  
    *Carries out the training process with provided data generators.*  
    - Compiles the model if it hasn’t been compiled yet.
    - Runs `model.fit` using the training and validation generators.
    - Saves the final model and plots training history.

## 3. Model Evaluation Script

- **File:** `src/evaluation/evaluate.py`
- **Purpose:**  
  Evaluate the trained model on unseen test data and perform detailed analysis.
- **Key Functions:**

  - **`evaluate`**  
    *Evaluates the model’s performance on the test data.*  
    - Checks if the model is compiled (and compiles it if necessary).
    - Uses `model.evaluate` to calculate loss and accuracy.
  
  - **`predict_and_analyze`**  
    *Generates predictions and performs detailed performance analysis.*  
    - Resets the test generator and makes predictions on test data.
    - Computes the confusion matrix using predictions and true labels.
    - Uses helper functions from `visualization.py` to plot and save the confusion matrix.
    - Logs a classification report and computes additional metrics such as the Matthews Correlation Coefficient.

## 4. Main Pipeline Script

- **File:** `main.py`
- **Purpose:**  
  Orchestrate the complete pipeline (data processing, model creation/training, evaluation, and results summarization).
- **Key Steps in main():**

  - **Device Setup:**  
    Checks for GPU availability and configures device settings for TensorFlow.
  
  - **Data Preparation:**  
    Loads data configurations and creates balanced data generators.
  
  - **Model Building/Loading:**  
    Uses `ModelFactory` (from `src/models/model_factory.py`) to either build a new model or load from checkpoint.
  
  - **Training:**  
    If not in evaluation-only mode, the pipeline trains and optionally fine-tunes the model using `ModelTrainer`.
  
  - **Evaluation:**  
    Uses `ModelEvaluator` to assess model performance and generate detailed analysis.
  
  - **Results Summarization:**  
    Writes a summary of evaluation metrics, classification report, and additional stats to a results file.

---

This markdown provides a clear and simple roadmap for developing the project step by step. Each file and function is described with its purpose and role within the overall pipeline, enabling a clear understanding of how the project components interact.