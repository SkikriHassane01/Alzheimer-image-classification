# Alzheimer's MRI Classification with ADNI & DenseNet

This repository implements a high-accuracy Alzheimer's disease classifier using the ADNI T1-weighted MRI dataset and transfer learning (DenseNet169, DenseNet201, ResNet50).

## 📖 Overview

- **Dataset**: ADNI T1-weighted MRI scans, organized into four classes—  
  - **CN** (Cognitive Normal / Non Demented)  
  - **EMCI** (Early Mild Cognitive Impairment / Very Mild Dementia)  
  - **LMCI** (Late Mild Cognitive Impairment / Mild Dementia)  
  - **AD** (Alzheimer's Disease / Moderate Dementia)  
- **Approach**:  
  1. Data loading & preprocessing (resize to 224×224, normalize, augment).  
  2. Modeling: fine-tune DenseNet169, DenseNet201, and ResNet50 backbones.  
  3. Training: two-phase (head only, then partial base unfreeze) with early stopping & LR scheduling.  
  4. Evaluation: accuracy, precision/recall/F1, confusion matrices, ROC/AUC curves.  
  5. Inference: save best checkpoints; softmax predictions on new scans.

## ✨ Features

- Config-driven data preprocessing, balancing & augmentation  
- Multiple pretrained architectures: DenseNet201, DenseNet169, ResNet50  
- Two-stage training (initial + fine-tuning)  
- TensorBoard integration, detailed logging & debugging  
- Comprehensive evaluation & visualization (confusion matrix, ROC, MCC)  
- Modular codebase for EDA, data loaders, training, evaluation

## 📁 Project Structure

```text
Alzheimer-image-classification/
├── configs/                  
│   ├── data_config.yaml      
│   ├── model_config.yaml     
│   └── train_config.yaml     
├── Data/                     
│   ├── raw/                  
│   └── processed/            
├── src/                      
│   ├── data/                 
│   ├── eda/                  
│   ├── evaluation/           
│   ├── models/               
│   ├── training/             
│   └── utils/                
├── output/                   
│   ├── checkpoints/          
│   └── results/              
├── logs/                     
├── figures/                  
└── main.py                   
```

## 🚀 Installation

### Prerequisites
- Python 3.8+  
- TensorFlow 2.x  
- (Optional) CUDA & cuDNN for GPU

### Setup
```bash
git clone https://github.com/yourusername/Alzheimer-image-classification.git
cd Alzheimer-image-classification
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📋 Usage

```bash
# Prepare raw data (creates Data/processed/dataset.csv)
python main.py --data-loader

# Train (initial + fine-tune)
python main.py --fine-tune

# Evaluate only
python main.py --evaluate-only --checkpoint path/to/model.keras

# Enable debug logs & visualizations
python main.py --debug
```

## 🧠 Model Architectures

- **DenseNet201 & DenseNet169**: selective freezing, custom head (Dropout, BN, Dense)  
- **ResNet50**: global pooling + Dense head  
- Fine-tuning with lower LR on deep blocks

## 🔄 Data Processing Pipeline

1. **load_dataset**: scan folders → DataFrame  
2. **balance_dataset**: downsample/augment to equalize classes  
3. **create_data_generators**: split into train/val/test; apply augmentation

## 📊 Evaluation Metrics & Visualizations

- Accuracy, precision, recall, F1-score  
- Confusion matrix (normalized or raw)  
- ROC/AUC curves (per class)  
- Matthews Correlation Coefficient (MCC)  
- Sample batch and prediction grids

## 📈 Results

Outputs saved under `output/` and `output/results/` include:  
- Model checkpoints  
- Training history plots  
- Confusion matrix & classification report CSV  
- Detailed predictions CSV  
- Sample visualizations

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- ADNI dataset: http://adni.loni.usc.edu/  
- TensorFlow & Keras teams  
- Open-source library contributors
