
# Alzheimer’s MRI Classification with ADNI & DenseNet

This repository implements a high-accuracy Alzheimer's disease classifier using the ADNI dataset and transfer-learning with DenseNet169 and DenseNet201.

## 📖 Overview

- **Dataset**: ADNI T1-weighted MRI scans, organized into four classes—  
  - **CN** (Cognitive Normal)  
  - **EMCI** (Early Mild Cognitive Impairment)  
  - **LMCI** (Late Mild Cognitive Impairment)  
  - **AD** (Alzheimer’s Disease)  
- **Approach**:  
  1. **Data loading & preprocessing**: resize to 224×224, normalize, optional augmentation.  
  2. **Modeling**: fine-tune both DenseNet169 and DenseNet201 backbones (freeze/unfreeze strategies).  
  3. **Training**: two-phase training (head only, then partial base unfreeze) with early stopping and LR scheduling.  
  4. **Evaluation**: accuracy, precision/recall/F1, confusion matrices, ROC/AUC curves.  
  5. **Inference**: save best checkpoints and run softmax predictions on new scans.
