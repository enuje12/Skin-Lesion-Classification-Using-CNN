# Skin Lesion Classification Using CNN

## Project Overview
This project implements an automated system to classify skin lesions as **benign** or **malignant** using deep learning. Early detection of skin cancer is critical for effective treatment and improved survival rates. The model leverages **transfer learning** with **MobileNetV2** and includes data preprocessing, augmentation, and real-time multi-image prediction.

---

## Features
- Classifies dermoscopic images as benign or malignant.
- Provides confidence scores alongside predictions.
- Real-time multi-image prediction support.
- Evaluates performance using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.

---

## Dataset
**Dataset:** [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
- Total Images: 10,015 dermoscopic images  
- Classes: Benign, Malignant  
- Metadata: `age_approx`, `anatom_site_general`, `isic_id`

**Example: Loading metadata in Python**
```python
import pandas as pd

metadata = pd.read_csv("ham10000_metadata.csv")
print(metadata.head())
