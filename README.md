# Skin Lesion Classification Using CNN

## Project Overview

This project focuses on **automated skin lesion classification** using **MobileNetV2**, a lightweight and highly efficient **Convolutional Neural Network (CNN)** architecture widely used in computer vision applications. The goal is to build an accurate and computationally efficient model capable of classifying dermoscopic skin images into multiple diagnostic categories.

The model leverages **transfer learning**, where the pretrained MobileNetV2 (trained on ImageNet) is fine-tuned on the **HAM10000** dermatology dataset. This approach significantly improves performance even with limited training resources, making it ideal for medical imaging tasks.

By integrating data preprocessing, augmentation, model training, validation, and evaluation steps, this project demonstrates the application of **deep learning and computer vision techniques** in healthcare. The workflow highlights how CNN-based architectures can assist in early detection of skin diseases, showcasing a strong understanding of applied machine learning for image classification.

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

---

## Methodology
**1. Data Preprocessing**

- Selected key features: isic_id, benign_malignant, age_approx, anatom_site_general
- Handled missing values:
Numeric values `age_approx` replaced by median, 
Categorical values `anatom_site_general` replaced by "unknown", 
Filtered out missing image files to prevent training errors

**2. Data Splitting**

- Training: 72%, Validation: 8%, Test: 20%
- Stratified splitting to maintain class distribution

**3. Data Augmentation**

- Techniques used: rotation (±20°), zoom (±20%), horizontal flip
- Pixel normalization to [0,1]
- Implemented using TensorFlow’s ImageDataGenerator

**4. Model Architecture**

- Base: MobileNetV2 pretrained on ImageNet (frozen convolutional layers)
- Custom layers:
1. GlobalAveragePooling2D
2. Dense layer (64 units, ReLU)
3. Dropout (0.3)
4. Output Dense layer (1 unit, Sigmoid)

**5. Training**

- Optimizer: Adam
- Loss function: Binary Crossentropy
- Epochs: 10
- Batch size: 32
- Data augmentation applied to training set

**6. Evaluation**

- Metrics: Accuracy, Confusion Matrix, Precision, Recall, F1-score, ROC-AUC
- Multi-image prediction implemented for real-time inference with visual display

**7. Results**

- Test Accuracy: 89.93%
- ROC-AUC Score: 0.85
- Confusion matrix demonstrates high precision and recall with minimal false positives and negatives

## Installation

1. Clone the repository:
```python
git clone https://github.com/your-username/skin-lesion-classification.git
cd skin-lesion-classification
```

2. Install dependencies
```python
pip install -r requirements.txt

```

3. Run the Python script
- Use a GPU-enabled environment for faster training (recommended: Google Colab)
```python
python cv_model.py
```

## Future Work

- Fine-tune deeper layers of MobileNetV2 to further improve accuracy
- Expand dataset with additional diverse images
- Develop a web-based or mobile deployment for real-world clinical use

## References

1. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data.
2. Chollet, F. (2017). Deep Learning with Python. Manning Publications.
3. TensorFlow Documentation: https://www.tensorflow.org
