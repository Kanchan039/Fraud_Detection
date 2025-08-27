# 🔍 Fraud Detection System using Machine Learning & Deep Learning

This project implements a **fraud detection system** on highly imbalanced credit card transactions.  
It combines classical machine learning models, deep learning autoencoders, and an ensemble approach to accurately detect fraudulent transactions.

---

## 📝 Project Overview

Credit card fraud is rare but costly. Detecting it is challenging due to the extreme imbalance in the data (~0.17% fraud cases).  

This project includes:

- **Data Preprocessing**: Handling class imbalance with **SMOTE**.
- **Machine Learning Models**:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Isolation Forest
- **Deep Learning Model**:
  - Autoencoder (TensorFlow/Keras) for anomaly detection
- **Combined Model**: Ensemble of multiple methods for better performance.
- **Evaluation**: AUC, Average Precision (AP), Confusion Matrix, Classification Report.

---

## 📊 Dataset

- **Total transactions**: 284,807  
- **Features**: 31 features including anonymized numerical features (`V1`–`V28`) and `Amount`, `Time`.  
- **Fraud cases**: 492 (~0.17% of the data).  

**Class distribution**:


---

## ⚙️ Results

| Model              | AUC    | Average Precision (AP) |
|--------------------|--------|-----------------------|
| Logistic Regression | 0.9721 | 0.7232                |
| Random Forest       | 0.9572 | 0.8556                |
| XGBoost             | 0.9700 | 0.8612                |
| Isolation Forest    | 0.9499 | 0.1653                |
| Autoencoder         | 0.9573 | 0.6237                |
| **Combined Model**  | 0.9641 | 0.8471                |

**Confusion Matrix (Combined Model)**:

**Classification Report**:

✅ The model achieves **high recall for fraud detection**, meaning it successfully identifies most fraudulent transactions.  
⚠️ Precision is lower for fraud due to some false positives, which is acceptable in real-world fraud detection where missing a fraud is costlier than a false alarm.

---

## 🛠️ Technology Stack

- Python 3.10
- TensorFlow / Keras (Autoencoder)
- scikit-learn (Logistic Regression, Random Forest, Isolation Forest, preprocessing)
- XGBoost
- imbalanced-learn (SMOTE oversampling)
- pandas, numpy, matplotlib, seaborn

---

## 🚀 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection

# Create and activate virtual environment
python -m venv tfenv
.\tfenv\Scripts\activate   # Windows PowerShell
# or
source tfenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the project
python fraud.py
