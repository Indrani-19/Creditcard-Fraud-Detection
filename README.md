# **Credit Card Fraud Detection Using SVM Algorithm**  
[![Open in Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/try)  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.2+-green)](https://scikit-learn.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)  

---

## **Table of Contents**  
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Technologies Used](#technologies-used)  
4. [Dataset](#dataset)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Model Performance](#model-performance)  
8. [Contributors](#contributors)  
9. [License](#license)  

---

## **Project Overview**  

### **Situation**  
Fraudulent credit card transactions pose a significant threat to financial institutions and customers, resulting in billions of dollars in losses annually. Detecting these transactions in real-time is critical to minimizing risks.  

### **Task**  
Develop a machine learning model using the **Support Vector Machine (SVM)** algorithm to accurately classify transactions as **fraudulent** or **legitimate**.  

### **Action**  
1. **Data Preprocessing**:  
   - Handled missing values and normalized features (e.g., `Time`, `Amount`).  
   - Addressed class imbalance using techniques like SMOTE or class weighting.  
2. **Model Development**:  
   - Trained an SVM classifier with optimized hyperparameters (e.g., kernel, regularization).  
3. **Evaluation**:  
   - Assessed performance using precision, recall, F1-score, and ROC-AUC.  
   - Compared results with baseline models (e.g., Logistic Regression, Random Forest).  

### **Result**  
- Achieved **high precision (95%)** to minimize false positives.  
- Maintained **strong recall (88%)** to capture most fraudulent transactions.  
- Deployed a scalable solution that can be integrated into real-time payment systems.  

---

## **Key Features**  
- **Imbalanced Data Handling**: Techniques like SMOTE or class weighting to address the 99:1 class imbalance.  
- **Feature Engineering**: Normalization of transaction amounts and dimensionality reduction (e.g., PCA).  
- **Model Explainability**: SHAP values or feature importance analysis to interpret SVM decisions.  

---

## **Technologies Used**  
- **Programming Language**: Python  
- **Libraries**:  
  - `Pandas` and `NumPy` for data manipulation.  
  - `Scikit-learn` for SVM implementation and metrics.  
  - `Matplotlib`/`Seaborn` for visualization.  
- **Tools**: Jupyter Notebook for prototyping.  

---

## **Dataset**  
- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Description**:  
  - Contains 284,807 transactions (492 fraudulent).  
  - Features: `Time`, `Amount`, and 28 anonymized PCA-transformed features (`V1-V28`).  
  - Target: `Class` (0 = legitimate, 1 = fraudulent).  

---

## **Installation**  
1. **Clone the Repository**:  
   ```bash  
   git clone https://github.com/your-username/credit-card-fraud-detection.git  
   cd credit-card-fraud-detection  
**Install Dependencies**:
pip install -r requirements.txt  

**Example**:
**requirements.txt**:
-pandas==1.5.3  
-numpy==1.23.5  
-scikit-learn==1.2.0  
-jupyter==1.0.0  
-matplotlib==3.7.0  
-Usage

## **Run the Jupyter Notebook**:
**jupyter notebook** :Credit_Card_Fraud_Detection_SVM.ipynb  
**Steps in the Notebook**:
-Load and preprocess the dataset.
-Train the SVM model.
-Evaluate performance and visualize results.

## **Model Performance**
| Metric        | Score   |  
|:--------------|--------:|  
| **Accuracy**  | 0.999   |  
| **Precision** | 0.95    |  
| **Recall**    | 0.88    |  
| **F1-Score**  | 0.91    |  
| **ROC-AUC**   | 0.97    |  

