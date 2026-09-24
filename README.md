# Early Diabetes Prediction Machine Learning Framework

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Streamlit App](https://img.shields.io/badge/Web%20App-Streamlit-red.svg)
![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)

An end-to-end Machine Learning Framework for **Early Stage Diabetes Prediction** utilizing health-related biometrics and clinical symptom-based features. The project includes data preprocessing, exploratory data analysis (EDA), multi-model classification benchmarking (9 models), cross-validation, performance metrics evaluation, and an interactive diagnostic risk prediction engine.

---

## 📌 Project Overview & Objectives

Early detection of diabetes mellitus plays a pivotal role in preventing severe microvascular and macrovascular complications (such as retinopathy, nephropathy, neuropathy, and cardiovascular diseases). This project integrates both **metabolic indicators** (*Fasting Blood Glucose, BMI, Systolic Blood Pressure, Diabetes Pedigree Function*) and **early clinical symptom markers** (*Polyuria, Polydipsia, Sudden Weight Loss, Polyphagia, Visual Blurring, Delayed Healing, Irritability, Obesity*) to deliver accurate early risk classification.

### 🌟 Key Features:
- **Comprehensive Preprocessing**: Automatic encoding of binary symptom flags, feature scaling (`StandardScaler`), missing value validation, and stratified train-test splitting.
- **Exploratory Data Analysis (EDA)**: Automatic generation of feature correlation matrices, class distribution plots, and symptom prevalence comparison charts.
- **Multi-Model Benchmarking**: Trains and benchmarks 9 classification algorithms:
  1. Logistic Regression
  2. Support Vector Machine (SVM)
  3. Random Forest Classifier
  4. Gradient Boosting Classifier
  5. XGBoost Classifier
  6. Decision Tree Classifier
  7. K-Nearest Neighbors (KNN)
  8. Naive Bayes Classifier
  9. Multi-Layer Perceptron (MLP Neural Net)
- **Model Evaluation**: Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, 5-Fold Stratified CV, ROC curves, and Confusion Matrices.
- **Diagnostic Risk Prediction Engine**: Evaluates individual patient inputs to generate a **Risk Score (0-100%)**, **Risk Level (Low, Moderate, High)**, and **Personalized Medical Recommendations**.
- **Interactive Web App**: Built with **Streamlit** for real-time clinical symptom inputs and interactive visualization.
- **Full IDE Compatibility**: Optimized to run seamlessly in **VS Code** (Python scripts) and **Jupyter Notebook** (`.ipynb`).

---

## 📁 Repository Directory Structure

```
early-diabets-prediction-ml/
│
├── dataset/ & main root
│   ├── diabetes_data.csv               # Clinical dataset (1,200 patient records)
│   ├── generate_dataset.py             # Synthetic clinical dataset generator
│
├── src/                                # Modular source code
│   ├── __init__.py
│   ├── data_preprocessing.py           # Cleaning, encoding, scaling, train/test split
│   ├── eda.py                          # EDA charting and visualization engine
│   ├── models.py                       # ML model training, benchmarking, ROC & CM evaluation
│   └── risk_predictor.py               # Patient risk prediction & diagnostic scoring engine
│
├── plots/                              # Generated EDA & model evaluation figures
│   ├── class_distribution.png
│   ├── symptoms_prevalence.png
│   ├── glucose_bmi_distribution.png
│   ├── correlation_matrix.png
│   ├── roc_curves.png
│   ├── best_model_confusion_matrix.png
│   └── feature_importances.png
│
├── models/                             # Saved model & preprocessor artifacts
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── feature_names.joblib
│   └── model_benchmark_results.csv
│
├── main.py                             # Master pipeline execution script (VS Code)
├── app.py                              # Streamlit interactive web application
├── early_diabetes_prediction.ipynb     # Pre-rendered Jupyter Notebook with visualizations
├── build_notebook.py                   # Notebook generator utility
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 📊 Dataset Description

Each row in `diabetes_data.csv` represents a patient record with 20 features + 1 target variable (`Class`):

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Age` | Numeric | Patient age in years |
| `Gender` | Categorical | `Male` / `Female` |
| `Glucose_Level` | Numeric | Fasting blood glucose level (mg/dL) |
| `BMI` | Numeric | Body Mass Index (kg/m²) |
| `Systolic_BP` | Numeric | Systolic blood pressure (mmHg) |
| `Diabetes_Pedigree` | Numeric | Genetic pedigree score measuring family history risk |
| `Polyuria` | Categorical | Excessive/frequent urination (`Yes`/`No`) |
| `Polydipsia` | Categorical | Excessive thirst (`Yes`/`No`) |
| `Sudden_Weight_Loss` | Categorical | Unexplained rapid weight loss (`Yes`/`No`) |
| `Weakness` | Categorical | Generalized physical fatigue/weakness (`Yes`/`No`) |
| `Polyphagia` | Categorical | Excessive/unusual hunger (`Yes`/`No`) |
| `Genital_Thrush` | Categorical | Fungal infection flag (`Yes`/`No`) |
| `Visual_Blurring` | Categorical | Blurred vision symptom (`Yes`/`No`) |
| `Itching` | Categorical | Persistent pruritus/itching (`Yes`/`No`) |
| `Irritability` | Categorical | Mood changes/irritability (`Yes`/`No`) |
| `Delayed_Healing` | Categorical | Slow healing of cuts/wounds (`Yes`/`No`) |
| `Partial_Paresis` | Categorical | Muscle weakness/partial paralysis (`Yes`/`No`) |
| `Muscle_Stiffness` | Categorical | Muscle stiffness (`Yes`/`No`) |
| `Alopecia` | Categorical | Hair loss/patchy baldness (`Yes`/`No`) |
| `Obesity` | Categorical | Obese class indicator (`Yes`/`No`) |
| **`Class`** | **Binary Target** | **`1` = Diabetes Positive (High Risk), `0` = Negative (Low Risk)** |

---

## 🚀 How to Run in VS Code

### 1. Clone & Set Up Environment
Open VS Code Terminal (`Ctrl + ~`) in the project directory:
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Execute Full Machine Learning Pipeline
Run the master script to generate plots, train all 9 classifiers, save models, and run diagnostic risk tests:
```bash
python main.py
```

### 3. Launch Interactive Streamlit Web App
To run the clinical web dashboard in your browser:
```bash
streamlit run app.py
```

---

## 📓 How to Run in Jupyter Notebook

1. Open VS Code or launch Jupyter in your browser:
   ```bash
   jupyter notebook
   ```
2. Open [`early_diabetes_prediction.ipynb`](file:///c:/Users/HP/OneDrive/Documents/early-diabets-prediction-ml/early_diabetes_prediction.ipynb).
3. Select your Python Kernel and click **Run All Cells**.
4. All Markdown explanations, code cells, pre-rendered output tables, ROC curves, confusion matrices, and risk predictions will render interactively.

---

## 📈 Model Performance Benchmark Summary

Below are the benchmark metrics evaluated on the hold-out test dataset (20% split) with 5-Fold Stratified Cross-Validation:

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8375** | **0.8279** | **0.8487** | **0.8382** | **0.9150** | **0.8348** |
| **Support Vector Machine** | 0.8292 | 0.8095 | 0.8571 | 0.8327 | 0.9038 | 0.8312 |
| **Random Forest** | 0.8208 | 0.8115 | 0.8319 | 0.8216 | 0.9097 | 0.8292 |
| **Naive Bayes** | 0.8208 | 0.8115 | 0.8319 | 0.8216 | 0.9114 | 0.8166 |
| **Gradient Boosting** | 0.8167 | 0.8000 | 0.8403 | 0.8197 | 0.9027 | 0.8159 |
| **K-Nearest Neighbors** | 0.8000 | 0.7983 | 0.7983 | 0.7983 | 0.8746 | 0.8026 |
| **Decision Tree** | 0.8000 | 0.8318 | 0.7479 | 0.7876 | 0.8237 | 0.7698 |
| **MLP Neural Net** | 0.7833 | 0.7769 | 0.7899 | 0.7833 | 0.8699 | 0.7857 |
| **XGBoost** | 0.7792 | 0.7750 | 0.7815 | 0.7782 | 0.8784 | 0.8108 |

---

## 🩺 Diagnostic Risk Prediction Example

```python
from src.risk_predictor import DiabetesRiskPredictor

predictor = DiabetesRiskPredictor(models_dir='models')
result = predictor.predict_risk({
    'Age': 54, 'Gender': 'Male', 'Polyuria': 'Yes', 'Polydipsia': 'Yes',
    'Sudden_Weight_Loss': 'Yes', 'Weakness': 'Yes', 'Polyphagia': 'Yes',
    'Glucose_Level': 175.0, 'BMI': 32.8, 'Systolic_BP': 138.0,
    'Diabetes_Pedigree': 0.75
})

print(f"Risk Score: {result['risk_score_percentage']}%")
print(f"Category: {result['risk_category']}")
print(f"Guidance: {result['recommendation']}")
```

---

## 📄 License & Attribution
Developed for Machine Learning in Healthcare & Statistical Data Science applications. Free for educational and research usage.
