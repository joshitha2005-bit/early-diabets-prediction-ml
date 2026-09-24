import os
import sys
import pandas as pd

from src.data_preprocessing import load_data, preprocess_data
from src.eda import run_eda
from src.models import train_and_evaluate_models
from src.risk_predictor import DiabetesRiskPredictor
from generate_dataset import generate_diabetes_dataset

def main():
    print("=" * 75)
    print("      EARLY DIABETES PREDICTION USING HEALTH & SYMPTOM FEATURES")
    print("=" * 75)
    
    # 1. Dataset Check/Generation
    dataset_path = 'diabetes_data.csv'
    if not os.path.exists(dataset_path):
        print("\n[Step 1] Dataset not found. Generating synthetic clinical dataset...")
        df = generate_diabetes_dataset(n_samples=1200, random_state=42)
        df.to_csv(dataset_path, index=False)
        print(f"Dataset generated with shape {df.shape} and saved to '{dataset_path}'.")
    else:
        print(f"\n[Step 1] Loading existing dataset from '{dataset_path}'...")
        df = load_data(dataset_path)
        print(f"Loaded dataset with shape {df.shape}.")
        
    # 2. Exploratory Data Analysis
    print("\n[Step 2] Executing Exploratory Data Analysis (EDA)...")
    run_eda(df, output_dir='plots')
    
    # 3. Data Preprocessing
    print("\n[Step 3] Preprocessing features and splitting train/test sets...")
    processed_data = preprocess_data(df, test_size=0.2, random_state=42)
    print(f"Training Samples: {len(processed_data['X_train'])}, Testing Samples: {len(processed_data['X_test'])}")
    
    # 4. Model Training and Benchmarking
    print("\n[Step 4] Training & Benchmarking Classification Models...")
    eval_results = train_and_evaluate_models(processed_data, models_dir='models', plots_dir='plots')
    
    # 5. Risk Predictor Demo
    print("\n[Step 5] Testing Diabetes Risk Predictor Engine...")
    predictor = DiabetesRiskPredictor(models_dir='models')
    
    sample_patient_high = {
        'Age': 56, 'Gender': 'Male', 'Polyuria': 'Yes', 'Polydipsia': 'Yes',
        'Sudden_Weight_Loss': 'Yes', 'Weakness': 'Yes', 'Polyphagia': 'Yes',
        'Genital_Thrush': 'No', 'Visual_Blurring': 'Yes', 'Itching': 'No',
        'Irritability': 'Yes', 'Delayed_Healing': 'Yes', 'Partial_Paresis': 'No',
        'Muscle_Stiffness': 'No', 'Alopecia': 'No', 'Obesity': 'Yes',
        'Glucose_Level': 182.0, 'BMI': 34.5, 'Systolic_BP': 142.0,
        'Diabetes_Pedigree': 0.85
    }
    
    sample_patient_low = {
        'Age': 28, 'Gender': 'Female', 'Polyuria': 'No', 'Polydipsia': 'No',
        'Sudden_Weight_Loss': 'No', 'Weakness': 'No', 'Polyphagia': 'No',
        'Genital_Thrush': 'No', 'Visual_Blurring': 'No', 'Itching': 'No',
        'Irritability': 'No', 'Delayed_Healing': 'No', 'Partial_Paresis': 'No',
        'Muscle_Stiffness': 'No', 'Alopecia': 'No', 'Obesity': 'No',
        'Glucose_Level': 95.0, 'BMI': 22.1, 'Systolic_BP': 112.0,
        'Diabetes_Pedigree': 0.22
    }
    
    res_high = predictor.predict_risk(sample_patient_high)
    res_low = predictor.predict_risk(sample_patient_low)
    
    print("\n--- High Risk Test Patient ---")
    print(f"Risk Score: {res_high['risk_score_percentage']}% ({res_high['risk_category']})")
    print(f"Recommendation: {res_high['recommendation']}")
    
    print("\n--- Low Risk Test Patient ---")
    print(f"Risk Score: {res_low['risk_score_percentage']}% ({res_low['risk_category']})")
    print(f"Recommendation: {res_low['recommendation']}")
    
    print("\n" + "=" * 75)
    print("  PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("  Models saved in: 'models/'")
    print("  Plots saved in:  'plots/'")
    print("=" * 75)

if __name__ == '__main__':
    main()
