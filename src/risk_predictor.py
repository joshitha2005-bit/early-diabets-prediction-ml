import os
import joblib
import pandas as pd
import numpy as np

class DiabetesRiskPredictor:
    """
    Early Diabetes Risk Predictor Engine.
    Loads trained model and scaler, accepts patient parameters, and provides risk score,
    risk classification (Low, Moderate, High), and personalized medical insights.
    """
    def __init__(self, models_dir='models'):
        model_path = os.path.join(models_dir, 'best_model.joblib')
        scaler_path = os.path.join(models_dir, 'scaler.joblib')
        features_path = os.path.join(models_dir, 'feature_names.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model artifacts not found! Please run training pipeline first.")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        
    def predict_risk(self, patient_dict):
        """
        Accepts a dictionary of patient health features and symptoms:
        Example:
        {
            'Age': 45, 'Gender': 'Male', 'Polyuria': 'Yes', 'Polydipsia': 'Yes',
            'Sudden_Weight_Loss': 'No', 'Weakness': 'Yes', 'Polyphagia': 'No',
            'Genital_Thrush': 'No', 'Visual_Blurring': 'Yes', 'Itching': 'No',
            'Irritability': 'No', 'Delayed_Healing': 'Yes', 'Partial_Paresis': 'No',
            'Muscle_Stiffness': 'No', 'Alopecia': 'No', 'Obesity': 'Yes',
            'Glucose_Level': 165.0, 'BMI': 31.5, 'Systolic_BP': 135.0,
            'Diabetes_Pedigree': 0.65
        }
        """
        binary_map = {'Yes': 1, 'No': 0, 1: 1, 0: 0, 'Male': 1, 'Female': 0}
        
        # Build input DataFrame
        input_data = {}
        for feature in self.feature_names:
            val = patient_dict.get(feature, 0)
            if isinstance(val, str) and val in binary_map:
                input_data[feature] = binary_map[val]
            else:
                input_data[feature] = float(val)
                
        input_df = pd.DataFrame([input_data])[self.feature_names]
        
        # Scale numerical features
        num_cols = ['Age', 'Glucose_Level', 'BMI', 'Systolic_BP', 'Diabetes_Pedigree']
        scaled_input = input_df.copy()
        scaled_input[num_cols] = self.scaler.transform(input_df[num_cols])
        
        # Predict probability
        if hasattr(self.model, 'predict_proba'):
            prob = self.model.predict_proba(scaled_input)[0][1]
        else:
            pred = self.model.predict(scaled_input)[0]
            prob = 0.90 if pred == 1 else 0.10
            
        risk_score = round(prob * 100, 1)
        
        if risk_score < 30.0:
            category = "Low Risk"
            recommendation = "Low likelihood of early diabetes. Maintain a healthy lifestyle, balanced diet, and periodic annual checkups."
            action_color = "Green"
        elif 30.0 <= risk_score < 65.0:
            category = "Moderate Risk (Pre-diabetes Alert)"
            recommendation = "Moderate risk indicators detected. Consult a physician for HbA1c/Fasting Glucose testing, physical activity increase, and dietary monitoring."
            action_color = "Yellow/Orange"
        else:
            category = "High Risk (Immediate Attention Advised)"
            recommendation = "High early diabetes risk detected! Urgent clinical evaluation, Comprehensive Metabolic Panel (CMP), and lifestyle/medical management recommended."
            action_color = "Red"
            
        # Key contributing symptoms highlighted
        key_symptoms = []
        symptom_list = ['Polyuria', 'Polydipsia', 'Sudden_Weight_Loss', 'Polyphagia', 'Visual_Blurring', 'Delayed_Healing', 'Obesity']
        for sym in symptom_list:
            if patient_dict.get(sym) in ['Yes', 1]:
                key_symptoms.append(sym.replace('_', ' '))
                
        return {
            'risk_score_percentage': risk_score,
            'risk_category': category,
            'action_color': action_color,
            'recommendation': recommendation,
            'key_symptoms_present': key_symptoms,
            'glucose_level': patient_dict.get('Glucose_Level'),
            'bmi': patient_dict.get('BMI')
        }

if __name__ == '__main__':
    predictor = DiabetesRiskPredictor()
    sample_patient = {
        'Age': 52,
        'Gender': 'Male',
        'Polyuria': 'Yes',
        'Polydipsia': 'Yes',
        'Sudden_Weight_Loss': 'Yes',
        'Weakness': 'Yes',
        'Polyphagia': 'Yes',
        'Genital_Thrush': 'No',
        'Visual_Blurring': 'Yes',
        'Itching': 'No',
        'Irritability': 'Yes',
        'Delayed_Healing': 'Yes',
        'Partial_Paresis': 'No',
        'Muscle_Stiffness': 'No',
        'Alopecia': 'No',
        'Obesity': 'Yes',
        'Glucose_Level': 175.0,
        'BMI': 33.2,
        'Systolic_BP': 138.0,
        'Diabetes_Pedigree': 0.75
    }
    res = predictor.predict_risk(sample_patient)
    print("\n--- SAMPLE PATIENT DIAGNOSTIC RESULT ---")
    for k, v in res.items():
        print(f"{k}: {v}")
