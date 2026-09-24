import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.risk_predictor import DiabetesRiskPredictor

st.set_page_config(
    page_title="Early Diabetes Risk Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 6px solid #DC2626;
        padding: 1.5rem;
        border-radius: 10px;
        color: #991B1B;
    }
    .risk-mod {
        background-color: #FEF3C7;
        border-left: 6px solid #D97706;
        padding: 1.5rem;
        border-radius: 10px;
        color: #92400E;
    }
    .risk-low {
        background-color: #D1FAE5;
        border-left: 6px solid #059669;
        padding: 1.5rem;
        border-radius: 10px;
        color: #065F46;
    }
</style>
""", unsafe_allow_dict=True)

st.markdown('<div class="main-header">🩺 Early Diabetes Risk Prediction System</div>', unsafe_allow_dict=True)
st.markdown('<div class="sub-header">Machine Learning Powered Clinical Symptom & Metabolic Risk Evaluator</div>', unsafe_allow_dict=True)

@st.cache_resource
def load_predictor():
    return DiabetesRiskPredictor(models_dir='models')

try:
    predictor = load_predictor()
except Exception as e:
    st.error("Model artifacts not found! Please ensure model files exist in 'models/'.")
    st.stop()

# Sidebar for patient input parameters
st.sidebar.header("📋 Patient Clinical Profile")

st.sidebar.subheader("1. Demographics & Biometrics")
age = st.sidebar.slider("Age (years)", 18, 85, 45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
glucose = st.sidebar.slider("Fasting Blood Glucose (mg/dL)", 70.0, 300.0, 115.0, step=1.0)
bmi = st.sidebar.slider("Body Mass Index (BMI kg/m²)", 15.0, 50.0, 26.5, step=0.1)
bp = st.sidebar.slider("Systolic Blood Pressure (mmHg)", 90, 200, 120)
pedigree = st.sidebar.slider("Diabetes Pedigree Function (Genetic Risk)", 0.08, 2.40, 0.45, step=0.01)

st.sidebar.subheader("2. Early Symptom Checklist")
col_sym1, col_sym2 = st.sidebar.columns(2)

with col_sym1:
    polyuria = st.checkbox("Frequent Urination (Polyuria)")
    polydipsia = st.checkbox("Excessive Thirst (Polydipsia)")
    weight_loss = st.checkbox("Sudden Weight Loss")
    weakness = st.checkbox("General Weakness/Fatigue")
    polyphagia = st.checkbox("Excessive Hunger (Polyphagia)")
    visual_blurring = st.checkbox("Visual Blurring")
    genital_thrush = st.checkbox("Genital Thrush")

with col_sym2:
    itching = st.checkbox("Itching/Pruritus")
    irritability = st.checkbox("Irritability")
    delayed_healing = st.checkbox("Delayed Wound Healing")
    partial_paresis = st.checkbox("Partial Paresis/Numbness")
    muscle_stiffness = st.checkbox("Muscle Stiffness")
    alopecia = st.checkbox("Alopecia (Hair Loss)")
    obesity = st.checkbox("Obesity Class")

patient_input = {
    'Age': age,
    'Gender': gender,
    'Polyuria': 'Yes' if polyuria else 'No',
    'Polydipsia': 'Yes' if polydipsia else 'No',
    'Sudden_Weight_Loss': 'Yes' if weight_loss else 'No',
    'Weakness': 'Yes' if weakness else 'No',
    'Polyphagia': 'Yes' if polyphagia else 'No',
    'Genital_Thrush': 'Yes' if genital_thrush else 'No',
    'Visual_Blurring': 'Yes' if visual_blurring else 'No',
    'Itching': 'Yes' if itching else 'No',
    'Irritability': 'Yes' if irritability else 'No',
    'Delayed_Healing': 'Yes' if delayed_healing else 'No',
    'Partial_Paresis': 'Yes' if partial_paresis else 'No',
    'Muscle_Stiffness': 'Yes' if muscle_stiffness else 'No',
    'Alopecia': 'Yes' if alopecia else 'No',
    'Obesity': 'Yes' if obesity else 'No',
    'Glucose_Level': glucose,
    'BMI': bmi,
    'Systolic_BP': bp,
    'Diabetes_Pedigree': pedigree
}

# Main tabs layout
tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Risk Assessment", "📈 Model Benchmarks & Insights", "📂 Dataset Explorer"])

with tab1:
    result = predictor.predict_risk(patient_input)
    risk_score = result['risk_score_percentage']
    category = result['risk_category']
    recommendation = result['recommendation']
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.subheader("Diabetes Risk Score")
        st.metric(label="Calculated Risk", value=f"{risk_score}%")
        
        # Risk meter gauge visualization
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.barh([0], [100], color='#E5E7EB', height=0.5)
        bar_color = '#DC2626' if risk_score >= 65 else ('#D97706' if risk_score >= 30 else '#059669')
        ax.barh([0], [risk_score], color=bar_color, height=0.5)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xticks([0, 30, 65, 100])
        ax.set_xticklabels(['0%', '30% (Low)', '65% (Mod)', '100% (High)'])
        ax.set_title("Probability Scale", fontsize=10, fontweight='bold')
        sns.despine(left=True, bottom=False)
        st.pyplot(fig)
        
    with col_res2:
        st.subheader("Diagnostic Classification & Medical Guidance")
        if risk_score >= 65:
            st.markdown(f'<div class="risk-high"><h3>🚨 {category}</h3><p>{recommendation}</p></div>', unsafe_allow_dict=True)
        elif risk_score >= 30:
            st.markdown(f'<div class="risk-mod"><h3>⚠️ {category}</h3><p>{recommendation}</p></div>', unsafe_allow_dict=True)
        else:
            st.markdown(f'<div class="risk-low"><h3>✅ {category}</h3><p>{recommendation}</p></div>', unsafe_allow_dict=True)
            
        if result['key_symptoms_present']:
            st.write("**Key Symptom Factors Identified:**", ", ".join(result['key_symptoms_present']))
        else:
            st.write("**Key Symptom Factors Identified:** None reported.")

with tab2:
    st.subheader("Machine Learning Classification Model Performance")
    if os.path.exists("models/model_benchmark_results.csv"):
        results_df = pd.read_csv("models/model_benchmark_results.csv")
        st.dataframe(results_df.style.highlight_max(axis=0, color='#D1FAE5'), use_container_width=True)
    else:
        st.info("Run `main.py` to generate benchmarking results CSV.")
        
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists("plots/roc_curves.png"):
            st.image("plots/roc_curves.png", caption="Model ROC-AUC Curves")
    with col_img2:
        if os.path.exists("plots/best_model_confusion_matrix.png"):
            st.image("plots/best_model_confusion_matrix.png", caption="Confusion Matrix - Best Model")

with tab3:
    st.subheader("Exploratory Data Analysis Plots")
    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        if os.path.exists("plots/symptoms_prevalence.png"):
            st.image("plots/symptoms_prevalence.png", caption="Symptom Prevalence in Diabetes Positive vs Negative")
        if os.path.exists("plots/class_distribution.png"):
            st.image("plots/class_distribution.png", caption="Dataset Target Class Distribution")
    with col_eda2:
        if os.path.exists("plots/glucose_bmi_distribution.png"):
            st.image("plots/glucose_bmi_distribution.png", caption="Fasting Glucose & BMI Distributions")
        if os.path.exists("plots/correlation_matrix.png"):
            st.image("plots/correlation_matrix.png", caption="Feature Correlation Matrix")
