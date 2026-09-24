import numpy as np
import pandas as pd

def generate_diabetes_dataset(n_samples=1200, random_state=42):
    np.random.seed(random_state)
    
    # Demographics and health indicators
    age = np.random.randint(20, 75, size=n_samples)
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.52, 0.48])
    bmi = np.round(np.random.normal(27.5, 5.8, size=n_samples), 1)
    bmi = np.clip(bmi, 16.0, 50.0)
    
    glucose = np.round(np.random.normal(118, 40, size=n_samples), 1)
    glucose = np.clip(glucose, 70.0, 280.0)
    
    systolic_bp = np.round(np.random.normal(122, 16, size=n_samples), 0)
    systolic_bp = np.clip(systolic_bp, 90, 190)
    
    diabetes_pedigree = np.round(np.random.exponential(scale=0.4, size=n_samples) + 0.08, 3)
    diabetes_pedigree = np.clip(diabetes_pedigree, 0.08, 2.40)
    
    # Latent risk score calculation
    latent_risk = (
        0.035 * (age - 45) +
        0.07 * (bmi - 25) +
        0.03 * (glucose - 100) +
        0.7 * (diabetes_pedigree - 0.4)
    )
    
    def symptom_prob(base_prob, risk_coeff):
        probs = 1 / (1 + np.exp(-(latent_risk * risk_coeff + np.log(base_prob / (1 - base_prob)))))
        return np.random.binomial(1, probs)
    
    polyuria = symptom_prob(0.38, 1.2)           # Frequent urination
    polydipsia = symptom_prob(0.35, 1.3)         # Excessive thirst
    sudden_weight_loss = symptom_prob(0.28, 1.0)
    weakness = symptom_prob(0.40, 0.6)
    polyphagia = symptom_prob(0.32, 0.8)         # Excessive hunger
    genital_thrush = symptom_prob(0.20, 0.6)
    visual_blurring = symptom_prob(0.26, 0.7)
    itching = symptom_prob(0.32, 0.5)
    irritability = symptom_prob(0.22, 0.6)
    delayed_healing = symptom_prob(0.25, 0.7)
    partial_paresis = symptom_prob(0.18, 0.6)
    muscle_stiffness = symptom_prob(0.28, 0.5)
    alopecia = symptom_prob(0.22, 0.4)
    obesity = np.where(bmi >= 30.0, 1, np.random.binomial(1, 0.12, size=n_samples))
    
    # Calculate final diagnostic target
    symptom_score = (
        2.2 * polyuria + 
        2.1 * polydipsia + 
        1.3 * sudden_weight_loss + 
        1.0 * polyphagia + 
        0.9 * visual_blurring + 
        0.8 * delayed_healing + 
        0.7 * weakness +
        0.6 * obesity
    )
    
    final_log_odds = -3.8 + 0.65 * latent_risk + 0.65 * symptom_score
    target_prob = 1 / (1 + np.exp(-final_log_odds))
    target = np.random.binomial(1, target_prob)
    
    binary_map = {1: 'Yes', 0: 'No'}
    
    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Polyuria': [binary_map[x] for x in polyuria],
        'Polydipsia': [binary_map[x] for x in polydipsia],
        'Sudden_Weight_Loss': [binary_map[x] for x in sudden_weight_loss],
        'Weakness': [binary_map[x] for x in weakness],
        'Polyphagia': [binary_map[x] for x in polyphagia],
        'Genital_Thrush': [binary_map[x] for x in genital_thrush],
        'Visual_Blurring': [binary_map[x] for x in visual_blurring],
        'Itching': [binary_map[x] for x in itching],
        'Irritability': [binary_map[x] for x in irritability],
        'Delayed_Healing': [binary_map[x] for x in delayed_healing],
        'Partial_Paresis': [binary_map[x] for x in partial_paresis],
        'Muscle_Stiffness': [binary_map[x] for x in muscle_stiffness],
        'Alopecia': [binary_map[x] for x in alopecia],
        'Obesity': [binary_map[x] for x in obesity],
        'Glucose_Level': glucose,
        'BMI': bmi,
        'Systolic_BP': systolic_bp,
        'Diabetes_Pedigree': diabetes_pedigree,
        'Class': target
    })
    
    return df

if __name__ == '__main__':
    df = generate_diabetes_dataset(n_samples=1200, random_state=42)
    df.to_csv('diabetes_data.csv', index=False)
    print(f"Dataset generated successfully! Shape: {df.shape}")
    print(f"Class balance:\n{df['Class'].value_counts(normalize=True)}")
