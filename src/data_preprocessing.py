import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath='diabetes_data.csv'):
    """Loads the raw dataset from CSV."""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Cleans, encodes categorical symptoms/gender, scales numerical features,
    and returns train/test splits along with preprocessors.
    """
    data = df.copy()
    
    # Identify binary categorical columns (Yes/No and Male/Female)
    binary_symptom_cols = [
        'Polyuria', 'Polydipsia', 'Sudden_Weight_Loss', 'Weakness', 'Polyphagia',
        'Genital_Thrush', 'Visual_Blurring', 'Itching', 'Irritability',
        'Delayed_Healing', 'Partial_Paresis', 'Muscle_Stiffness', 'Alopecia', 'Obesity'
    ]
    
    # Binary mapping: Yes -> 1, No -> 0
    for col in binary_symptom_cols:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
            
    # Gender mapping: Male -> 1, Female -> 0
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
        
    # Numerical features to standard scale
    num_cols = ['Age', 'Glucose_Level', 'BMI', 'Systolic_BP', 'Diabetes_Pedigree']
    
    X = data.drop(columns=['Class'])
    y = data['Class']
    
    # Train test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    feature_names = list(X.columns)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'num_cols': num_cols
    }

if __name__ == '__main__':
    df = load_data()
    processed = preprocess_data(df)
    print("Preprocessing completed successfully!")
    print("Train X shape:", processed['X_train'].shape)
    print("Test X shape:", processed['X_test'].shape)
