import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_eda(df, output_dir='plots'):
    """Performs EDA and generates visual insights, saving figures to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Target Class Distribution Plot
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x='Class', hue='Class', palette=['#3498db', '#e74c3c'], legend=False)
    plt.title('Early Diabetes Class Distribution (0=Negative, 1=Positive)', fontsize=13, fontweight='bold')
    plt.xlabel('Diagnostic Outcome (0 = Low Risk, 1 = High Risk)', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Key Symptoms Prevalence by Diabetes Class
    symptom_cols = [
        'Polyuria', 'Polydipsia', 'Sudden_Weight_Loss', 'Weakness', 'Polyphagia',
        'Visual_Blurring', 'Delayed_Healing', 'Obesity'
    ]
    
    symptom_data = []
    for sym in symptom_cols:
        pos_rate = (df[df['Class'] == 1][sym] == 'Yes').mean() * 100
        neg_rate = (df[df['Class'] == 0][sym] == 'Yes').mean() * 100
        symptom_data.append({'Symptom': sym, 'Positive_Group_%': pos_rate, 'Negative_Group_%': neg_rate})
        
    sym_df = pd.DataFrame(symptom_data).melt(id_vars='Symptom', var_name='Group', value_name='Prevalence_%')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=sym_df, x='Symptom', y='Prevalence_%', hue='Group', palette=['#e74c3c', '#2ecc71'])
    plt.title('Early Symptoms Prevalence: Diabetes Positive vs. Negative Group', fontsize=14, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontsize=11)
    plt.ylabel('Prevalence Rate (%)', fontsize=12)
    plt.legend(title='Patient Group', labels=['Diabetes Positive', 'Diabetes Negative'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'symptoms_prevalence.png'), dpi=300)
    plt.close()
    
    # 3. Glucose & BMI Distribution by Class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(data=df, x='Glucose_Level', hue='Class', kde=True, ax=axes[0], palette=['#3498db', '#e74c3c'], element='step')
    axes[0].set_title('Glucose Level Distribution by Diabetes Risk', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Fasting Glucose Level (mg/dL)', fontsize=11)
    
    sns.histplot(data=df, x='BMI', hue='Class', kde=True, ax=axes[1], palette=['#3498db', '#e74c3c'], element='step')
    axes[1].set_title('Body Mass Index (BMI) Distribution by Diabetes Risk', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('BMI (kg/m²)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'glucose_bmi_distribution.png'), dpi=300)
    plt.close()
    
    # 4. Feature Correlation Matrix
    numeric_df = df.copy()
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 1: 1, 0: 0}
    for col in numeric_df.columns:
        numeric_df[col] = numeric_df[col].map(lambda x: binary_map.get(x, x))
        
    plt.figure(figsize=(14, 11))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
    plt.close()
    
    print(f"EDA visual plots generated and saved to '{output_dir}/'!")

if __name__ == '__main__':
    df = pd.read_csv('diabetes_data.csv')
    run_eda(df)
