import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_and_evaluate_models(processed_data, models_dir='models', plots_dir='plots'):
    """
    Trains multiple ML classification models, evaluates accuracy, precision, recall, F1, ROC-AUC,
    cross-validation, saves visual plots and outputs the benchmark report.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    scaler = processed_data['scaler']
    feature_names = processed_data['feature_names']
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss'),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    results = []
    trained_model_objs = {}
    roc_data = {}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "="*70)
    print("       EARLY DIABETES PREDICTION - MACHINE LEARNING BENCHMARK")
    print("="*70)
    
    best_f1 = -1
    best_model_name = None
    best_model_obj = None
    
    for name, model in models.items():
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
        mean_cv_f1 = np.mean(cv_scores)
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            '5-Fold CV F1': mean_cv_f1
        })
        
        trained_model_objs[name] = model
        
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data[name] = (fpr, tpr, auc)
            
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model
            
    results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    
    print(results_df.to_string(index=False))
    print("="*70)
    print(f"Top Performing Model: {best_model_name} (F1-Score: {best_f1:.4f})")
    print("="*70)
    
    # Save best model, scaler, and feature names
    joblib.dump(best_model_obj, os.path.join(models_dir, 'best_model.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.joblib'))
    
    # Save results summary table to CSV
    results_df.to_csv(os.path.join(models_dir, 'model_benchmark_results.csv'), index=False)
    
    # Plot ROC Curves
    plt.figure(figsize=(10, 7))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curves - Early Diabetes Classification Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curves.png'), dpi=300)
    plt.close()
    
    # Confusion Matrix for Best Model
    y_pred_best = best_model_obj.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Best Model ({best_model_name})', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'best_model_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Feature Importance for Random Forest / XGBoost / Tree models
    if hasattr(best_model_obj, 'feature_importances_'):
        importances = best_model_obj.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
        plt.title(f'Feature Importances - {best_model_name}', fontsize=13, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=11)
        plt.ylabel('Feature', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_importances.png'), dpi=300)
        plt.close()
        
    return {
        'results_df': results_df,
        'best_model_name': best_model_name,
        'best_model': best_model_obj,
        'trained_models': trained_model_objs
    }

if __name__ == '__main__':
    from data_preprocessing import load_data, preprocess_data
    df = load_data()
    processed = preprocess_data(df)
    train_and_evaluate_models(processed)
