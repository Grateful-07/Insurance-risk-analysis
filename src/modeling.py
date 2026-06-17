# src/modeling.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

def prepare_modeling_data(df):
    """
    Cleans missing values, engineers structural features, and encodes categoricals.
    """
    model_df = df.copy()
    
    # 1. Handle missing values safely
    numeric_cols = model_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        model_df[col] = model_df[col].fillna(model_df[col].median())
        
    cat_cols = model_df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        model_df[col] = model_df[col].fillna(model_df[col].mode()[0])
        
    # 2. Advanced Feature Engineering
    # Safeguard feature creation against missing raw column markers
    if 'Age' in model_df.columns and 'RiskScore' in model_df.columns:
        model_df['Age_Risk_Interaction'] = model_df['Age'] * model_df['RiskScore']
    else:
        model_df['Age_Risk_Interaction'] = model_df['RiskScore'] if 'RiskScore' in model_df.columns else 0
        
    # 3. Label Encoding for Categoricals
    le = LabelEncoder()
    features_to_encode = ['Gender', 'Province', 'VehicleType', 'AutoMake', 'CoverType']
    for col in features_to_encode:
        if col in model_df.columns:
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            
    return model_df

def evaluate_regression_models(X_train, X_test, y_train, y_test, models_dict):
    """
    Trains and tracks model scoring metrics across RMSE and R2 parameters.
    """
    metrics_log = []
    trained_models = {}
    
    for name, model in models_dict.items():
        # Fit model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict and evaluate
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        metrics_log.append({
            "Algorithm Model": name,
            "RMSE (Rand)": rmse,
            "R² Score": r2
        })
        
    return pd.DataFrame(metrics_log), trained_models