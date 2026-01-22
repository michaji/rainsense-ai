import pandas as pd
import numpy as np
import pickle
import os
import argparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# -------------------------------------------------------------------------
# 1. Configuration & Constants
# -------------------------------------------------------------------------
MODEL_PATH = 'models/rainfall_model.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'
DATA_PATH = 'data/train.csv'

# Optimized Hyperparameters (from our Best Optuna Trial)
params = {
    'n_estimators': 166,
    'max_depth': 7,
    'learning_rate': 0.29222896222670636,
    'subsample': 0.7191457777546555,
    'colsample_bytree': 0.6698285187076998,
    'gamma': 0.0013388765602434446,
    'min_child_weight': 2
}

# -------------------------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------------------------
def load_data(path):
    """Loads the dataset from the specified path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

def extract_date_features(df):
    """Extracts temporal features from prediction_time."""
    # Ensure datetime format
    df['prediction_time'] = pd.to_datetime(df['prediction_time'])
    
    # Extract features
    df['month'] = df['prediction_time'].dt.month
    df['day_of_year'] = df['prediction_time'].dt.dayofyear
    df['hour'] = df['prediction_time'].dt.hour
    df['day_of_week'] = df['prediction_time'].dt.dayofweek
    return df

def clean_data(df):
    """Performs basic cleaning and imputation."""
    # Impute categorical indicators
    df['indicator'] = df['indicator'].fillna('None')
    df['indicator_description'] = df['indicator_description'].fillna('')
    
    # Apply feature extraction
    df = extract_date_features(df)
    
    return df

def get_pipeline():
    """Defines the preprocessing and modeling pipeline."""
    
    # Define column groups
    categorical_cols = ['community', 'district', 'indicator']
    numerical_cols = ['confidence', 'predicted_intensity', 'forecast_length', 
                      'month', 'day_of_year', 'hour', 'day_of_week']

    # Preprocessing for numerical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Create the full pipeline with the optimized Classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(**params))
    ])
    
    return pipeline

# -------------------------------------------------------------------------
# 3. Main Execution
# -------------------------------------------------------------------------
def main():
    # A. Setup directories
    os.makedirs('models', exist_ok=True)

    # B. Load and Clean Data
    df = load_data(DATA_PATH)
    df = clean_data(df)
    
    # C. Prepare X and y
    # Drop columns that are not useful for training
    drop_cols = ['ID', 'user_id', 'indicator_description', 'time_observed', 'prediction_time', 'Target']
    X = df.drop(columns=drop_cols)
    y = df['Target']
    
    print(f"Training Data Shape: {X.shape}")
    print(f"Features used: {list(X.columns)}")

    # D. Encode Target
    # We must save this encoder to decode predictions later!
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Saving Target Label Encoder...")
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    
    # E. Train Model
    print("Initializing Pipeline with Optimized Parameters...")
    model_pipeline = get_pipeline()
    
    print("Training Model (this may take a moment)...")
    model_pipeline.fit(X, y_encoded)
    
    # F. Save Model
    print(f"Saving final model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_pipeline, f)
        
    print("✅ Training Complete. Model exported successfully.")

if __name__ == "__main__":
    main()