"""This script loads the raw data, applies essential feature engineering, and saves the processed data.
 For simplicity, we'll use a ColumnTransformer to handle both numeric scaling and one-hot encoding."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pickle
import os
#defining file paths
RAW_DATA_PATH = 'data/raw/german_credit_data.csv'
PROCESSES_DATA_PATH = 'processed/processed_data.pkl'
PREPROCESSOR_PATH = 'model/preprocessor.pkl'

# The standard target column in the German Credit dataset
TARGET_COLUMN_ORIGINAL = 'Risk' 
TARGET_COLUMN_RENAMED = 'target' 

def load_data(path):
    """Loads the raw CSV data and renames the target column."""
    try:
        # Assuming the CSV uses ';' as a separator as per the original code
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {path}. Please check the path and file name.")
        return None
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Rename the original target column for consistency
    if TARGET_COLUMN_ORIGINAL in df.columns:
        df.rename(columns={TARGET_COLUMN_ORIGINAL: TARGET_COLUMN_RENAMED}, inplace=True)
    else:
        print(f"Warning: Target column '{TARGET_COLUMN_ORIGINAL}' not found. Check dataset structure.")
        # If the target is missing, the script will likely fail later.
        
    return df

def create_preprocessor(X_train):
    """Defines and fits the preprocessing pipeline (ColumnTransformer) ONLY on X_train."""
    
    # Define features based on the standard German Credit dataset structure
    # 'Job' is often treated as categorical in ML pipelines, though it's numeric in the raw data.
    numerical_features_subset = ['Duration', 'Credit amount', 'Age']
    categorical_features = ['Job', 'Housing', 'Sex', 'Saving accounts', 'Checking account', 'Purpose']

    # --- Preprocessor Steps ---
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features_subset),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any remaining columns untransformed
    )
    
    # Fit the processor ONLY on the training features (X_train)
    preprocessor.fit(X_train)
    return preprocessor

def main():
    # --- Setup and Loading ---
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    os.makedirs('processed', exist_ok=True)
    print("Loading and preparing data...")
    data = load_data(RAW_DATA_PATH)
    
    if data is None:
        return # Stop execution if file not loaded

    # 1. Separate Features (X) and Target (y)
    X = data.drop(TARGET_COLUMN_RENAMED, axis=1, errors='ignore')
    y = data[TARGET_COLUMN_RENAMED]
    
    # 2. Split Data INTO TRAIN/TEST (CRITICAL: Prevents Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y # Use stratify for classification target
    )

    # 3. Fit Preprocessor ONLY on Training Data
    preprocessor = create_preprocessor(X_train) 
    print("Preprocessor successfully defined and fitted on training data.")
    
    # 4. Transform Data
    print("Transforming data...")
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 5. Package and Save Processed Data (for downstream training)
    processed_data = {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train.values,
        'y_test': y_test.values
    }

    with open(PROCESSES_DATA_PATH, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Processed features and targets saved to: {PROCESSES_DATA_PATH}")

    # 6. Save the Preprocessor Object (for deployment/inference)
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"Preprocessor object saved to: {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    main()



        