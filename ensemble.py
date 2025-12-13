import os
import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm
import warnings
from pathlib import Path
import joblib

# --- Deep Learning & Utility Imports ---
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
tqdm.pandas()

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculates SMAPE and returns it as a percentage."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100

def main():
    """
    This script runs the definitive single-model pipeline, carefully designed to
    prevent overfitting and produce a high-quality submission.
    """
    print("--- Amazon ML Challenge: Final Submission Pipeline (Single Model) ---")

    # --- Path Definitions ---
    DATASET_FOLDER, IMAGE_FOLDER, FEATURES_FOLDER, MODELS_FOLDER = './dataset/', './images/', './features/', './models/'
    TRAIN_FEATURES_PATH = os.path.join(FEATURES_FOLDER, 'train_resnet50v2_features.npy')
    TEST_FEATURES_PATH = os.path.join(FEATURES_FOLDER, 'test_resnet50v2_features.npy')
    MODEL_SAVE_PATH = os.path.join(MODELS_FOLDER, 'final_submission_model.pkl')
    SUBMISSION_PATH = os.path.join(DATASET_FOLDER, 'test_out_final_submission.csv')
    
    for folder in [FEATURES_FOLDER, MODELS_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    # --- Data & Feature Loading ---
    print("--- Loading data and pre-computed features ---")
    full_train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    if not os.path.exists(TRAIN_FEATURES_PATH) or not os.path.exists(TEST_FEATURES_PATH):
        print("ERROR: Feature files (.npy) not found. Please run 'extract_features_only.py' first.")
        return
        
    X_train_full_image_features = np.load(TRAIN_FEATURES_PATH)
    X_test_image_features = np.load(TEST_FEATURES_PATH)
    print("Data and features loaded successfully.")

    # --- Text/Tabular Feature Engineering ---
    def parse_catalog_content(df):
        df['value'] = df['catalog_content'].str.extract(r"Value:\s*([0-9]+\.?[0-9]*)", expand=False).astype(float)
        df['unit'] = df['catalog_content'].str.extract(r"Unit:\s*(\w+)", expand=False).str.lower()
        df['text_features'] = df['catalog_content'].str.replace(r"Value:.*|Unit:.*", "", regex=True).str.strip()
        df['value'] = df['value'].fillna(df['value'].median())
        df['unit'] = df['unit'].fillna('unknown')
        df['text_features'] = df['text_features'].fillna('missing')
        return df
    full_train_df = parse_catalog_content(full_train_df)
    test_df = parse_catalog_content(test_df)

    # --- [Step 1] Validation & Finding Optimal Iterations ---
    print("\n--- [Step 1/2] Validating model and finding optimal training rounds ---")
    indices = np.arange(full_train_df.shape[0])
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_df = full_train_df.iloc[train_indices]
    val_df = full_train_df.iloc[val_indices]
    y_train = train_df['price']
    y_val = val_df['price']
    
    X_train_image_features = X_train_full_image_features[train_indices]
    X_val_image_features = X_train_full_image_features[val_indices]

    tfidf = TfidfVectorizer(max_features=20000, min_df=5, stop_words='english', ngram_range=(1, 2))
    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown='ignore')
    
    X_train_text = tfidf.fit_transform(train_df['text_features'].values)
    X_train_num = scaler.fit_transform(train_df[['value']].values)
    X_train_cat = ohe.fit_transform(train_df[['unit']].values)
    
    X_val_text = tfidf.transform(val_df['text_features'].values)
    X_val_num = scaler.transform(val_df[['value']].values)
    X_val_cat = ohe.transform(val_df[['unit']].values)

    X_train_combined = hstack([X_train_text, X_train_num, X_train_cat, csr_matrix(X_train_image_features)]).tocsr()
    X_val_combined = hstack([X_val_text, X_val_num, X_val_cat, csr_matrix(X_val_image_features)]).tocsr()

    # Use a strong, well-regularized set of parameters
    lgbm_params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'device': 'gpu', 'random_state': 42,
        'n_estimators': 5000, # High number for early stopping
        'learning_rate': 0.02,
        'num_leaves': 60,
        'max_depth': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    
    val_lgbm = lgb.LGBMRegressor(**lgbm_params)
    val_lgbm.fit(X_train_combined, y_train, eval_set=[(X_val_combined, y_val)], callbacks=[lgb.early_stopping(100, verbose=True)])
    
    # This is the optimal number of training rounds found
    optimal_iterations = val_lgbm.best_iteration_
    print(f"\nOptimal number of training rounds found: {optimal_iterations}")
    
    val_preds = val_lgbm.predict(X_val_combined)
    smape = symmetric_mean_absolute_percentage_error(y_val, val_preds)
    
    print("\n" + "="*60)
    print("     FINAL MODEL VALIDATION SCORE")
    print("="*60)
    print(f"Estimated SMAPE: {smape:.4f}%")
    print("="*60)

    # --- [Step 2] Final Training and Prediction ---
    print("\n--- [Step 2/2] Training final model on ALL data and creating submission ---")
    
    tfidf_final = TfidfVectorizer(max_features=20000, min_df=5, stop_words='english', ngram_range=(1, 2)).fit(full_train_df['text_features'].values)
    scaler_final = StandardScaler().fit(full_train_df[['value']].values)
    ohe_final = OneHotEncoder(handle_unknown='ignore').fit(full_train_df[['unit']].values)

    X_train_full_text = tfidf_final.transform(full_train_df['text_features'].values)
    X_train_full_num = scaler_final.transform(full_train_df[['value']].values)
    X_train_full_cat = ohe_final.transform(full_train_df[['unit']].values)
    X_train_full_combined = hstack([X_train_full_text, X_train_full_num, X_train_full_cat, csr_matrix(X_train_full_image_features)]).tocsr()
    
    X_test_text = tfidf_final.transform(test_df['text_features'].values)
    X_test_num = scaler_final.transform(test_df[['value']].values)
    X_test_cat = ohe_final.transform(test_df[['unit']].values)
    X_test_combined = hstack([X_test_text, X_test_num, X_test_cat, csr_matrix(X_test_image_features)]).tocsr()

    # Create the final model with the optimal number of iterations
    final_model_params = lgbm_params.copy()
    final_model_params['n_estimators'] = optimal_iterations
    
    final_model = lgb.LGBMRegressor(**final_model_params)
    final_model.fit(X_train_full_combined, full_train_df['price'])
    print("Final model trained on 100% of the training data.")
    
    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")
    
    final_predictions = final_model.predict(X_test_combined)
    final_predictions[final_predictions < 0] = 0

    submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': final_predictions})
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission file created at: {SUBMISSION_PATH}")
    
    print("\n--- Process Finished ---")

if __name__ == "__main__":
    main()
