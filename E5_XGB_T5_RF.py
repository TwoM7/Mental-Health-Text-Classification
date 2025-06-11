import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# --- 1. DATA LOADING AND PREPROCESSING ---

def load_and_preprocess_data(filepath="Combined Data.csv"):
    print(f"1. Loading data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None, None, None, None, None

    def clean_text(text):
        if not isinstance(text, str): return ""
        return re.sub(r'\s+', ' ', text.lower()).strip()

    df['cleaned_text'] = df['statement'].apply(clean_text)
    df.dropna(subset=['cleaned_text', 'status'], inplace=True)
    df = df[df['cleaned_text'] != '']
    
    df['label_encoded'] = df['status'].astype('category').cat.codes
    labels = df['label_encoded']
    class_names = df['status'].astype('category').cat.categories.tolist()
    
    print("Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], labels, test_size=0.2, random_state=42, stratify=labels
    )
    print("Data loading and preprocessing complete.")
    return X_train, X_test, y_train, y_test, class_names

# --- 2. EMBEDDING GENERATION HELPER ---

def get_sentence_embeddings(texts, model_name):
    """Helper function to get embeddings from SentenceTransformer."""
    print(f"\n--- Generating embeddings using {model_name}... ---")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts.tolist(), show_progress_bar=True, batch_size=32)
    return embeddings

# --- 3. RUN EXPERIMENTS ---

def run_experiments(X_train, X_test, y_train, y_test, class_names):
    """
    Runs the E5 and T5 experiments and prints their reports.
    """
    
    # --- E5 + XGBoost Experiment ---
    print("\n==================== E5 + XGBoost ====================")
    X_train_e5 = get_sentence_embeddings(X_train, 'intfloat/e5-large-v2')
    X_test_e5 = get_sentence_embeddings(X_test, 'intfloat/e5-large-v2')
    
    print("\nTraining XGBoost classifier...")
    xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    xgb_clf.fit(X_train_e5, y_train)
    
    print("Evaluating E5 + XGBoost model...")
    y_pred_e5 = xgb_clf.predict(X_test_e5)
    acc_e5 = accuracy_score(y_test, y_pred_e5)
    
    print(f"\nE5 + XGBoost Accuracy: {acc_e5:.4f}")
    print("\nClassification Report (E5 + XGBoost):")
    print(classification_report(y_test, y_pred_e5, target_names=class_names, digits=4))

    # --- T5 + Random Forest Experiment ---
    print("\n==================== T5 + Random Forest ====================")
    X_train_t5 = get_sentence_embeddings(X_train, 'sentence-t5-base')
    X_test_t5 = get_sentence_embeddings(X_test, 'sentence-t5-base')

    print("\nTraining Random Forest classifier...")
    rf_clf_t5 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf_t5.fit(X_train_t5, y_train)

    print("Evaluating T5 + Random Forest model...")
    y_pred_t5 = rf_clf_t5.predict(X_test_t5)
    acc_t5 = accuracy_score(y_test, y_pred_t5)
    
    print(f"\nT5 + Random Forest Accuracy: {acc_t5:.4f}")
    print("\nClassification Report (T5 + Random Forest):")
    print(classification_report(y_test, y_pred_t5, target_names=class_names, digits=4))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data()
    
    if X_train is not None:
        run_experiments(X_train, X_test, y_train, y_test, class_names)
        print("\nScript finished successfully.")
