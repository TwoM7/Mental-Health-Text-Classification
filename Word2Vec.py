import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
import warnings

warnings.filterwarnings('ignore')

# Data loading and preprocessing

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
    print("loading and preprocessing are done")
    return X_train, X_test, y_train, y_test, class_names

# Embedding and classification

def run_word2vec_model(X_train, X_test, y_train, y_test, class_names):
    """
    Trains a Word2Vec model and uses it with a Random Forest classifier.
    """
    print("\n--- Training Word2Vec Model ---")
    tokenized_train = [doc.split() for doc in X_train]
    
    w2v_model = Word2Vec(sentences=tokenized_train, vector_size=100, window=5, min_count=1, workers=4)
    print("Word2Vec model trained.")

    def document_vector(doc, model):
        doc_vec = [model.wv[word] for word in doc.split() if word in model.wv]
        if not doc_vec:
            return np.zeros(model.vector_size)
        return np.mean(doc_vec, axis=0)

    X_train_w2v = np.array([document_vector(doc, w2v_model) for doc in X_train])
    X_test_w2v = np.array([document_vector(doc, w2v_model) for doc in X_test])

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train_w2v, y_train)

    y_pred = rf_clf.predict(X_test_w2v)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nWord2Vec + RF Accuracy: {acc:.4f}")

    print("\n--- Classification Report (Word2Vec + RF) ---")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    return y_pred

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data()
    
    if X_train is not None:
        run_word2vec_model(X_train, X_test, y_train, y_test, class_names)
