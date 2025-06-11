import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

#  Data loading and preprocessing

def load_and_preprocess_data(filepath="Combined Data.csv"):

    print(f"1. Loading and preprocessing data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None, None, None, None, None

    def clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'[‘’“”…]', '', text)
        text = re.sub(r'\n', ' ', text).strip()
        return text

    df['cleaned_text'] = df['statement'].apply(clean_text)
    df.dropna(subset=['cleaned_text', 'status'], inplace=True)
    df = df[df['cleaned_text'] != '']

    print("vectorizing now")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(df['cleaned_text'])

    df['label_encoded'] = df['status'].astype('category').cat.codes
    y = df['label_encoded']
    class_names = df['status'].astype('category').cat.categories.tolist()
    
    print("Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("balancing data using SMOTE")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("Preprocessing complete.")
    return X_train_resampled, X_test, y_train_resampled, y_test, class_names

# Training and evaluation

def run_models(X_train, X_test, y_train, y_test, class_names):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Naive Bayes": MultinomialNB(),
        "SGD": SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    }

    for name, model in models.items():
        print(f"\n{'='*20} {name} {'='*20}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
        
        acc = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        print(f"Accuracy Score: {acc * 100:.4f}%")
        print(f"Balanced Accuracy Score: {balanced_acc * 100:.4f}%")

        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues', display_labels=class_names)
        plt.title(f'Confusion Matrix - {name}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    X_train_res, X_test_tfidf, y_train_res, y_test_labels, classes = load_and_preprocess_data()
    
    if X_train_res is not None:
        run_models(X_train_res, X_test_tfidf, y_train_res, y_test_labels, classes)
