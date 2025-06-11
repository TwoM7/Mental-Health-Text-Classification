import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, models
from torch.nn import CrossEntropyLoss

import warnings
import os

os.environ["WANDB_DISABLED"] = "true"
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

# --- STAGE 1: FINE-TUNE SENTENCE-BERT MODEL---

class FineTuningDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item_idx):
        text = self.texts[item_idx]
        label = self.labels[item_idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def fine_tune_sbert(X_train, y_train, class_names):
    """
    Fine-tunes a Sentence-BERT model using a standard PyTorch training loop.
    """
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tuned_model_path = './sbert_finetuned_mental_health'
    num_classes = len(class_names)
    
    print(f"\n--- STAGE 1: Fine-Tuning {model_name} for Classification ---")
    
    if os.path.exists(tuned_model_path):
        print(f"Found existing fine-tuned model at '{tuned_model_path}'. Skipping fine-tuning.")
        return tuned_model_path

    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=num_classes,
        activation_function=torch.nn.Identity()
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    train_dataset = FineTuningDataset(
        texts=X_train.values,
        labels=y_train.values,
        tokenizer=model.tokenizer,
        max_len=128
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    

    loss_fct = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    print("Starting fine-tuning process... (This may take some time)")
    model.train()
    for epoch in range(1): 
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model({'input_ids': input_ids, 'attention_mask': attention_mask})['sentence_embedding']
            
            loss = loss_fct(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch 1 complete. Last batch loss: {loss.item():.4f}")

    model.save(tuned_model_path)
    print(f"Fine-tuning complete. Model saved to '{tuned_model_path}'.")
    return tuned_model_path


# --- STAGE 2: TRAIN LSTM WITH FINE-TUNED EMBEDDINGS ---

class MentalHealthDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_embeddings):
        text_embeddings = text_embeddings.unsqueeze(1)
        _, (hidden, cell) = self.lstm(text_embeddings)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

def train_final_classifier(tuned_sbert_path, X_train, y_train, X_test, y_test, class_names):
    print("\n--- STAGE 2: Training LSTM Classifier with Fine-Tuned Embeddings ---")

    print(f"Loading fine-tuned model from '{tuned_sbert_path}'...")
    sbert_model = SentenceTransformer(tuned_sbert_path)
    
    print("Generating embeddings with the new fine-tuned model...")
    X_train_embeddings = sbert_model.encode(X_train.tolist(), show_progress_bar=True, batch_size=64)
    X_test_embeddings = sbert_model.encode(X_test.tolist(), show_progress_bar=True, batch_size=64)
    
    train_dataset = MentalHealthDataset(X_train_embeddings, y_train.values)
    test_dataset = MentalHealthDataset(X_test_embeddings, y_test.values)
    
    BATCH_SIZE = 32
    EMBEDDING_DIM = X_train_embeddings.shape[1]
    HIDDEN_DIM = 128
    OUTPUT_DIM = len(class_names)
    N_LAYERS = 2
    DROPOUT = 0.5
    N_EPOCHS = 15
    LEARNING_RATE = 1e-4

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CrossEntropyLoss()

    print("Starting final classifier training...")
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(embeddings)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch: {epoch+1:02}/{N_EPOCHS} | Train Loss: {epoch_loss/len(train_loader):.4f}')
    
    print("\nTraining finished. Evaluating final model on the test set...")
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    final_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nFINAL TEST ACCURACY: {final_accuracy:.4f}")
    
    print("\n--- Final Per-Class Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    print("Generating final confusion matrix plot...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix for Fine-Tuned S-BERT + LSTM', fontsize=16)
    plt.tight_layout()
    plt.savefig("sbert_lstm_confusion_matrix_finetuned.png")
    print("Final confusion matrix saved as 'sbert_lstm_confusion_matrix_finetuned.png'")

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data()
    
    if X_train is not None:
        tuned_model_path = fine_tune_sbert(X_train, y_train, class_names)
        train_final_classifier(tuned_model_path, X_train, y_train, X_test, y_test, class_names)
        
        print("\nScript finished successfully.")
