# Mental-Health-Text-Classification

This repository contains the official code and resources for the paper:
**"Mental Health Classification from Text: An NLP-Driven Evaluation of Computational Models"**

## Abstract
Natural Language Processing (NLP) serves as an invaluable instrument for the investigation of mental health. It has the potential to enhance diagnostic accuracy, facilitate timely interventions, and optimize therapeutic outcomes. This research examines the efficacy of various NLP methodologies and machine learning algorithms in the classification of mental health disorders through textual data. The investigators utilized a corpus comprising 53,042 textual entries sourced from mental health discussion platforms and anonymized user interactions. The dataset was meticulously prepared and balanced, employing a technique known as SMOTE. Analytical approaches such as TF-IDF were employed alongside more sophisticated methodologies like Word2Vec and BERT to achieve a deeper comprehension of the textual material. A total of six machine learning algorithms were evaluated: Logistic Regression, Naive Bayes, Stochastic Gradient Descent, K-Nearest Neighbors, Decision Tree, and Random Forest. Their performance was assessed using metrics such as accuracy, precision, recall, F1-score, and confusion matrix. The findings indicated that deep learning architectures leveraging word embeddings and LSTM networks outperformed conventional techniques utilizing TF-IDF. Nonetheless, there exist challenges including the interpretation of subjective language, the scarcity of diverse datasets, and heightened sensitivity to class imbalance and vague self-reported information. Future inquiries should aim at developing models tailored to specific domains, enhancing dataset diversity, and integrating varied data types to render mental health classification through NLP more robust and applicable.

## Keywords
NLP, Mental Health, Machine Learning, Text Data, Word Embeddings, Deep Learning, Large Language Models

## Repository Structure
- `main_model.py`: The main script to reproduce our best-performing model. It fine-tunes a Sentence-BERT model and then trains an LSTM classifier.
- `classical_models.py`: A script to reproduce the results for the six classical machine learning baselines (Logistic Regression, Random Forest, etc.).
- `Word2Vec.py`: A script to reproduce the results for the Word2Vec
- `E5_XGB_T5_RF.py`: A script to reproduce the results of our both T5 + Radnom forest and E5 + XGBoost
- `requirements.txt`: A list of all Python packages required to run the codes.

## Setup

### 1. Environment
It is recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Dataset
The dataset used in this study is the "Sentiment Analysis for Mental Health" which can be downloaded from Kaggle:
https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
Please download the dataset file and place it in the root directory of this project before running the scripts.
Don't forget to change filepath or dataset name for it to function correctly.
