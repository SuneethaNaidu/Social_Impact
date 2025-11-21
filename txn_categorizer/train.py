"""
txn_categorizer/train.py
Train and evaluate a TF-IDF + Logistic Regression baseline.
Saves model and metrics for reproducibility.
"""
import argparse
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    # expected columns: merchant,label
    df = df.dropna(subset=['merchant','label'])
    return df['merchant'].astype(str).tolist(), df['label'].astype(str).tolist()

def build_pipeline():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
        ('clf', LogisticRegression(max_iter=1000, class_weight=None))
    ])
    return pipe

def train(args):
    X, y = load_data(args.data)
    labels = sorted(list(set(y)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    print('Classification report:')
    print(classification_report(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds, labels=labels)
    print('Confusion matrix (rows=true, cols=pred):')
    print(labels)
    print(cm)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(pipe, args.output_model)

    # Save metrics and confusion matrix
    metrics = {'classification_report': report, 'labels': labels, 'confusion_matrix': cm.tolist()}
    with open(args.metrics_output, 'w') as f:
        yaml.safe_dump(metrics, f)

    print(f"Model saved to: {args.output_model}")
    print(f"Metrics saved to: {args.metrics_output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='CSV with merchant,label')
    parser.add_argument('--output_model', required=True, help='Output model joblib path')
    parser.add_argument('--metrics_output', default='metrics.yaml', help='YAML path to save metrics')
    args = parser.parse_args()
    train(args)
