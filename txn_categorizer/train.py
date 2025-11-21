import argparse, joblib, yaml, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def load_data(path):
    df = pd.read_csv(path)
    return df["merchant"], df["label"]

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

def main(args):
    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(model, args.output_model)
    print(f"Saved model to {args.output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_model", default="model.joblib")
    args = parser.parse_args()
    main(args)
