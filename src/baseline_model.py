"""
baseline_model.py
Trains two TF-IDF + LinearSVC pipelines on the processed ticket data:
  - one to predict `category` (Ticket Type)
  - one to predict `urgency`  (Ticket Priority)

Saves both trained pipelines to models/ using joblib.

Usage:
    python src/baseline_model.py
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
MODELS_DIR = "models"


def load_split(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_pipeline() -> Pipeline:
    """TF-IDF + LinearSVC pipeline. Same shape reused for both labels."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),   # unigrams + bigrams
            stop_words="english",
            min_df=2,
        )),
        ("clf", LinearSVC(
            C=1.0,
            class_weight="balanced",  # guards against any label imbalance
            random_state=42,
        )),
    ])


def train_and_evaluate(X_train, y_train, X_val, y_val, label_name: str) -> Pipeline:
    print(f"\n{'='*50}")
    print(f"Training model for: {label_name}")
    print(f"{'='*50}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    print(f"\nValidation performance for '{label_name}':")
    print(classification_report(y_val, y_pred))

    return pipeline


def main():
    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)

    X_train = train_df["ticket_text"]
    X_val = val_df["ticket_text"]

    # --- Category model ---
    category_model = train_and_evaluate(
        X_train, train_df["category"],
        X_val, val_df["category"],
        label_name="category",
    )
    joblib.dump(category_model, f"{MODELS_DIR}/baseline_category_model.joblib")

    # --- Urgency model ---
    urgency_model = train_and_evaluate(
        X_train, train_df["urgency"],
        X_val, val_df["urgency"],
        label_name="urgency",
    )
    joblib.dump(urgency_model, f"{MODELS_DIR}/baseline_urgency_model.joblib")

    print(f"\nSaved both models to {MODELS_DIR}/")


if __name__ == "__main__":
    main()