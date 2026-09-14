"""
evaluate.py
Evaluates baseline (TF-IDF + LinearSVC) and fine-tuned DistilBERT models
on the held-out test set. Produces:
  - classification_report (F1 macro + weighted) per model/task
  - confusion matrix plots saved to reports/
  - inference latency benchmark (baseline vs DistilBERT)
  - a combined metrics_summary.json for embedding in the README

Usage:
    python src/evaluate.py
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

TEST_PATH = "data/processed/test.csv"
REPORTS_DIR = "reports"
MODELS_DIR = "models"
TASKS = ["category", "urgency"]

os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_test_data():
    return pd.read_csv(TEST_PATH)


def load_baseline(task: str):
    path = f"{MODELS_DIR}/baseline_{task}_model.joblib"
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_distilbert(task: str):
    path = f"{MODELS_DIR}/distilbert_{task}"
    if not os.path.exists(path):
        return None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_baseline_batch(pipeline, texts):
    return pipeline.predict(texts)


def predict_distilbert_batch(tokenizer, model, texts):
    import torch

    preds = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
            logits = model(**inputs).logits[0]
            pred_idx = int(np.argmax(logits.numpy()))
            preds.append(model.config.id2label[pred_idx])
    return preds


def benchmark_latency_baseline(pipeline, texts, n=50):
    sample = texts[:n] if len(texts) >= n else texts
    start = time.perf_counter()
    for text in sample:
        pipeline.predict([text])
    elapsed = time.perf_counter() - start
    return (elapsed / len(sample)) * 1000  # ms per prediction


def benchmark_latency_distilbert(tokenizer, model, texts, n=50):
    import torch

    sample = texts[:n] if len(texts) >= n else texts
    start = time.perf_counter()
    with torch.no_grad():
        for text in sample:
            inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
            model(**inputs)
    elapsed = time.perf_counter() - start
    return (elapsed / len(sample)) * 1000  # ms per prediction


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/{filename}")
    plt.close()
    print(f"Saved confusion matrix -> {REPORTS_DIR}/{filename}")


def evaluate_model(y_true, y_pred, model_name, task):
    print(f"\n{'='*60}\n{model_name} — {task}\n{'='*60}")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, zero_division=0))

    labels = sorted(y_true.unique())
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}_{task}.png"
    save_confusion_matrix(y_true, y_pred, labels, f"{model_name} — {task}", filename)

    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "accuracy": report["accuracy"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    test_df = load_test_data()
    texts = test_df["ticket_text"].tolist()
    summary = {}

    for task in TASKS:
        y_true = test_df[task]
        summary[task] = {}

        # --- Baseline ---
        baseline_pipeline = load_baseline(task)
        if baseline_pipeline is not None:
            y_pred = predict_baseline_batch(baseline_pipeline, texts)
            metrics = evaluate_model(pd.Series(y_true), pd.Series(y_pred), "Baseline", task)
            latency = benchmark_latency_baseline(baseline_pipeline, texts)
            metrics["avg_latency_ms"] = latency
            print(f"Baseline avg inference latency: {latency:.2f} ms/prediction")
            summary[task]["baseline"] = metrics
        else:
            print(f"No baseline model found for task: {task}, skipping.")

        # --- DistilBERT ---
        distilbert = load_distilbert(task)
        if distilbert is not None:
            tokenizer, model = distilbert
            y_pred = predict_distilbert_batch(tokenizer, model, texts)
            metrics = evaluate_model(pd.Series(y_true), pd.Series(y_pred), "DistilBERT", task)
            latency = benchmark_latency_distilbert(tokenizer, model, texts)
            metrics["avg_latency_ms"] = latency
            print(f"DistilBERT avg inference latency: {latency:.2f} ms/prediction")
            summary[task]["distilbert"] = metrics
        else:
            print(f"No fine-tuned DistilBERT model found for task: {task}, skipping.")

    with open(f"{REPORTS_DIR}/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved combined metrics summary -> {REPORTS_DIR}/metrics_summary.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for task, models in summary.items():
        print(f"\n{task.upper()}:")
        for model_name, metrics in models.items():
            print(
                f"  {model_name:12s} | F1 macro: {metrics['f1_macro']:.3f} | "
                f"F1 weighted: {metrics['f1_weighted']:.3f} | "
                f"Latency: {metrics['avg_latency_ms']:.2f} ms"
            )


if __name__ == "__main__":
    main()