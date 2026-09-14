"""
streamlit_app.py
Support Ticket Triage Classifier — interactive demo.

Design note: prediction logic is abstracted behind `predict()`, which
routes to whichever model is selected. Today only "Baseline (TF-IDF + LinearSVC)"
is functional. In Phase 4, after fine-tuning DistilBERT, fill in the
`predict_distilbert()` function below and flip DISTILBERT_READY to True —
no other part of this file needs to change.

Run with:
    streamlit run app/streamlit_app.py
"""

import json
import os

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CATEGORY_MODEL_PATH = "models/baseline_category_model.joblib"
URGENCY_MODEL_PATH = "models/baseline_urgency_model.joblib"
METRICS_SUMMARY_PATH = "reports/metrics_summary.json"

# Flip this to True once DistilBERT is fine-tuned and predict_distilbert()
# is implemented (Phase 4).
DISTILBERT_READY = True

DISTILBERT_CATEGORY_PATH = "models/distilbert_category"
DISTILBERT_URGENCY_PATH = "models/distilbert_urgency"


# ---------------------------------------------------------------------------
# Model loading (cached so it only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_baseline_models():
    category_model = joblib.load(CATEGORY_MODEL_PATH)
    urgency_model = joblib.load(URGENCY_MODEL_PATH)
    return category_model, urgency_model


@st.cache_resource
def load_distilbert_models():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    category_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_CATEGORY_PATH)
    category_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_CATEGORY_PATH)
    category_model.eval()

    urgency_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_URGENCY_PATH)
    urgency_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_URGENCY_PATH)
    urgency_model.eval()

    return {
        "category_tokenizer": category_tokenizer,
        "category_model": category_model,
        "urgency_tokenizer": urgency_tokenizer,
        "urgency_model": urgency_model,
    }


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def softmax(scores: np.ndarray) -> np.ndarray:
    """Convert LinearSVC decision_function scores into a confidence-like distribution."""
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def predict_with_pipeline(pipeline, text: str):
    """Returns (predicted_label, confidence) for a single sklearn pipeline."""
    scores = pipeline.decision_function([text])[0]
    probs = softmax(scores)
    pred_idx = int(np.argmax(probs))
    label = pipeline.classes_[pred_idx]
    confidence = float(probs[pred_idx])
    return label, confidence, dict(zip(pipeline.classes_, probs))


def predict_baseline(text: str):
    category_model, urgency_model = load_baseline_models()
    category, cat_conf, cat_dist = predict_with_pipeline(category_model, text)
    urgency, urg_conf, urg_dist = predict_with_pipeline(urgency_model, text)
    return {
        "category": category,
        "category_confidence": cat_conf,
        "category_distribution": cat_dist,
        "urgency": urgency,
        "urgency_confidence": urg_conf,
        "urgency_distribution": urg_dist,
    }


def predict_with_transformer(text: str, tokenizer, model):
    import torch

    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).numpy()
    pred_idx = int(np.argmax(probs))
    label = model.config.id2label[pred_idx]
    confidence = float(probs[pred_idx])
    distribution = {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}
    return label, confidence, distribution


def predict_distilbert(text: str):
    models = load_distilbert_models()

    category, cat_conf, cat_dist = predict_with_transformer(
        text, models["category_tokenizer"], models["category_model"]
    )
    urgency, urg_conf, urg_dist = predict_with_transformer(
        text, models["urgency_tokenizer"], models["urgency_model"]
    )
    return {
        "category": category,
        "category_confidence": cat_conf,
        "category_distribution": cat_dist,
        "urgency": urgency,
        "urgency_confidence": urg_conf,
        "urgency_distribution": urg_dist,
    }


def predict(text: str, model_choice: str):
    """Single entry point the UI calls. Routes to the selected model."""
    if model_choice == "Baseline (TF-IDF + LinearSVC)":
        return predict_baseline(text)
    elif model_choice == "DistilBERT (fine-tuned)":
        return predict_distilbert(text)
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")


# ---------------------------------------------------------------------------
# Metrics summary loading (produced by src/evaluate.py)
# ---------------------------------------------------------------------------
@st.cache_data
def load_metrics_summary():
    if not os.path.exists(METRICS_SUMMARY_PATH):
        return None
    with open(METRICS_SUMMARY_PATH) as f:
        return json.load(f)


def metrics_to_dataframe(summary: dict, task: str) -> pd.DataFrame:
    """Turns {"baseline": {...}, "distilbert": {...}} for one task into a tidy table."""
    rows = []
    label_map = {"baseline": "Baseline (TF-IDF + LinearSVC)", "distilbert": "DistilBERT (fine-tuned)"}
    for model_key, metrics in summary.get(task, {}).items():
        rows.append({
            "Model": label_map.get(model_key, model_key),
            "Accuracy": metrics.get("accuracy"),
            "F1 (macro)": metrics.get("f1_macro"),
            "F1 (weighted)": metrics.get("f1_weighted"),
            "Latency (ms/prediction)": metrics.get("avg_latency_ms"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Support Ticket Triage Classifier", page_icon="🎫", layout="centered")

st.title("🎫 Support Ticket Triage Classifier")

tab_classify, tab_metrics = st.tabs(["Classify a Ticket", "Model Performance"])

# ---------------------------------------------------------------------------
# Tab 1: live classification
# ---------------------------------------------------------------------------
with tab_classify:
    st.write("Paste a customer support ticket below to see its predicted category and urgency.")

    model_options = ["Baseline (TF-IDF + LinearSVC)", "DistilBERT (fine-tuned)"]
    if not DISTILBERT_READY:
        model_options[1] += " — coming soon"

    model_choice = st.radio("Model", model_options, horizontal=True)

    ticket_text = st.text_area(
        "Ticket text",
        height=150,
        placeholder="e.g. My GoPro Hero keeps crashing every time I try to open it. This is urgent, it's blocking my work.",
    )

    compare_mode = False
    if DISTILBERT_READY:
        compare_mode = st.checkbox("Compare Baseline vs DistilBERT side by side")

    run_button = st.button("Classify Ticket", type="primary")

    if run_button:
        if not ticket_text.strip():
            st.warning("Please enter some ticket text first.")
        else:
            if compare_mode:
                col1, col2 = st.columns(2)
                for col, choice in zip(
                    [col1, col2],
                    ["Baseline (TF-IDF + LinearSVC)", "DistilBERT (fine-tuned)"],
                ):
                    with col:
                        st.subheader(choice)
                        result = predict(ticket_text, choice)
                        st.metric("Category", result["category"], f"{result['category_confidence']:.1%} confidence")
                        st.metric("Urgency", result["urgency"], f"{result['urgency_confidence']:.1%} confidence")
            else:
                if "coming soon" in model_choice:
                    st.error("DistilBERT isn't fine-tuned yet — select the baseline model for now.")
                else:
                    result = predict(ticket_text, model_choice)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Category", result["category"])
                        st.caption(f"Confidence: {result['category_confidence']:.1%}")
                    with col2:
                        st.metric("Predicted Urgency", result["urgency"])
                        st.caption(f"Confidence: {result['urgency_confidence']:.1%}")

                    with st.expander("See full confidence breakdown"):
                        st.write("**Category distribution:**")
                        st.bar_chart(result["category_distribution"])
                        st.write("**Urgency distribution:**")
                        st.bar_chart(result["urgency_distribution"])

    st.divider()
    st.caption(
        "Note: confidence scores for the baseline model are derived from SVM decision "
        "margins via softmax, not true calibrated probabilities."
    )

# ---------------------------------------------------------------------------
# Tab 2: benchmark metrics from src/evaluate.py (reports/metrics_summary.json)
# ---------------------------------------------------------------------------
with tab_metrics:
    st.write("Benchmark results from the held-out test set (generated by `src/evaluate.py`).")

    summary = load_metrics_summary()

    if summary is None:
        st.info(
            "No metrics found yet. Run `python src/evaluate.py` first to generate "
            "`reports/metrics_summary.json`, then reload this page."
        )
    else:
        for task in ["category", "urgency"]:
            if task not in summary or not summary[task]:
                continue

            st.subheader(f"{task.capitalize()} classification")
            df = metrics_to_dataframe(summary, task)

            st.dataframe(
                df.style.format({
                    "Accuracy": "{:.3f}",
                    "F1 (macro)": "{:.3f}",
                    "F1 (weighted)": "{:.3f}",
                    "Latency (ms/prediction)": "{:.2f}",
                }),
                hide_index=True,
                use_container_width=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.caption("F1 (macro) comparison")
                st.bar_chart(df.set_index("Model")["F1 (macro)"])
            with col2:
                st.caption("Inference latency comparison (ms)")
                st.bar_chart(df.set_index("Model")["Latency (ms/prediction)"])

            st.divider()

        st.caption(
            "Latency is measured per single prediction on CPU. DistilBERT's contextual "
            "understanding comes at a meaningful latency cost — worth weighing against "
            "any accuracy gain for a given use case."
        )