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

import numpy as np
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CATEGORY_MODEL_PATH = "models/baseline_category_model.joblib"
URGENCY_MODEL_PATH = "models/baseline_urgency_model.joblib"

# Flip this to True once DistilBERT is fine-tuned and predict_distilbert()
# is implemented (Phase 4).
DISTILBERT_READY = False


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
    """
    PHASE 4 TODO:
    Load your fine-tuned DistilBERT model(s)/tokenizer(s) here, e.g.:

        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("models/distilbert_category")
        category_model = AutoModelForSequenceClassification.from_pretrained("models/distilbert_category")
        ... same for urgency ...
        return tokenizer, category_model, urgency_model

    Return whatever predict_distilbert() below needs.
    """
    return None


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


def predict_distilbert(text: str):
    """
    PHASE 4 TODO: implement inference with your fine-tuned DistilBERT model(s).
    Must return the same dict shape as predict_baseline() so the rest of the
    app doesn't need to change:

        {
            "category": str, "category_confidence": float, "category_distribution": dict,
            "urgency": str, "urgency_confidence": float, "urgency_distribution": dict,
        }
    """
    raise NotImplementedError("DistilBERT inference not implemented yet — coming in Phase 4.")


def predict(text: str, model_choice: str):
    """Single entry point the UI calls. Routes to the selected model."""
    if model_choice == "Baseline (TF-IDF + LinearSVC)":
        return predict_baseline(text)
    elif model_choice == "DistilBERT (fine-tuned)":
        return predict_distilbert(text)
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Support Ticket Triage Classifier", page_icon="🎫", layout="centered")

st.title("🎫 Support Ticket Triage Classifier")
st.write("Paste a customer support ticket below to see its predicted category and urgency.")

model_options = ["Baseline (TF-IDF + LinearSVC)"]
if DISTILBERT_READY:
    model_options.append("DistilBERT (fine-tuned)")
else:
    model_options.append("DistilBERT (fine-tuned) — coming soon")

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