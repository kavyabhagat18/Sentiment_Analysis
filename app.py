"""
app.py
-------
Streamlit web app for the Twitter Airline Sentiment classifier.

Run with:
    streamlit run app.py

The app loads the pre-trained TF-IDF vectorizer + Logistic Regression model
(produced by train_model.py) and classifies any text the user pastes in as
Positive, Neutral, or Negative, along with the model's confidence for each
class.
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

from preprocessing import clean_text

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
METRICS_PATH = "outputs/evaluation_metrics.json"
CONFUSION_MATRIX_IMG = "outputs/confusion_matrix.png"

st.set_page_config(
    page_title="Airline Tweet Sentiment Classifier",
    page_icon="✈️",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


LABEL_DISPLAY = {
    "positive": ("Positive", "😊"),
    "neutral": ("Neutral", "😐"),
    "negative": ("Negative", "🙁"),
}

model, vectorizer = load_artifacts()
metrics = load_metrics()

st.title("✈️ Airline Tweet Sentiment Classifier")
st.write(
    "Paste any piece of feedback below (originally trained on airline tweets, "
    "but works on general feedback text too) and get an instant "
    "**Positive / Neutral / Negative** sentiment prediction."
)

if model is None or vectorizer is None:
    st.error(
        "Model artifacts not found. Please run `python3 train_model.py` "
        "first to train and save the model before launching this app."
    )
    st.stop()

text_input = st.text_area(
    "Enter feedback text",
    height=120,
    placeholder="e.g. The flight was delayed by 3 hours and no one explained why...",
)

col1, col2 = st.columns([1, 1])
with col1:
    classify_clicked = st.button("Classify sentiment", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if classify_clicked:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(text_input)
        if not cleaned:
            st.warning(
                "After cleaning, no meaningful words were left to analyse "
                "(text may contain only links, mentions, or stopwords)."
            )
        else:
            X = vectorizer.transform([cleaned])
            pred_label = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            class_order = model.classes_

            display_name, emoji = LABEL_DISPLAY.get(pred_label, (pred_label, ""))
            st.markdown(f"### Result: {emoji} **{display_name}**")

            proba_df = pd.DataFrame(
                {
                    "Sentiment": [LABEL_DISPLAY.get(c, (c, ""))[0] for c in class_order],
                    "Confidence": proba,
                }
            ).sort_values("Confidence", ascending=False)

            st.bar_chart(proba_df.set_index("Sentiment"))

            with st.expander("See cleaned text used by the model"):
                st.code(cleaned)

            with st.expander("See raw confidence scores"):
                st.dataframe(
                    proba_df.assign(Confidence=lambda d: (d["Confidence"] * 100).round(2).astype(str) + "%"),
                    hide_index=True,
                    use_container_width=True,
                )

st.divider()

with st.expander("📊 Model performance (on held-out test set)"):
    if metrics is None:
        st.info("Run train_model.py to generate evaluation metrics.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        m2.metric("Precision (wtd)", f"{metrics['precision_weighted']*100:.1f}%")
        m3.metric("Recall (wtd)", f"{metrics['recall_weighted']*100:.1f}%")
        m4.metric("F1-score (wtd)", f"{metrics['f1_weighted']*100:.1f}%")

        st.caption(
            f"Trained on {metrics['train_size']} tweets, evaluated on "
            f"{metrics['test_size']} held-out tweets. "
            f"TF-IDF vocabulary size: {metrics['vocabulary_size']}."
        )

        if os.path.exists(CONFUSION_MATRIX_IMG):
            st.image(CONFUSION_MATRIX_IMG, caption="Confusion matrix on test set", use_container_width=True)

        per_class = metrics.get("per_class_report", {})
        rows = []
        for label in ["negative", "neutral", "positive"]:
            if label in per_class:
                rows.append(
                    {
                        "Class": label,
                        "Precision": round(per_class[label]["precision"], 3),
                        "Recall": round(per_class[label]["recall"], 3),
                        "F1-score": round(per_class[label]["f1-score"], 3),
                        "Support": int(per_class[label]["support"]),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.caption(
    "Model: TF-IDF + Logistic Regression · Preprocessing: NLTK tokenization, "
    "stopword removal, lemmatization · Dataset: Twitter US Airline Sentiment"
)