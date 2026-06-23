"""
train_model.py
----------------
End-to-end training pipeline for the airline tweet sentiment classifier.

Steps:
    1. Load Tweets.csv (Twitter US Airline Sentiment dataset)
    2. Clean text with preprocessing.clean_text (tokenize, remove stopwords,
       lemmatize - via NLTK)
    3. Vectorize cleaned text with TF-IDF
    4. Train a Logistic Regression classifier (class_weight='balanced' to
       offset the dataset's heavy skew toward negative tweets)
    5. Evaluate on a held-out test split: accuracy, precision, recall,
       F1-score (per class + macro/weighted averages)
    6. Plot and save a confusion matrix
    7. Persist the fitted vectorizer + model to disk so the Streamlit app
       can load them instantly without retraining

Run with:  python3 train_model.py
"""

import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from preprocessing import clean_text

DATA_PATH = "data/Tweets.csv"
MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
METRICS_PATH = "outputs/evaluation_metrics.json"
CONFUSION_MATRIX_PATH = "outputs/confusion_matrix.png"
LABEL_ORDER = ["negative", "neutral", "positive"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["text", "airline_sentiment"]].dropna()
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def main():
    print("=" * 60)
    print("STEP 1/6 - Loading dataset")
    print("=" * 60)
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} unique, non-null tweets")
    print(df["airline_sentiment"].value_counts(), "\n")

    print("=" * 60)
    print("STEP 2/6 - Cleaning text (NLTK: tokenize, stopwords, lemmatize)")
    print("=" * 60)
    t0 = time.time()
    df["clean_text"] = df["text"].apply(clean_text)
    # Drop rows that became empty after cleaning (e.g. tweet was just a URL/mention)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"Done in {time.time() - t0:.1f}s. Example:")
    print(" RAW  :", df["text"].iloc[0])
    print(" CLEAN:", df["clean_text"].iloc[0], "\n")

    print("=" * 60)
    print("STEP 3/6 - Train / test split (80 / 20, stratified)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["airline_sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["airline_sentiment"],
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}\n")

    print("=" * 60)
    print("STEP 4/6 - TF-IDF vectorization")
    print("=" * 60)
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),   # unigrams + bigrams (captures "not good" etc.)
        min_df=2,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}\n")

    print("=" * 60)
    print("STEP 5/6 - Training Logistic Regression classifier")
    print("=" * 60)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # dataset is ~63% negative, 21% neutral, 16% positive
        C=5.0,
        random_state=42,
    )
    model.fit(X_train_tfidf, y_train)
    print("Training complete.\n")

    print("=" * 60)
    print("STEP 6/6 - Evaluation")
    print("=" * 60)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    report_str = classification_report(y_test, y_pred, labels=LABEL_ORDER, digits=3)
    report_dict = classification_report(
        y_test, y_pred, labels=LABEL_ORDER, digits=3, output_dict=True
    )

    print(f"Accuracy            : {accuracy:.4f}")
    print(f"Precision (macro)   : {precision_macro:.4f}")
    print(f"Recall    (macro)   : {recall_macro:.4f}")
    print(f"F1-score  (macro)   : {f1_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall    (weighted): {recall_weighted:.4f}")
    print(f"F1-score  (weighted): {f1_weighted:.4f}\n")
    print("Per-class report:")
    print(report_str)

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Confusion Matrix - Airline Tweet Sentiment\n(Logistic Regression + TF-IDF)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

    # ---- Save metrics to JSON for reference / the Streamlit app ----
    metrics_out = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "per_class_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "labels": LABEL_ORDER,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    # ---- Persist model + vectorizer ----
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()