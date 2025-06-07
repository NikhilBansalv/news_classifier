import os
import sys
import joblib
import streamlit as st
from transformers import pipeline

# ─── Page config MUST come before any Streamlit commands ────────────────────
st.set_page_config(page_title="News Classifier & Summarizer", layout="centered")

# ─── Path Setup ─────────────────────────────────────────────────────────────
HERE    = os.path.dirname(__file__)          # NEWS_CLASSIFIER/
SRC_DIR = os.path.join(HERE, "src")          # NEWS_CLASSIFIER/src
sys.path.insert(0, SRC_DIR)                  # Prepend src/ to Python path

from preprocess import preprocess_text       # Now import works

# ─── Load Vectorizer & Classifier ───────────────────────────────────────────
MODELS_DIR      = os.path.join(HERE, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH      = os.path.join(MODELS_DIR, "best_model.pkl")

vectorizer = joblib.load(VECTORIZER_PATH)
model      = joblib.load(MODEL_PATH)

# ─── Summarizer Pipeline ──────────────────────────────────────────────────────
@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        tokenizer="sshleifer/distilbart-cnn-12-6",
        device=-1
    )

summarizer = load_summarizer()

# ─── Category Map ─────────────────────────────────────────────────────────────
category_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("📰 News Topic Classifier & Summarizer")

user_input = st.text_area("Enter your news text here:", height=200)

if st.button("Classify & Summarize"):
    if not user_input.strip():
        st.warning("Please enter some text before proceeding.")
    else:
        # Classification
        clean_text_input = preprocess_text(user_input)
        X_input          = vectorizer.transform([clean_text_input])
        pred_num         = model.predict(X_input)[0]
        pred_cat         = category_map.get(int(pred_num), "Unknown")
        st.subheader(f"**Predicted Category:** {pred_cat}")

        # Summarization
        with st.spinner("Generating summary..."):
            try:
                summary = summarizer(
                    user_input,
                    max_length=200,
                    min_length=90,
                    do_sample=False
                )[0]["summary_text"]
                st.subheader("📝 Summary")
                st.write(summary)
            except Exception:
                st.info("Unable to summarize (text may be too short).")
                st.write(user_input)
