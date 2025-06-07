import os
import sys
import joblib
import streamlit as st

# Setup paths
APP_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_ROOT)

from preprocess import preprocess_text

# Load model and vectorizer
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

# Label map
category_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

# Streamlit App UI
st.set_page_config(page_title="News Topic Classifier", layout="centered")
st.title("📰 News Topic Classification")
st.write("Enter a news article or headline to predict its category:")

# Input box
user_input = st.text_area("News Text", height=200)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some news text to classify.")
    else:
        # Preprocess
        clean_text = preprocess_text(user_input)
        # Vectorize
        X = vectorizer.transform([clean_text])
        # Predict
        prediction_num = model.predict(X)[0]
        prediction = category_map.get(int(prediction_num), "Unknown")

        # Output
        st.success(f"**Predicted Category:** {prediction}")
