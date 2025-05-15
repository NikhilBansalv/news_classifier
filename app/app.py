import os
import sys
from flask import Flask, render_template, request
import joblib

# Ensure src/ is on the path so we can import preprocess_text
APP_ROOT = os.path.dirname(__file__)                   # .../NEWS_CLASSIFIER/app
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_ROOT)

from preprocess import preprocess_text

app = Flask(__name__)

# Load model & vectorizer from ../models/
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH      = os.path.join(MODELS_DIR, "best_model.pkl")

vectorizer = joblib.load(VECTORIZER_PATH)
model      = joblib.load(MODEL_PATH)

# Map numeric labels → category names
category_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        raw_text = request.form["news_text"]
        # Preprocess
        clean = preprocess_text(raw_text)
        # Vectorize (expects an iterable)
        X = vectorizer.transform([clean])
        # Predict numeric label
        num_pred = model.predict(X)[0]
        # Map to string
        prediction = category_map.get(int(num_pred), "Unknown")
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
