import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
HERE       = os.path.dirname(__file__)
DATA_DIR   = os.path.join(HERE, "..", "data")
MODELS_DIR = os.path.join(HERE, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1) Load cleaned data
df = pd.read_csv(os.path.join(DATA_DIR, "train_cleaned.csv"))

# 2) TF‑IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["Cleaned_Text"])
y = df["Class Index"]  # this is your numeric label column

# 3) Save artifacts
with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(MODELS_DIR, "X_features.pkl"), "wb") as f:
    pickle.dump(X, f)

with open(os.path.join(MODELS_DIR, "y_labels.pkl"), "wb") as f:
    pickle.dump(y, f)

print("✅ Saved TF‑IDF vectorizer, X_features.pkl, and y_labels.pkl in the models/ folder.")
