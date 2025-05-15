import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Paths relative to this script
HERE        = os.path.dirname(__file__)
MODELS_DIR  = os.path.join(HERE, "..", "models")

# 1️⃣ Load TF‑IDF features & labels saved in models/
X = joblib.load(os.path.join(MODELS_DIR, "X_features.pkl"))
y = joblib.load(os.path.join(MODELS_DIR, "y_labels.pkl"))

# 2️⃣ Split into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Define models (no RandomForest)
models = {
    "naivebayes": MultinomialNB(),
    "logisticregression": LogisticRegression(max_iter=1000),
    "linearsvc": LinearSVC()
}

# 4️⃣ Ensure models/ directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

best_model      = None
best_score      = 0.0
best_model_name = ""

# 5️⃣ Train, evaluate, and save each model
for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, preds))

    # Save this model artifact under models/
    model_path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    joblib.dump(model, model_path)

    # Track best
    if acc > best_score:
        best_score      = acc
        best_model      = model
        best_model_name = name

# 6️⃣ Save the best model
best_path = os.path.join(MODELS_DIR, "best_model.pkl")
joblib.dump(best_model, best_path)
print(f"\n✅ Best model: {best_model_name} (Accuracy: {best_score:.4f})")
