import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from preprocess import preprocess_text

# Paths relative to this script
HERE       = os.path.dirname(__file__)
DATA_DIR   = os.path.join(HERE, "..", "data")
MODELS_DIR = os.path.join(HERE, "..", "models")

# 1️⃣ Load & preprocess test set
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
df_test = df_test.rename(columns={
    "Class Index": "ClassIndex",
    "Title":       "Title",
    "Description": "Description"
})
# Map numeric labels → category strings
category_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
df_test["Category"] = df_test["ClassIndex"].map(category_map)

# Combine and clean
df_test["Text"]         = df_test["Title"].astype(str) + " " + df_test["Description"].astype(str)
df_test["Cleaned_Text"] = df_test["Text"].apply(preprocess_text)

# 2️⃣ Vectorize with saved TF‑IDF
tfidf   = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
X_test  = tfidf.transform(df_test["Cleaned_Text"])
y_test  = df_test["Category"]

# 3️⃣ Load each model and collect confusion matrices
model_files = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith("_model.pkl"))
cms, names = [], []

for fname in model_files:
    name  = fname.replace("_model.pkl", "")
    model = joblib.load(os.path.join(MODELS_DIR, fname))

    # Numeric predictions → map back to strings
    y_pred_num = model.predict(X_test)
    y_pred     = [category_map[int(n)] for n in y_pred_num]

    # Metrics
    print(f"\n=== {name} ===")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
    print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))
    print("\n" + classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(category_map.values()))
    cms.append(cm)
    names.append(name)

# 4️⃣ Plot all confusion matrices in a 2×2 grid
labels = list(category_map.values())
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, cm, title in zip(axes, cms, names):
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax
    )
    ax.set_title(f"{title} CM")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# Hide any unused subplots
for i in range(len(cms), 4):
    axes[i].axis("off")

plt.tight_layout()
plt.show()
