# News Topic Classifier

A web‑app that classifies news snippets into one of four categories: **World**, **Sports**, **Business**, or **Sci/Tech**—built with Python, scikit‑learn, and Flask.

---

---

## 📝 Project Overview

This project trains a text classifier on the AG News dataset to automatically categorize news headlines and descriptions.  
- **Modeling approach**:  
  1. **Preprocess** raw text (tokenize, remove stopwords)  
  2. **TF‑IDF** feature extraction  
  3. **Baseline models**: Naive Bayes, Logistic Regression, LinearSVC  
  4. **Select best** model by accuracy  
  5. **Deploy** via Flask + Gunicorn

---

## 📊 Dataset Description

- **Source**: AG News (4 classes, 30 000 train + 1 900 test per class)  
- **Columns**:
  - `Class Index` (1–4)
  - `Title` (headline)
  - `Description` (short snippet)  
- **Categories**:
  1. World  
  2. Sports  
  3. Business  
  4. Sci/Tech  

Raw CSVs live in `data/train.csv` and `data/test.csv`.

---

## ⚙️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/NikhilBansalv/news_classifier.git
   cd news_classifier
2. **Create & activate a virtual environment**
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
3. **Install dependencies**
   pip install -r requirements.txt

---

 **Running the Pipelines**
1. Exploratory Data Analysis
    python notebooks/eda_ag_news.py

2. Preprocessing
    python src/preprocess.py
    Reads data/train.csv, outputs data/train_cleaned.csv.

3. Feature Engineering
    python src/feature_engineering.py
    Reads data/train_cleaned.csv.
    Saves models/tfidf_vectorizer.pkl, models/X_features.pkl, models/y_labels.pkl.

4. Model Training
    python src/model_baseline.py
    Trains three classifiers.
    Saves each in models/ plus best_model.pkl.

5. Model Evaluation
    python src/evaluate_models.py
    Loads data/test.csv, preprocesses, vectorizes, and evaluates against each saved model.
    Displays metrics and a 2×2 confusion‑matrix grid.

---

📈 **Model Evaluation Results**

      Model	                  Accuracy
      Naive Bayes	              0.90
      Logistic Regression	      0.92
      LinearSVC                	0.93
      Best Model (LinearSVC)	  0.93

(Your actual scores may vary slightly.)

---

🌐 **Running the Web App**
    Ensure models/ contains best_model.pkl & tfidf_vectorizer.pkl.

    Run Streamlit app.
    cd app
    python -m streamlit run app.py.

    
