import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
import string

# Download necessary resources (you can remove these lines after first run)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize tokenizer and stopwords
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))

# Define your preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = tokenizer.tokenize(text)
    # Remove punctuation and stopwords
    cleaned_tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    # Join back to string
    return ' '.join(cleaned_tokens)

# Main execution block (only runs when this file is executed directly)
if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")) # Make sure train.csv exists in the same folder
    df["Cleaned_Text"] = df["Description"].apply(preprocess_text)
    
    # Optional: Save cleaned data for modeling
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    df.to_csv(os.path.join(DATA_DIR, "train_cleaned.csv"), index=False)
    print("✅ Preprocessing complete. Cleaned data saved to data/train_cleaned.csv.")