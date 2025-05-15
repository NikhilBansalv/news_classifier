import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

import pandas as pd

# Load the CSV assuming the first row is the header
df = pd.read_csv("data/train.csv")

# Check if column names are what we expect
print(df.columns)

df = df.rename(columns={
    'Class Index': 'ClassIndex',
    'Title': 'Title',
    'Description': 'Description'
})

# Convert ClassIndex to int
df['ClassIndex'] = df['ClassIndex'].astype(int)

# Map class index to category name
category_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}
df['Category'] = df['ClassIndex'].map(category_map)

# Preview
print(df.head())
print(df['Category'].value_counts())


plt.figure(figsize=(8, 4))
sns.countplot(x="Category", data=df, palette="viridis")
plt.title("Class Distribution")
plt.xlabel("News Category")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

df["Text"] = df["Title"] + " " + df["Description"]

df["Word Count"] = df["Text"].apply(lambda x: len(tokenizer.tokenize(x)))
df["Char Count"] = df["Text"].apply(len)

print("\nAverage Word Count:", df["Word Count"].mean())
print("Average Character Count:", df["Char Count"].mean())

plt.figure(figsize=(8, 4))
sns.histplot(df["Word Count"], bins=30, kde=True)
plt.title("Distribution of Word Counts")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

categories = df["Category"].unique()
n_categories = len(categories)

# Set number of rows and columns for subplots
cols = 2
rows = (n_categories + 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
axes = axes.flatten()  # Make axes iterable

for i, label in enumerate(categories):
    text = " ".join(df[df["Category"] == label]["Text"].values)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    
    axes[i].imshow(wc, interpolation="bilinear")
    axes[i].axis("off")
    axes[i].set_title(f"WordCloud for {label}", fontsize=14)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
