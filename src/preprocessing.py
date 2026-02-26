# src/preprocessing.py

import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words("english"))

# -----------------------------
# Text Cleaning / Tokenization
# -----------------------------
def custom_tokenizer(text):
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    return [t for t in tokens if t.isalpha() and t not in stop_words]

# -----------------------------
# Load Dataset
# -----------------------------
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# -----------------------------
# Train-Test Split
# -----------------------------
def split_data(df, text_column="job_description", target_column="fraudulent"):
    X = df[text_column]
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
def vectorize_text(X_train, X_test, max_features=5000):
    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        max_features=max_features
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer