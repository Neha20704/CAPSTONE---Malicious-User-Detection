import re
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

# -----------------------------
# Utility: Text cleaning
# -----------------------------
def clean_text(text):
    """
    Basic text cleaning for email content or logs:
    - Lowercase
    - Remove special characters, URLs, numbers
    - Strip whitespace
    """
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(df, text_column="message", feature_columns_path="../models/models_2/feature_columns.pkl",
                     vectorizer_path="../models/models_2/tfidf_vectorizer.pkl",
                     imputer_path="../models/models_2/simple_imputer.pkl"):
    """
    Extracts & transforms input data into model-ready features.
    - Cleans text
    - Applies TF-IDF vectorizer
    - Aligns metadata columns
    - Imputes missing values
    Returns: Numpy array of features + transformed DataFrame
    """

    # Load fitted preprocessing objects
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(imputer_path, "rb") as f:
        imputer = pickle.load(f)
    with open(feature_columns_path, "rb") as f:
        expected_columns = pickle.load(f)

    # Clean text column
    if text_column not in df.columns:
        raise ValueError(f"'{text_column}' column not found in input DataFrame")

    df["clean_text"] = df[text_column].astype(str).apply(clean_text)

    # TF-IDF transformation
    tfidf_matrix = vectorizer.transform(df["clean_text"])

    # Convert TF-IDF to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
        index=df.index
    )

    # Combine metadata + tfidf
    metadata_cols = [col for col in df.columns if col not in [text_column, "clean_text"]]
    combined_df = pd.concat([df[metadata_cols], tfidf_df], axis=1)

    # Ensure columns match training
    for col in expected_columns:
        if col not in combined_df.columns:
            combined_df[col] = 0  # Add missing columns with 0

    # Reorder to match training
    combined_df = combined_df[expected_columns]

    # Impute missing values
    features = imputer.transform(combined_df)

    return features, combined_df
