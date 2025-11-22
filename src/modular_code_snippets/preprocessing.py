# src/preprocessing.py

import pandas as pd
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer


def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercase, remove digits & punctuation.
    """
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)          # remove numbers
    text = re.sub(r'[^\w\s]', '', text)      # remove punctuation
    return text.strip()


def extract_features(df: pd.DataFrame, fit_vectorizer=True, vectorizer=None, imputer=None):
    """
    Extract structured + text features from email data.

    Args:
        df (pd.DataFrame): input dataframe with cols like from, to, date, cleaned_message
        fit_vectorizer (bool): if True, fit a new TF-IDF vectorizer, else use existing one
        vectorizer (TfidfVectorizer): pre-fitted vectorizer (when fit_vectorizer=False)
        imputer (SimpleImputer): pre-fitted imputer (when fit_vectorizer=False)

    Returns:
        df (pd.DataFrame): dataframe with extra feature columns
        features (pd.DataFrame): numeric feature matrix
        vectorizer (TfidfVectorizer): fitted vectorizer
        imputer (SimpleImputer): fitted imputer
    """
    # --- Communication patterns ---
    df["num_to"] = df["to"].fillna("").apply(lambda x: len(x.split(",")) if x else 0)
    df["num_cc"] = df["cc"].fillna("").apply(lambda x: len(x.split(",")) if x else 0) if "cc" in df.columns else 0
    df["num_bcc"] = df["bcc"].fillna("").apply(lambda x: len(x.split(",")) if x else 0) if "bcc" in df.columns else 0

    # --- Temporal patterns ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = df["date"].dt.hour
    median_hour = df["hour"].median()
    df["hour"] = df["hour"].fillna(median_hour)
    df["is_off_hours"] = df["hour"].apply(lambda x: x < 6 or x > 20)

    # --- Message-level features ---
    df["char_length"] = df["cleaned_message"].astype(str).str.len()
    df["word_count"] = df["cleaned_message"].astype(str).str.split().str.len()

    # Unique recipients per sender
    if "to" in df.columns:
        sender_recipient_map = df.groupby("from")["to"].apply(
            lambda x: set(",".join(x.dropna()).split(","))
        )
        df["unique_recipient_count"] = df["from"].map(
            lambda s: len(sender_recipient_map.get(s, []))
        )
    else:
        df["unique_recipient_count"] = 0

    # --- Sentiment analysis ---
    df["sentiment_polarity"] = df["cleaned_message"].astype(str).apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    # --- Threat keyword features ---
    keywords = {
        "confidential", "internal", "secret", "leak", "hr", "access",
        "credentials", "breach", "login", "download", "report",
        "copy", "exfiltrate", "unauthorized"
    }
    df["threat_keyword_count"] = df["cleaned_message"].astype(str).apply(
        lambda x: sum(1 for word in x.split() if word in keywords)
    )

    # --- Text vectorization ---
    if fit_vectorizer:
        vectorizer = TfidfVectorizer(max_features=1000)
        X_text = vectorizer.fit_transform(df["cleaned_message"].astype(str))
    else:
        X_text = vectorizer.transform(df["cleaned_message"].astype(str))

    # --- Structured features ---
    structured_cols = [
        "num_to", "num_cc", "num_bcc", "hour", "is_off_hours",
        "char_length", "word_count", "unique_recipient_count",
        "sentiment_polarity", "threat_keyword_count"
    ]
    X_structured = df[structured_cols]

    # --- Combine structured + vectorized ---
    features = pd.concat(
        [X_structured.reset_index(drop=True),
         pd.DataFrame(X_text.toarray())],
        axis=1
    )

    # --- Impute missing values ---
    if fit_vectorizer:
        imputer = SimpleImputer(strategy="mean")
        features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
    else:
        features = pd.DataFrame(imputer.transform(features), columns=features.columns)

    return df, features, vectorizer, imputer
