import pandas as pd
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def extract_features(df: pd.DataFrame, fit_vectorizer=True, vectorizer=None):
    # Recipient counts
    df["num_to"] = df["to"].fillna("").apply(lambda x: len(x.split(",")))
    df["num_cc"] = df["cc"].fillna("").apply(lambda x: len(x.split(",")))
    df["num_bcc"] = df["bcc"].fillna("").apply(lambda x: len(x.split(",")))

    # Time features
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = df["date"].dt.hour
    median_hour = df["hour"].median()
    df["hour"] = df["hour"].fillna(median_hour)
    df["is_off_hours"] = df["hour"].apply(lambda x: x < 6 or x > 20)

    # Message length
    df["char_length"] = df["cleaned_message"].astype(str).str.len()
    df["word_count"] = df["cleaned_message"].astype(str).str.split().str.len()

    # Unique recipients per sender
    sender_recipient_map = df.groupby("from")["to"].apply(
        lambda x: set(",".join(x.dropna()).split(","))
    )
    df["unique_recipient_count"] = df["from"].map(
        lambda s: len(sender_recipient_map.get(s, []))
    )

    # Sentiment
    df["sentiment_polarity"] = df["cleaned_message"].astype(str).apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    # Threat keywords
    keywords = {"confidential", "internal", "secret", "leak", "hr", "access",
                "credentials", "breach", "login", "download", "report",
                "copy", "exfiltrate", "unauthorized"}
    df["threat_keyword_count"] = df["cleaned_message"].astype(str).apply(
        lambda x: sum(1 for word in x.split() if word in keywords)
    )

    # Vectorization
    if fit_vectorizer:
        vectorizer = TfidfVectorizer(max_features=1000)
        X_text = vectorizer.fit_transform(df["cleaned_message"].astype(str))
    else:
        X_text = vectorizer.transform(df["cleaned_message"].astype(str))

    X_structured = df[[
        "num_to","num_cc","num_bcc","hour","is_off_hours",
        "char_length","word_count","sentiment_polarity","threat_keyword_count"
    ]]

    features = pd.concat([X_structured.reset_index(drop=True),
                          pd.DataFrame(X_text.toarray())], axis=1)

    imputer = SimpleImputer(strategy="mean")
    features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

    return df, features, vectorizer, imputer
