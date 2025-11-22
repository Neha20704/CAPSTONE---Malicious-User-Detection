# dashboard.py
import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# -----------------------------
# Load Models + Imputer
# -----------------------------
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_insider.pkl"))
iso_model = joblib.load(os.path.join(MODEL_DIR, "isolation_forest_insider.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
imputer = joblib.load(os.path.join(MODEL_DIR, "simple_imputer.pkl"))

# For TF-IDF (must match training settings)
tfidf = TfidfVectorizer(max_features=1000)
# In practice: load fitted vectorizer
# but if not saved, you need to refit on your dataset before running dashboard
# joblib.dump(tfidf, "tfidf_vectorizer.pkl")
# tfidf = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# Helper functions
# -----------------------------
def extract_features(df):
    """Recompute engineered features from raw email data for dashboard inference."""
    # Basic counts
    df["num_to"] = df["to"].fillna("").apply(lambda x: len(x.split(",")))
    df["num_cc"] = df["cc"].fillna("").apply(lambda x: len(x.split(",")))
    df["num_bcc"] = df["bcc"].fillna("").apply(lambda x: len(x.split(",")))

    # Time features
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = df["date"].dt.hour
    median_hour = df["hour"].median()
    df["hour"] = df["hour"].fillna(median_hour)
    df["is_off_hours"] = df["hour"].apply(lambda x: x < 6 or x > 20)

    # Length
    df["char_length"] = df["cleaned_message"].str.len()
    df["word_count"] = df["cleaned_message"].str.split().str.len()

    # Sentiment
    df["sentiment_polarity"] = df["cleaned_message"].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Keyword tagging
    keywords = [
        "confidential", "internal", "secret", "leak", "hr", "access", "credentials",
        "breach", "login", "download", "report", "copy", "exfiltrate", "unauthorized"
    ]
    df["threat_keyword_count"] = df["cleaned_message"].apply(
        lambda x: sum(1 for word in x.split() if word in keywords)
    )

    # TF-IDF
    tfidf_matrix = tfidf.transform(df["cleaned_message"].fillna(""))

    # Merge structured + tfidf
    features = pd.concat([
        df[["num_to", "num_cc", "num_bcc", "hour", "is_off_hours",
            "char_length", "word_count", "sentiment_polarity", "threat_keyword_count"]],
        pd.DataFrame(tfidf_matrix.toarray())
    ], axis=1)

    # Impute
    features = pd.DataFrame(imputer.transform(features), columns=features.columns)
    return features, df


# -----------------------------
# Streamlit Dashboard
# -----------------------------
st.set_page_config(page_title="Insider Threat Dashboard", layout="wide")

st.title("📧 Insider Threat Detection Dashboard")

uploaded_file = st.file_uploader("Upload preprocessed email CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} emails")

    # Extract features
    features, df = extract_features(df)

    # Predictions
    df["anomaly_score"] = if_model.predict(features)
    df["rf_label"] = rf_model.predict(features)

    # Display
    st.subheader("📊 Dataset Overview")
    st.dataframe(df[["from", "to", "subject", "date", "rf_label", "anomaly_score"]].head(20))

    # Charts
    st.subheader("📈 Metrics Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Model Predictions (Random Forest)")
        fig, ax = plt.subplots()
        df["rf_label"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Anomaly Detection (Isolation Forest)")
        fig, ax = plt.subplots()
        df["anomaly_score"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # Search suspicious emails
    st.subheader("🔍 Investigate Suspicious Emails")
    suspicious = df[(df["rf_label"] == 1) | (df["anomaly_score"] == -1)]
    st.dataframe(suspicious[["from", "to", "subject", "date", "cleaned_message"]].head(50))

    st.download_button(
        label="Download flagged emails",
        data=suspicious.to_csv(index=False),
        file_name="suspicious_emails.csv",
        mime="text/csv",
    )

else:
    st.info("👆 Upload a CSV to get started")
