# streamlit_app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from visualizations import (
    plot_anomaly_distribution,
    plot_model_predictions,
    plot_time_series
)

# --- Utility Functions ---

# Text cleaning
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Parse email metadata safely
def parse_email_metadata(message: str):
    headers = {"from": None, "to": None, "cc": None, "bcc": None, "date": None, "subject": None}
    if not isinstance(message, str):
        return headers

    patterns = {
        "from": r"^From:\s*(.*)$",
        "to": r"^To:\s*(.*)$",
        "cc": r"^Cc:\s*(.*)$",
        "bcc": r"^Bcc:\s*(.*)$",
        "date": r"^Date:\s*(.*)$",
        "subject": r"^Subject:\s*(.*)$"
    }

    lines = message.splitlines()
    for key, pattern in patterns.items():
        for line in lines:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                headers[key] = match.group(1).strip()
                break

    # Parse date
    if headers["date"]:
        try:
            headers["date"] = pd.to_datetime(headers["date"], errors="coerce")
        except:
            headers["date"] = None
    return headers

# --- Feature Extraction ---
def extract_features_for_streamlit(df, vectorizer, imputer, feature_cols):
    # Ensure recipient columns exist
    for col in ["to", "cc", "bcc"]:
        if col not in df.columns:
            df[col] = ""

    # Safe counting of recipients
    def safe_count(x):
        if isinstance(x, str) and x.strip():
            return len([addr for addr in x.split(",") if addr.strip()])
        return 0

    df["num_to"] = df["to"].apply(safe_count)
    df["num_cc"] = df["cc"].apply(safe_count)
    df["num_bcc"] = df["bcc"].apply(safe_count)

    df["hour"] = pd.to_datetime(df["date"], errors="coerce").dt.hour
    df["hour"] = df["hour"].fillna(df["hour"].median())
    df["is_off_hours"] = df["hour"].apply(lambda x: x < 6 or x > 20)

    df["char_length"] = df["cleaned_message"].astype(str).str.len()
    df["word_count"] = df["cleaned_message"].astype(str).str.split().str.len()

    # Threat keyword count
    keywords = {"confidential", "internal", "secret", "leak", "hr", "access",
                "credentials", "breach", "login", "download", "report",
                "copy", "exfiltrate", "unauthorized"}
    df["threat_keyword_count"] = df["cleaned_message"].astype(str).apply(
        lambda x: sum(1 for word in x.split() if word.lower() in keywords)
    )

    # Placeholder columns if missing in raw data
    if "unique_recipient_count" not in df.columns:
        df["unique_recipient_count"] = df[["to","cc","bcc"]].apply(
            lambda row: len(set(",".join([str(x) if x else "" for x in row]).split(","))), axis=1
        )
    if "sentiment_polarity" not in df.columns:
        df["sentiment_polarity"] = 0.0  # or compute sentiment if needed

    # Vectorize text
    X_text = vectorizer.transform(df["cleaned_message"].astype(str))
    X_struct = df[["num_to","num_cc","num_bcc","hour","is_off_hours",
                   "char_length","word_count","unique_recipient_count",
                   "sentiment_polarity","threat_keyword_count"]]
    features = pd.concat([X_struct.reset_index(drop=True), pd.DataFrame(X_text.toarray())], axis=1)

    # --- Fix for mixed column types ---
    features.columns = features.columns.astype(str)

    # Impute missing values
    features = pd.DataFrame(imputer.transform(features), columns=features.columns)

    # Ensure all training columns exist
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0
    # Reorder exactly as during training
    features = features[feature_cols]

    return df, features

# --- Streamlit UI ---
st.title("Enron Insider Threat Detection")

uploaded_file = st.file_uploader("Upload your raw Enron email CSV", type=["csv"])
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip')

    # Parse email headers
    parsed = raw_df["message"].apply(parse_email_metadata).apply(pd.Series)
    raw_df = pd.concat([raw_df, parsed], axis=1)

    # Clean text
    raw_df["cleaned_message"] = raw_df["message"].astype(str).apply(clean_text)

    # Load models & preprocessing
    iso_model = pickle.load(open("models/models_2/isolation_forest.pkl", "rb"))
    svm_model = pickle.load(open("models/models_2/ocsvm.pkl", "rb"))
    auto_model = load_model("models/models_2/autoencoder_model.keras")
    vectorizer = pickle.load(open("models/models_2/tfidf_vectorizer.pkl", "rb"))
    imputer = pickle.load(open("models/models_2/simple_imputer.pkl", "rb"))
    feature_cols = pickle.load(open("models/models_2/feature_columns.pkl", "rb"))

    # Extract features
    df_features, features = extract_features_for_streamlit(raw_df, vectorizer, imputer, feature_cols)

    # --- Model inference ---
    iso_pred = iso_model.predict(features)        # -1 = anomaly
    svm_pred = svm_model.predict(features)        # -1 = anomaly
    features_array = features.to_numpy()
    reconstruction = auto_model.predict(features_array)
    mse = np.mean((features_array - reconstruction) ** 2, axis=1)
    threshold = np.percentile(mse, 95)
    auto_pred = (mse > threshold).astype(int)

    # Majority vote ensemble
    final_pred = []
    for i in range(len(features)):
        votes = [iso_pred[i] == -1, svm_pred[i] == -1, auto_pred[i] == 1]
        final_pred.append(sum(votes) >= 2)
    df_features["final_anomaly"] = final_pred

    st.success("✅ Analysis complete!")

    # Show top anomalies
    st.subheader("Top anomalies")
    st.dataframe(df_features[df_features["final_anomaly"]].head(10))

    # 🔍 Visualizations
    plot_anomaly_distribution(df_features)
    plot_model_predictions(df_features)
    plot_time_series(df_features)


    # Download CSV
    csv = df_features.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name='enron_anomaly_predictions.csv',
        mime='text/csv'
    )
