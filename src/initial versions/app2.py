# app.py — Combined NLP + UBA Risk Streamlit App (Fixed Import)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from visualizations import (
    plot_anomaly_distribution,
    plot_model_predictions,
    plot_time_series
)

# --- Utility Functions ---

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

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

    for key, pattern in patterns.items():
        for line in message.splitlines():
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                headers[key] = match.group(1).strip()
                break

    # Parse date safely
    if headers["date"]:
        headers["date"] = pd.to_datetime(headers["date"], errors="coerce")
    return headers

def normalize_user(email):
    if pd.isna(email):
        return None
    name = email.split('@')[0].lower()
    return name.replace('.', '-').replace('_', '-')

# 🧩 Feature extraction (inlined version)
def extract_features_for_streamlit(df, vectorizer, imputer, feature_cols):
    for col in ["to", "cc", "bcc"]:
        if col not in df.columns:
            df[col] = ""

    def safe_count(x):
        if isinstance(x, str) and x.strip():
            return len([addr for addr in x.split(",") if addr.strip()])
        return 0

    df["num_to"] = df["to"].apply(safe_count)
    df["num_cc"] = df["cc"].apply(safe_count)
    df["num_bcc"] = df["bcc"].apply(safe_count)

    df["hour"] = pd.to_datetime(df["date"], errors="coerce").dt.hour
    df["hour"] = df["hour"].fillna(12)
    df["is_off_hours"] = df["hour"].apply(lambda x: x < 6 or x > 20)

    df["char_length"] = df["cleaned_message"].astype(str).str.len()
    df["word_count"] = df["cleaned_message"].astype(str).str.split().str.len()

    keywords = {"confidential", "internal", "secret", "leak", "hr", "access",
                "credentials", "breach", "login", "download", "report",
                "copy", "exfiltrate", "unauthorized"}
    df["threat_keyword_count"] = df["cleaned_message"].astype(str).apply(
        lambda x: sum(1 for word in x.split() if word.lower() in keywords)
    )

    if "unique_recipient_count" not in df.columns:
        df["unique_recipient_count"] = df[["to","cc","bcc"]].apply(
            lambda row: len(set(",".join([str(x) if x else "" for x in row]).split(","))), axis=1
        )
    if "sentiment_polarity" not in df.columns:
        df["sentiment_polarity"] = 0.0

    X_text = vectorizer.transform(df["cleaned_message"].astype(str))
    X_struct = df[["num_to","num_cc","num_bcc","hour","is_off_hours",
                   "char_length","word_count","unique_recipient_count",
                   "sentiment_polarity","threat_keyword_count"]]
    features = pd.concat([X_struct.reset_index(drop=True), pd.DataFrame(X_text.toarray())], axis=1)
    features.columns = features.columns.astype(str)

    features = pd.DataFrame(imputer.transform(features), columns=features.columns)

    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0
    features = features[feature_cols]

    return df, features

# --- Streamlit UI ---
st.title("🧠 Enron Insider Threat Detection (NLP + UBA)")

st.sidebar.header("📂 Upload Datasets")
email_file = st.sidebar.file_uploader("Upload Enron Email Dataset (enron_recleaned.csv)", type=["csv"])
uba_file = st.sidebar.file_uploader("Upload System Log Dataset (system_logs.csv)", type=["csv"])

if email_file:
    raw_df = pd.read_csv(email_file, engine='python', on_bad_lines='skip')

    parsed = raw_df["message"].apply(parse_email_metadata).apply(pd.Series)
    raw_df = pd.concat([raw_df, parsed], axis=1)
    raw_df["cleaned_message"] = raw_df["message"].astype(str).apply(clean_text)

    iso_model = pickle.load(open("models/models_2/isolation_forest.pkl", "rb"))
    svm_model = pickle.load(open("models/models_2/ocsvm.pkl", "rb"))
    auto_model = load_model("models/models_2/autoencoder_model.keras")
    vectorizer = pickle.load(open("models/models_2/tfidf_vectorizer.pkl", "rb"))
    imputer = pickle.load(open("models/models_2/simple_imputer.pkl", "rb"))
    feature_cols = pickle.load(open("models/models_2/feature_columns.pkl", "rb"))

    df_features, features = extract_features_for_streamlit(raw_df, vectorizer, imputer, feature_cols)

    iso_pred = iso_model.predict(features)
    svm_pred = svm_model.predict(features)
    features_array = features.to_numpy()
    reconstruction = auto_model.predict(features_array)
    mse = np.mean((features_array - reconstruction) ** 2, axis=1)
    threshold = np.percentile(mse, 95)
    auto_pred = (mse > threshold).astype(int)

    final_pred = []
    for i in range(len(features)):
        votes = [iso_pred[i] == -1, svm_pred[i] == -1, auto_pred[i] == 1]
        final_pred.append(sum(votes) >= 2)

    df_features["final_anomaly"] = final_pred
    df_features["anomaly_score"] = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
    df_features["user_key"] = df_features["from"].apply(normalize_user)

    st.success("✅ NLP anomaly detection complete!")

    if uba_file:
        uba = pd.read_csv(uba_file)
        uba["user_key"] = uba["user"].apply(normalize_user)

        user_anomaly_df = (
            df_features.groupby("user_key")["anomaly_score"]
            .mean()
            .reset_index()
            .rename(columns={"anomaly_score": "avg_anomaly_score"})
        )

        merged_df = pd.merge(uba, user_anomaly_df, on="user_key", how="inner")
        merged_df["final_risk"] = merged_df["severity_score"] / merged_df["severity_score"].max()
        merged_df["combined_risk"] = (
            0.7 * merged_df["avg_anomaly_score"] + 0.3 * merged_df["final_risk"]
        )

        merged_df["combined_risk"] = (
            (merged_df["combined_risk"] - merged_df["combined_risk"].min()) /
            (merged_df["combined_risk"].max() - merged_df["combined_risk"].min())
        )

        final_df = pd.merge(
            df_features[["file", "message", "from", "to", "cc", "bcc", "date", "subject",
                         "anomaly_score", "final_anomaly", "user_key"]],
            merged_df[["user_key", "final_risk", "combined_risk"]],
            on="user_key",
            how="left"
        )

        st.success("✅ NLP + UBA risk fusion complete!")

        st.subheader("🚨 High-Risk Communications")
        top_risky = final_df.sort_values("combined_risk", ascending=False).head(10)
        st.dataframe(top_risky.style.background_gradient(subset=["combined_risk"], cmap="Reds"))

        plot_anomaly_distribution(final_df)
        plot_model_predictions(final_df)
        plot_time_series(final_df)

        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Combined Results (CSV)",
            data=csv,
            file_name='enron_nlp_uba_combined.csv',
            mime='text/csv'
        )

        st.markdown(f"**Matched users:** {merged_df['user_key'].nunique()}")
        st.markdown("### 🔝 Top 10 Users by Risk")
        st.dataframe(
            merged_df.sort_values("combined_risk", ascending=False)
            [["user", "avg_anomaly_score", "final_risk", "combined_risk"]]
            .head(10)
            .style.background_gradient(subset=["combined_risk"], cmap="Reds")
        )

    else:
        st.info("📊 Upload the system log (UBA) CSV in the sidebar to combine behavioral risk data.")
