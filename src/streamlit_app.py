# src/streamlit_app.py
"""
Streamlit UI that:
- Loads trained anomaly models (IsolationForest, OneClassSVM, Autoencoder),
  TF-IDF vectorizer and SimpleImputer / scaler / feature_columns.
- Accepts two CSV uploads:
    1) Email dataset (system_logs.csv) -> NLP/anomaly inference
    2) UBA dataset (enron_recleaned.csv) -> user behavior analytics
- Produces message-level anomaly scores + user-level final combined risk score.
- Allows download of merged results and displays simple visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import pickle
from tensorflow.keras.models import load_model
from typing import List
# ---  Federated Privacy Layer (Simulation) ---
from federated.coordinator import federated_aggregate

# -------------------------
# Config / constants
# -------------------------
MODEL_DIR = os.path.join("models", "models_2")
# Model filenames (edit if you used other names)
AUTOENCODER_FN = "autoencoder_model.keras"
OCSVM_FN = "ocsvm.pkl"
ISO_FN = "isolation_forest.pkl"
VECTORIZER_FN = "tfidf_vectorizer.pkl"
IMPUTER_FN = "simple_imputer.pkl"
FEATURE_COLS_FN = "feature_columns.pkl"
SCALER_FN = "scaler.pkl"

# Weights for ensemble
AE_WEIGHT = 0.4
SVM_WEIGHT = 0.3
ISO_WEIGHT = 0.3
# Final merge weight (UBA vs NLP)
NLP_WEIGHT = 0.7
UBA_WEIGHT = 0.3

# -------------------------
# Utility functions
# -------------------------
def clean_text(text: str) -> str:
    """Simple cleaning used by app (safe for already-cleaned text too)."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def parse_email_metadata(message: str):
    """Extract simple headers when present in the message text."""
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
    for line in message.splitlines():
        for key, pattern in patterns.items():
            match = re.match(pattern, line, flags=re.IGNORECASE)
            if match and not headers[key]:
                headers[key] = match.group(1).strip()
    # parse date to pd.Timestamp (coerced)
    if headers["date"]:
        try:
            headers["date"] = pd.to_datetime(headers["date"], errors="coerce", utc=True)
        except:
            headers["date"] = None
    return headers

def safe_count_recipients(x):
    """Count comma-separated recipients robustly."""
    if isinstance(x, str) and x.strip():
        return len([addr for addr in x.split(",") if addr.strip()])
    return 0

def normalize_user_key(email: str):
    """Map email -> normalized user key to join UBA and NLP.
       e.g. phillip.allen@enron.com -> phillip-allen
       Handles cases like 'allen-p' by leaving as-is if no '@' present.
    """
    if pd.isna(email):
        return None
    s = str(email).strip()
    if "@" in s:
        name = s.split("@", 1)[0].lower()
    else:
        name = s.lower()
    name = name.replace(".", "-").replace("_", "-")
    # remove stray characters
    name = re.sub(r"[^a-z0-9\-]", "", name)
    return name

def batch_transform_vectorizer(vectorizer, texts: pd.Series, batch_size: int = 2000):
    """TF-IDF transform in batches to handle modestly large uploads."""
    rows = []
    n = len(texts)
    for i in range(0, n, batch_size):
        chunk = texts.iloc[i:i+batch_size].astype(str)
        Xc = vectorizer.transform(chunk)
        rows.append(Xc.toarray())
    return np.vstack(rows)

def ensure_feature_columns_are_strings(feature_cols):
    """feature_columns.pkl might be a list of ints or strings; convert to strings."""
    return [str(c) for c in feature_cols]

# -------------------------
# Load models (deferred until click to avoid slow startup)
# -------------------------
@st.cache_resource
def load_models_and_artifacts():
    """Load models & preprocessing objects from MODEL_DIR (cached)."""
    # Paths
    auto_path = os.path.join(MODEL_DIR, AUTOENCODER_FN)
    ocsvm_path = os.path.join(MODEL_DIR, OCSVM_FN)
    iso_path = os.path.join(MODEL_DIR, ISO_FN)
    vec_path = os.path.join(MODEL_DIR, VECTORIZER_FN)
    imp_path = os.path.join(MODEL_DIR, IMPUTER_FN)
    feat_path = os.path.join(MODEL_DIR, FEATURE_COLS_FN)
    scaler_path = os.path.join(MODEL_DIR, SCALER_FN)

    # Load / defensive checks
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Vectorizer not found: {vec_path}")
    vectorizer = load(vec_path)

    if not os.path.exists(imp_path):
        raise FileNotFoundError(f"Imputer not found: {imp_path}")
    imputer = load(imp_path)

    if not os.path.exists(feat_path):
        # fallback: build simple numeric + tfidf feature list
        feature_cols = None
    else:
        with open(feat_path, "rb") as f:
            feature_cols = pickle.load(f)
        # convert to strings to match DataFrame columns
        feature_cols = ensure_feature_columns_are_strings(feature_cols)

    # load sklearn models
    if not os.path.exists(ocsvm_path) or not os.path.exists(iso_path):
        raise FileNotFoundError("OCSVM or IsolationForest model missing in models/models_2")
    ocsvm = load(ocsvm_path)
    iso = load(iso_path)

    # load autoencoder (Keras)
    if not os.path.exists(auto_path):
        raise FileNotFoundError(f"Autoencoder model not found: {auto_path}")
    autoencoder = load_model(auto_path)

    # optional scaler
    scaler = load(scaler_path) if os.path.exists(scaler_path) else None

    return {
        "vectorizer": vectorizer,
        "imputer": imputer,
        "feature_cols": feature_cols,
        "ocsvm": ocsvm,
        "iso": iso,
        "autoencoder": autoencoder,
        "scaler": scaler
    }

# -------------------------
# Feature extraction for NLP dataset
# -------------------------
def extract_nlp_features(nlp_df: pd.DataFrame, vectorizer, imputer, feature_cols: List[str]):
    """Given raw NLP dataframe, produce aligned feature DataFrame ready for model inference.

    - nlp_df: must contain text column (cleaned_message or message)
    - returns (nlp_df_with_structures, features_df_aligned)
    """
    df = nlp_df.copy()

    # ensure basic columns exist and create cleaned_message if missing
    if "cleaned_message" not in df.columns:
        df["cleaned_message"] = df["message"].astype(str).apply(clean_text)

    # ensure recipient columns exist
    for c in ["to", "cc", "bcc"]:
        if c not in df.columns:
            df[c] = ""

    # structural features
    df["num_to"] = df["to"].apply(safe_count_recipients)
    df["num_cc"] = df["cc"].apply(safe_count_recipients)
    df["num_bcc"] = df["bcc"].apply(safe_count_recipients)

    # robust parse of date -> hour
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["hour"] = parsed.dt.hour.fillna(12)
    else:
        df["hour"] = 12

    df["is_off_hours"] = df["hour"].apply(lambda x: (x < 6) or (x > 20))
    df["char_length"] = df["cleaned_message"].astype(str).str.len().fillna(0).astype(int)
    df["word_count"] = df["cleaned_message"].astype(str).str.split().str.len().fillna(0).astype(int)

    keywords = {"confidential", "internal", "secret", "leak", "hr", "access",
                "credentials", "breach", "login", "download", "report",
                "copy", "exfiltrate", "unauthorized"}
    df["threat_keyword_count"] = df["cleaned_message"].astype(str).apply(
        lambda x: sum(1 for w in str(x).split() if w.lower() in keywords)
    )

    # unique recipients
    df["unique_recipient_count"] = df[["to", "cc", "bcc"]].apply(
        lambda row: len(set(addr.strip() for part in row for addr in str(part).split(",") if addr and addr.strip())), axis=1
    )

    # placeholder sentiment (you can plug a real sentiment function)
    if "sentiment_polarity" not in df.columns:
        df["sentiment_polarity"] = 0.0

    # TF-IDF transform (batch safe)
    texts = df["cleaned_message"].astype(str)
    try:
        X_text = batch_transform_vectorizer(vectorizer, texts)
    except MemoryError:
        st.warning("TF-IDF transform ran out of memory — try uploading fewer rows or run locally/Colab.")
        raise

    # Build structured DataFrame and tfidf DataFrame
    tfidf_cols = [str(i) for i in range(X_text.shape[1])]
    tfidf_df = pd.DataFrame(X_text, index=df.index, columns=tfidf_cols)

    X_struct = df[[
        "num_to", "num_cc", "num_bcc", "hour", "is_off_hours",
        "char_length", "word_count", "unique_recipient_count",
        "sentiment_polarity", "threat_keyword_count"
    ]].reset_index(drop=True)

    # combine (drop duplicate columns if any)
    features = pd.concat([X_struct.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    features = features.loc[:, ~features.columns.duplicated()].copy()

    # Ensure columns are strings (sklearn imputer in some versions requires string names)
    features.columns = features.columns.astype(str)

    # Impute missing values (imputer expects same columns as during training or accepts new columns)
    features_imputed = pd.DataFrame(imputer.transform(features), columns=features.columns)

    # Align to training feature columns (feature_cols may be None if missing)
    if feature_cols is None:
        # fallback: keep current columns and warn
        st.warning("feature_columns.pkl not found — proceeding with current features (may mismatch model).")
        aligned = features_imputed
    else:
        # Ensure feature_cols is list of strings
        feature_cols_str = [str(c) for c in feature_cols]
        # add missing columns (zero)
        for c in feature_cols_str:
            if c not in features_imputed.columns:
                features_imputed[c] = 0
        # keep only training columns in the same order
        aligned = features_imputed[feature_cols_str]

    return df, aligned

# -------------------------
# Inference helpers
# -------------------------
def run_models_and_ensemble(features: pd.DataFrame, models: dict):
    """Run ocsvm, isolation forest and autoencoder and produce:
       - final_pred boolean per row (majority vote)
       - anomaly score (0..1) computed from normalized components
    """
    ocsvm = models["ocsvm"]
    iso = models["iso"]
    autoencoder = models["autoencoder"]

    # Convert to numpy float32
    X = features.to_numpy().astype(np.float32)

    # Autoencoder expects full combined vector size (it was trained on combined features)
    # recon shape -> (n, n_features)
    recon = autoencoder.predict(X, verbose=0)
    ae_loss = np.mean(np.square(X - recon), axis=1)

    # OCSVM & IsolationForest produce scores (decision_function, score_samples)
    try:
        ocsvm_score = -ocsvm.decision_function(X)   # negative -> anomaly
    except Exception as e:
        st.warning("OCSVM decision_function failed — check input shape. " + str(e))
        ocsvm_score = np.zeros(len(X))

    try:
        isof_score = -iso.score_samples(X)
    except Exception as e:
        st.warning("IsolationForest score_samples failed — check input shape. " + str(e))
        isof_score = np.zeros(len(X))

    # Normalize each to 0..1
    def norm01(a):
        a = np.array(a, dtype=float)
        mn, mx = a.min(), a.max()
        if mx - mn < 1e-9:
            return np.zeros_like(a)
        return (a - mn) / (mx - mn)

    ae_norm = norm01(ae_loss)
    svm_norm = norm01(ocsvm_score)
    iso_norm = norm01(isof_score)

    # composite anomaly score
    anomaly_score = AE_WEIGHT * ae_norm + SVM_WEIGHT * svm_norm + ISO_WEIGHT * iso_norm

    # majority vote (thresholded predictions)
    iso_pred = iso.predict(features)  # -1 anomaly
    try:
        svm_pred = ocsvm.predict(features)  # -1 anomaly
    except Exception:
        # ocsvm sometimes expects array; fallback to decision function sign
        svm_pred = np.where(svm_norm > np.percentile(svm_norm, 90), -1, 1)

    # auto_pred: 1==anomaly for our earlier convention (mse > threshold)
    thr = np.percentile(ae_loss, 95)
    auto_pred = (ae_loss > thr).astype(int)

    final_bool = []
    for i in range(len(features)):
        votes = [iso_pred[i] == -1, (svm_pred[i] == -1), (auto_pred[i] == 1)]
        final_bool.append(sum(votes) >= 2)

    return np.array(final_bool, dtype=bool), anomaly_score

# -------------------------
# Simple plotting helpers (Streamlit-friendly)
# -------------------------
def plot_anomaly_distribution(df, score_col="anomaly_score"):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.histplot(df[score_col].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title("Anomaly score distribution (NLP)")
    ax.set_xlabel("anomaly_score")
    st.pyplot(fig)

def plot_top_users(merged_df, k=10):
    # Try to use combined_risk if present, else federated_risk
    risk_col = "combined_risk"
    if "federated_risk" in merged_df.columns:
        risk_col = "federated_risk"
    elif "combined_risk" not in merged_df.columns:
        st.warning("No risk column found in DataFrame — skipping plot.")
        return

    top = merged_df.sort_values(risk_col, ascending=False).head(k)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=risk_col, y="user_key" if "user_key" in merged_df.columns else "user", data=top, ax=ax)
    ax.set_title(f"Top {risk_col.replace('_', ' ').title()} Users")
    st.pyplot(fig)


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Enron Insider Threat — NLP + UBA", layout="wide")
st.title("Enron Insider Threat Detection — NLP + UBA (Unified)")

st.markdown("""
Upload:
- **Email CSV** (system_logs / NLP dataset) — columns: `file, message, from, to, cc, bcc, date, subject, cleaned_message`  
- **UBA CSV** (enron_recleaned / UBA dataset) — columns: `timestamp, user, event_type, resource, action_status, device, location, severity_score`
""")

with st.sidebar:
    st.header("Models / Files")
    st.write(f"Model dir: `{MODEL_DIR}`")
    if st.button("Load models & artifacts"):
        st.info("Loading models... (cached)")
        try:
            models = load_models_and_artifacts()
            st.success("Models loaded ✓")
        except Exception as e:
            st.error(f"Failed to load models: {e}")

uploaded_email = st.file_uploader("Upload email CSV (TestEmails)", type=["csv"], key="email")
uploaded_uba = st.file_uploader("Upload UBA CSV (system_logs)", type=["csv"], key="uba")

if uploaded_email and uploaded_uba:
    # Load models now (cached)
    try:
        artifacts = load_models_and_artifacts()
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

    vectorizer = artifacts["vectorizer"]
    imputer = artifacts["imputer"]
    feature_cols = artifacts["feature_cols"]
    ocsvm = artifacts["ocsvm"]
    iso = artifacts["iso"]
    autoencoder = artifacts["autoencoder"]
    scaler = artifacts["scaler"]

    # read csvs (safe)
    try:
        email_df = pd.read_csv(uploaded_email, engine="python", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Failed to read uploaded email CSV: {e}")
        st.stop()

    try:
        uba_df = pd.read_csv(uploaded_uba, engine="python", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Failed to read uploaded UBA CSV: {e}")
        st.stop()

    st.success(f"Email rows: {len(email_df)} — UBA rows: {len(uba_df)}")

    # Parse headers into columns (if message contains headers)
    if "message" in email_df.columns:
        parsed = email_df["message"].apply(parse_email_metadata).apply(pd.Series)
        # only add columns that aren't present yet to avoid duplicate-label issues
        for c in parsed.columns:
            if c not in email_df.columns:
                email_df[c] = parsed[c]

    # Ensure cleaned_message exists
    if "cleaned_message" not in email_df.columns:
        email_df["cleaned_message"] = email_df["message"].astype(str).apply(clean_text)

    st.info("Extracting NLP features... (this may take a moment for larger files)")
    try:
        nlp_df, features = extract_nlp_features(email_df, vectorizer, imputer, feature_cols)
    except Exception as e:
        st.error("Feature extraction failed: " + str(e))
        st.stop()

    st.write("✅ Feature extraction done. Feature shape:", features.shape)

    # Run models + ensemble (ensure features columns line up to what models expect)
    # Sometimes the models were trained with a scaler; if scaler available, apply it
    if scaler is not None:
        try:
            features_array = scaler.transform(features)
            features = pd.DataFrame(features_array, columns=features.columns)
        except Exception:
            # if scaler complains about names, attempt transform on values
            try:
                features_array = scaler.transform(features.values)
                features = pd.DataFrame(features_array, columns=features.columns)
            except Exception:
                st.warning("Scaler transform skipped (incompatible). Proceeding without scaler.")

    st.info("Running model inference (OCSVM, IsolationForest, Autoencoder)...")
    try:
        final_bool, anomaly_scores = run_models_and_ensemble(features, {
            "ocsvm": ocsvm, "iso": iso, "autoencoder": autoencoder
        })
    except Exception as e:
        st.error("Model inference failed: " + str(e))
        st.stop()

    nlp_df["final_anomaly"] = final_bool
    nlp_df["anomaly_score"] = anomaly_scores

    st.success("✅ NLP inference complete")

    # -------------------------
    # UBA processing (simple risk scoring example)
    # -------------------------
    st.info("Computing UBA risk scores (simple heuristic).")
    uba = uba_df.copy()
    # normalize column names if necessary
    if "user" not in uba.columns:
        st.error("UBA CSV must contain 'user' column (the username or email).")
        st.stop()

    # Example risk computation: combine severity_score and event counts into risk
    # This is intentionally simple and should be replaced with your domain logic.
    uba_agg = uba.groupby("user").agg(
        failed_logins=("event_type", lambda s: (s.str.contains("failed", case=False, na=False)).sum()),
        total_events=("event_type", "count"),
        avg_severity=("severity_score", "mean")
    ).reset_index()
    # normalize and combine
    uba_agg["failed_logins_norm"] = uba_agg["failed_logins"] / (uba_agg["failed_logins"].max() + 1e-9)
    uba_agg["total_events_norm"] = uba_agg["total_events"] / (uba_agg["total_events"].max() + 1e-9)
    uba_agg["avg_severity_norm"] = (uba_agg["avg_severity"] - uba_agg["avg_severity"].min()) / (
        uba_agg["avg_severity"].max() - uba_agg["avg_severity"].min() + 1e-9
    )
    # simple weighted risk
    uba_agg["final_risk"] = 0.4 * uba_agg["failed_logins_norm"] + 0.35 * uba_agg["total_events_norm"] + 0.25 * uba_agg["avg_severity_norm"]

    # normalize final_risk
    uba_agg["final_risk"] = (uba_agg["final_risk"] - uba_agg["final_risk"].min()) / (
        uba_agg["final_risk"].max() - uba_agg["final_risk"].min() + 1e-9
    )

    st.success(f"✅ UBA aggregation done. Users: {len(uba_agg)}")

    # -------------------------
    # Normalize user keys & merge
    # -------------------------
    nlp_df["user_key"] = nlp_df["from"].apply(normalize_user_key)
    uba_agg["user_key"] = uba_agg["user"].apply(normalize_user_key)

    # aggregate anomaly per user
    user_anom = nlp_df.groupby("user_key")["anomaly_score"].mean().reset_index().rename(columns={"anomaly_score": "avg_anomaly_score"})

    merged = pd.merge(uba_agg, user_anom, on="user_key", how="left")
    merged["avg_anomaly_score"] = merged["avg_anomaly_score"].fillna(0)

    # combined risk
    merged["combined_risk"] = NLP_WEIGHT * merged["avg_anomaly_score"] + UBA_WEIGHT * merged["final_risk"]
    merged["combined_risk"] = (merged["combined_risk"] - merged["combined_risk"].min()) / (merged["combined_risk"].max() - merged["combined_risk"].min() + 1e-9)

    # join message-level final results for download/display
    final_df = pd.merge(
        nlp_df,
        merged[["user_key", "final_risk", "combined_risk"]],
        on="user_key",
        how="left"
    )
    # -------------------------
    # Federated Privacy call (in-app version)
    # -------------------------
    merged_fed = None
    try:
        st.markdown("### 🛡️ Running Federated Privacy Aggregation")

        # Pass the already processed dataframes
        merged_fed = federated_aggregate(
            emails_df=final_df,     # your NLP results dataframe
            system_logs_df=uba_df   # your UBA/system logs dataframe
        )

        if merged_fed is not None and not merged_fed.empty:
            st.success(f"✅ Federated aggregation completed for {len(merged_fed)} users.")
            st.dataframe(
                merged_fed.sort_values("federated_risk", ascending=False).head(15)
            )
            st.markdown("**Top Federated Risky Users**")
            st.bar_chart(
                merged_fed.sort_values("federated_risk", ascending=False).set_index("user")["federated_risk"].head(10)
            )
        else:
            st.warning("Federated aggregation returned no matching users. Check user IDs or keys.")

    except Exception as e:
        st.error(f"Federated aggregation failed: {e}")

    # -------------------------
    # UI Tabs and outputs
    # -------------------------
    tab1, tab2, tab3, tab4  = st.tabs(["NLP Anomalies", "UBA Overview", "Combined Risk","Federated Privacy"])

    with tab1:
        st.subheader("Top NLP anomalies (message-level)")
        st.write("Showing top messages flagged by ensemble majority + anomaly score")
        st.dataframe(final_df.sort_values("anomaly_score", ascending=False)[["file", "from", "subject", "anomaly_score", "final_anomaly"]].head(50))

        st.markdown("**Anomaly score distribution**")
        plot_anomaly_distribution(final_df, score_col="anomaly_score")

    with tab2:
        st.subheader("UBA aggregated (user-level)")
        st.dataframe(uba_agg.sort_values("final_risk", ascending=False).head(50))
        st.markdown("###Top UBA risky users")
        st.dataframe(
            uba_agg.sort_values("final_risk", ascending=False)
            [["user", "avg_severity_norm", "final_risk"]]
            .head(10)
            .style.background_gradient(subset=["final_risk"], cmap="Reds")
        )
        
    with tab3:
        st.subheader("Combined risk (NLP + UBA)")
        st.write("Top combined-risk users")
        st.dataframe(merged.sort_values("combined_risk", ascending=False).head(50))
        st.markdown("Top risky users (bar)")
        plot_top_users(merged, k=15)

    with tab4:
        st.subheader("Federated Privacy Aggregation")
        if merged_fed is not None:
            st.dataframe(merged_fed.sort_values("federated_risk", ascending=False).head(30))
            st.markdown("**Combined Federated Risk (NLP + UBA)**")
            plot_top_users(merged_fed, k=15)
        else:
            st.warning("Federated results unavailable.")
    
    # -------------------------
    # Download / Save
    # -------------------------
    csv_bytes = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download final merged CSV", data=csv_bytes, file_name="nlp_uba_combined.csv", mime="text/csv")

    # also show simple counts
    st.write("Matched users between NLP and UBA:", merged["user_key"].nunique(), "of", len(merged))
