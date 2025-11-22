import pandas as pd
import numpy as np

def run_local_nlp_model(cleaned_emails_df):
    """
    Simulated local NLP node.
    Receives structured output (emails already analyzed in Streamlit),
    not raw unstructured text.
    """
    if "file" not in cleaned_emails_df.columns:
        raise KeyError("Expected 'file' column in NLP data")

    # In your pipeline, after cleaning + NLP scoring,
    # you likely already compute something like 'anomaly_score'
    # Let's handle both cases:
    if "anomaly_score" not in cleaned_emails_df.columns:
        # fallback: random score (for demo)
        cleaned_emails_df["anomaly_score"] = np.random.rand(len(cleaned_emails_df))

    # Simulate per-user grouping by file (or inferred sender)
    cleaned_emails_df["user_key"] = cleaned_emails_df["file"].apply(
        lambda x: str(x).split("_")[0].lower() if pd.notna(x) else "unknown"
    )

    # Aggregate per user
    user_scores = (
        cleaned_emails_df.groupby("user_key")["anomaly_score"]
        .mean()
        .reset_index()
        .rename(columns={"anomaly_score": "local_score_nlp"})
    )

    return user_scores
