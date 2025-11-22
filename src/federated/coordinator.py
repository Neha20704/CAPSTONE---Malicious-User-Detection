# federated/coordinator.py
import pandas as pd

def federated_aggregate(emails_df, system_logs_df):
    """Combine NLP and UBA insights without exposing raw data."""

    if emails_df is None or system_logs_df is None:
        print("❌ Missing dataframes for federated aggregation")
        return None

    # Derive per-user NLP summary
    if "from" in emails_df.columns:
        nlp_summary = (
            emails_df.groupby("from")["anomaly_score"]
            .mean()
            .reset_index()
            .rename(columns={"from": "user", "anomaly_score": "nlp_risk_score"})
        )
    elif "user_key" in emails_df.columns:
        nlp_summary = (
            emails_df.groupby("user_key")["anomaly_score"]
            .mean()
            .reset_index()
            .rename(columns={"user_key": "user", "anomaly_score": "nlp_risk_score"})
        )
    else:
        raise KeyError("Expected a 'from' or 'user_key' column in NLP dataframe")

    # Derive per-user UBA summary
    uba_summary = (
        system_logs_df.groupby("user")["severity_score"]
        .mean()
        .reset_index()
        .rename(columns={"severity_score": "uba_risk_score"})
    )

    # Merge summaries
    merged = pd.merge(nlp_summary, uba_summary, on="user", how="inner")

    # Weighted aggregation (tune these weights later if needed)
    merged["federated_risk"] = (
        0.6 * merged["nlp_risk_score"] + 0.4 * merged["uba_risk_score"]
    )

    return merged

