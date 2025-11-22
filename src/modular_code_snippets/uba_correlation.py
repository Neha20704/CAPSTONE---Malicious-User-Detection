# src/uba_correlation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def compute_uba_metrics(log_path: str):
    """Compute user behavior metrics from system logs."""
    logs = pd.read_csv(log_path, parse_dates=['timestamp'])
    
    # Aggregate by user
    uba_features = logs.groupby("user").agg(
        total_events=("event_type", "count"),
        failed_logins=("event_type", lambda x: (x == "login_failure").sum()),
        data_exfiltrations=("event_type", lambda x: (x == "data_exfiltration").sum()),
        file_deletions=("event_type", lambda x: (x == "file_delete").sum()),
        unique_resources=("resource", "nunique"),
        mean_severity=("severity_score", "mean"),
        high_severity_actions=("severity_score", lambda x: (x > 0.8).sum())
    ).reset_index()

    # Derived features
    uba_features["fail_to_total_ratio"] = (
        uba_features["failed_logins"] / uba_features["total_events"]
    ).fillna(0)
    uba_features["risk_behavior_score"] = (
        0.2 * uba_features["fail_to_total_ratio"]
        + 0.3 * uba_features["data_exfiltrations"]
        + 0.3 * uba_features["file_deletions"]
        + 0.2 * uba_features["mean_severity"]
    )

    # Normalize risk score
    scaler = MinMaxScaler()
    uba_features["normalized_risk_score"] = scaler.fit_transform(
        uba_features[["risk_behavior_score"]]
    )

    return uba_features


def merge_with_communication(uba_df, comm_df, comm_pred_path=None):
    """Merge UBA features with communication-level risk data."""
    comm_df = comm_df.copy()
    
    # Optional: if you have predictions saved (e.g., enron_anomaly_predictions.csv)
    if comm_pred_path:
        preds = pd.read_csv(comm_pred_path)
        comm_df = comm_df.merge(preds, on="file", how="left")

    # Aggregate communication data per user
    comm_features = comm_df.groupby("from").agg(
        avg_msg_length=("cleaned_message", lambda x: np.mean(x.str.len())),
        email_count=("cleaned_message", "count"),
        # if you have model predictions
        mean_predicted_risk=("prediction", "mean") if "prediction" in comm_df.columns else 0
    ).reset_index().rename(columns={"from": "user"})

    # Merge
    combined = pd.merge(uba_df, comm_features, on="user", how="outer").fillna(0)

    # Compute overall combined score
    combined["final_risk_score"] = (
        0.6 * combined["normalized_risk_score"] +
        0.4 * combined["mean_predicted_risk"]
    )

    return combined


if __name__ == "__main__":
    # Paths
    log_path = "data/system_logs.csv"
    comm_path = "data/enron_recleaned.csv"
    pred_path = "data/enron_anomaly_predictions.csv"  # optional if exists

    # Load data
    uba_df = compute_uba_metrics(log_path)
    comm_df = pd.read_csv(comm_path)
    comm_df.columns = comm_df.columns.str.strip()

    # Merge & correlate
    combined_df = merge_with_communication(uba_df, comm_df, pred_path)

    combined_df.to_csv("outputs/user_risk_profiles.csv", index=False)
    print("✅ Correlation complete. Saved to outputs/user_risk_profiles.csv")
