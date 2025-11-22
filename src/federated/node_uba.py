import pandas as pd
import numpy as np

def run_local_uba_model(system_logs_df):
    """
    Simulated local UBA node using in-memory behavior logs.
    """
    if "user" not in system_logs_df.columns:
        raise KeyError("Expected 'user' column in system logs")

    df = system_logs_df.copy()
    df["user_key"] = df["user"].str.lower().str.replace(".", "-").str.replace("_", "-")

    # Normalize severity if needed
    if "severity_score" not in df.columns:
        df["severity_score"] = np.random.rand(len(df))

    df["local_score_uba"] = df["severity_score"] / df["severity_score"].max()

    # Aggregate per user
    user_scores = (
        df.groupby("user_key")["local_score_uba"]
        .mean()
        .reset_index()
    )

    return user_scores
