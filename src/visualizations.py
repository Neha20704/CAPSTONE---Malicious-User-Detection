# visualizations.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Distribution of final anomalies ---
def plot_anomaly_distribution(df):
    if df is None or df.empty or "final_anomaly" not in df.columns:
        st.warning("No anomaly predictions available to plot.")
        return

    st.subheader("Anomaly Distribution (Final Ensemble)")
    fig, ax = plt.subplots()
    df["final_anomaly"].value_counts().plot(kind="bar", ax=ax, color=["green", "red"])
    ax.set_xticklabels(["Normal", "Anomaly"], rotation=0)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# --- 2. Model prediction comparison ---
def plot_model_predictions(df):
    required_cols = ["final_anomaly"]
    if not all(col in df.columns for col in required_cols):
        st.warning("Model predictions are missing.")
        return

    st.subheader("Model Prediction Comparison")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(
        data=df.melt(value_vars=["final_anomaly"]),
        x="variable", hue="value", ax=ax, palette="Set1"
    )
    ax.set_ylabel("Count")
    st.pyplot(fig)

# --- 3. Time series plot (if dates exist) ---
def plot_time_series(df):
    if "date" not in df.columns or df["date"].isna().all():
        st.info("No date information available for time-series plot.")
        return

    st.subheader("Anomalies Over Time")
    daily = df.groupby(df["date"].dt.date)["final_anomaly"].sum()

    fig, ax = plt.subplots()
    daily.plot(ax=ax, marker="o", color="red")
    ax.set_ylabel("Number of anomalies")
    ax.set_xlabel("Date")
    st.pyplot(fig)
