import pickle
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# -----------------------------
# Load models once
# -----------------------------
def load_all_models(base_path="../models/models_2"):
    """
    Loads all trained models into memory.
    Returns: dict of model objects.
    """
    models = {}

    # Isolation Forest
    with open(f"{base_path}/isolation_forest.pkl", "rb") as f:
        models["isolation_forest"] = pickle.load(f)

    # OCSVM
    with open(f"{base_path}/ocsvm.pkl", "rb") as f:
        models["ocsvm"] = pickle.load(f)

    # Autoencoder
    models["autoencoder"] = tf.keras.models.load_model(f"{base_path}/autoencoder_model.keras")

    return models


# -----------------------------
# Model Predictions
# -----------------------------
def predict_with_models(models, X):
    """
    Runs inference with all models and returns individual + ensemble predictions.
    Returns:
        - results: dict with each model's raw prediction
        - final_preds: array of final majority-voted predictions (1 = anomaly, 0 = normal)
    """
    results = {}

    # Isolation Forest → -1 anomaly, 1 normal
    iso_pred = models["isolation_forest"].predict(X)
    results["isolation_forest"] = np.where(iso_pred == -1, 1, 0)

    # OCSVM → -1 anomaly, 1 normal
    ocsvm_pred = models["ocsvm"].predict(X)
    results["ocsvm"] = np.where(ocsvm_pred == -1, 1, 0)

    # Autoencoder → reconstruction error threshold
    reconstructions = models["autoencoder"].predict(X, verbose=0)
    errors = np.mean(np.square(X - reconstructions), axis=1)
    threshold = np.percentile(errors, 95)  # can be tuned
    autoencoder_pred = (errors > threshold).astype(int)
    results["autoencoder"] = autoencoder_pred

    # -----------------------------
    # Majority Voting
    # -----------------------------
    all_preds = np.vstack(list(results.values()))
    votes = np.sum(all_preds, axis=0)

    # If majority voted anomaly → 1 else 0
    final_preds = (votes >= 2).astype(int)

    return results, final_preds, errors


# -----------------------------
# Human-readable mapping
# -----------------------------
def prediction_summary(final_preds, errors):
    """
    Creates a clean summary table for visualization or output.
    """
    summary = []
    for i, pred in enumerate(final_preds):
        summary.append({
            "Index": i,
            "Result": "Anomaly" if pred == 1 else "Normal",
            "Anomaly_Score": round(errors[i], 5)
        })
    return summary
