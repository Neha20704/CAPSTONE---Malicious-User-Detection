"""
synthetic_eval_test.py
----------------------------------
Standalone test harness for your evaluate_models() function.

It simulates anomaly detection model outputs (AE, OCSVM, IF)
plus optional NLP/UBA fusion vectors, and runs the full
grid search + stacking + bootstrap CI evaluation.
"""

import numpy as np
import pandas as pd
from evaluation_module import evaluate_models

# --- CONFIG ---
N_SAMPLES = 500
ANOMALY_RATE = 0.1  # 10% anomalies

# --- Step 1: Synthetic ground truth labels ---
y_true = np.zeros(N_SAMPLES, dtype=int)
anomaly_idx = np.random.choice(N_SAMPLES, int(N_SAMPLES * ANOMALY_RATE), replace=False)
y_true[anomaly_idx] = 1

# --- Step 2: Simulate model anomaly scores ---
# Each model has different “signal quality”
rng = np.random.default_rng(42)

# AE tends to separate well (high recall)
ae_score = rng.normal(0.5, 0.1, N_SAMPLES)
ae_score[anomaly_idx] += rng.uniform(0.3, 0.5, size=len(anomaly_idx))

# OCSVM tends to overflag a bit
svm_score = rng.normal(0.4, 0.15, N_SAMPLES)
svm_score[anomaly_idx] += rng.uniform(0.2, 0.4, size=len(anomaly_idx))

# Isolation Forest is conservative
iso_score = rng.normal(0.45, 0.1, N_SAMPLES)
iso_score[anomaly_idx] += rng.uniform(0.25, 0.35, size=len(anomaly_idx))

# Clip to [0,1]
ae_score = np.clip(ae_score, 0, 1)
svm_score = np.clip(svm_score, 0, 1)
iso_score = np.clip(iso_score, 0, 1)

# --- Step 3: Optional NLP + UBA fusion scores ---
nlp_score = rng.normal(0.4, 0.1, N_SAMPLES)
nlp_score[anomaly_idx] += rng.uniform(0.3, 0.4, size=len(anomaly_idx))
nlp_score = np.clip(nlp_score, 0, 1)

uba_score = rng.normal(0.45, 0.1, N_SAMPLES)
uba_score[anomaly_idx] += rng.uniform(0.25, 0.35, size=len(anomaly_idx))
uba_score = np.clip(uba_score, 0, 1)

# --- Step 4: Evaluate ---
results = evaluate_models(
    y_true=y_true,
    ae=ae_score,
    svm=svm_score,
    iso=iso_score,
    nlp=nlp_score,
    uba=uba_score,
    streamlit=False  # set True if testing inside Streamlit
)

print("\n=== SYNTHETIC EVALUATION SUMMARY ===")
print(pd.DataFrame(results["Best_Weights"], index=[0]))
print("\nWeighted Model:", results["Weighted_Model"])
print("\nStacked Model:", results["Stacked_Model"])
if results["Fusion"] is not None:
    print("\nFusion Metrics:", results["Fusion"])
print("\nROC_AUC 95% CI:", results["ROC_CI"])
print("PR_AUC  95% CI:", results["PR_CI"])
