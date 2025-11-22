"""
evaluate_synthetic_v2.py
--------------------------------------
Runs ensemble evaluation (grid search + stacking + bootstrap CI)
on the synthetic NLP and UBA datasets.

Requires:
- nlp_synthetic_v2.csv   (from nlp_synthetic_v2.py)
- uba_synthetic_v2.csv   (from uba_synthetic_v2.py)
- evaluation_module.py
"""

import os
import functools
import pandas as pd
import numpy as np

# =======================================
# ⚙️ Setup for headless + immediate output
# =======================================
# Force all prints to flush immediately
print = functools.partial(print, flush=True)

# Use a non-interactive backend for matplotlib
import matplotlib
matplotlib.use("Agg")

from evaluation_module import evaluate_models

# =======================================
# 1️⃣ Load Synthetic Datasets
# =======================================
# Resolve paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> src
ROOT_DIR = os.path.dirname(BASE_DIR)                                   # -> project root
TEST_DIR = os.path.join(ROOT_DIR, "testing")
DATA_DIR = os.path.join(ROOT_DIR,"data")
nlp_path = os.path.join(DATA_DIR, "TestEmails.csv")
uba_path = os.path.join(DATA_DIR, "system_logs.csv")

if not os.path.exists(nlp_path) or not os.path.exists(uba_path):
    raise FileNotFoundError(f"❌ Could not find NLP/UBA synthetic files in {TEST_DIR}")

nlp_df = pd.read_csv(nlp_path)
uba_df = pd.read_csv(uba_path)

# Align user counts
n = min(len(nlp_df), len(uba_df))
nlp_df = nlp_df.sample(n, random_state=42).reset_index(drop=True)
uba_df = uba_df.sample(n, random_state=42).reset_index(drop=True)

# Ground truth labels (1 if either is anomalous)
y_true = np.logical_or(nlp_df["is_anomaly"], uba_df["is_anomaly"]).astype(int)

# =======================================
# 2️⃣ Generate Synthetic Model Scores
# =======================================
rng = np.random.default_rng(42)

ae_scores = np.clip(0.6 * y_true + rng.normal(0.2, 0.15, n), 0, 1)
svm_scores = np.clip(0.55 * y_true + rng.normal(0.25, 0.15, n), 0, 1)
iso_scores = np.clip(0.5 * y_true + rng.normal(0.25, 0.15, n), 0, 1)

nlp_scores = np.clip(0.65 * nlp_df["is_anomaly"] + rng.normal(0.25, 0.15, n), 0, 1)
uba_scores = np.clip(0.6 * uba_df["is_anomaly"] + rng.normal(0.25, 0.15, n), 0, 1)

# =======================================
# 3️⃣ Evaluate Models
# =======================================
print("\n=== SYNTHETIC ENSEMBLE EVALUATION (v2) ===\n")

results = evaluate_models(
    y_true=y_true,
    ae=ae_scores,
    svm=svm_scores,
    iso=iso_scores,
    nlp=nlp_scores,
    uba=uba_scores,
    streamlit=False
)

# =======================================
# 4️⃣ Display Summary
# =======================================
print("=== GRID SEARCH (AE/SVM/IF weights) ===")
for k, v in results["Best_Weights"].items():
    print(f"{k}: {v}")

print("\n--- Weighted Model ---")
for k, v in results["Weighted_Model"].items():
    print(f"{k}: {v}")

print("\n--- Stacked Model ---")
for k, v in results["Stacked_Model"].items():
    print(f"{k}: {v}")

if results["Fusion"] is not None:
    print("\n--- Fusion (NLP + UBA) ---")
    for k, v in results["Fusion"].items():
        print(f"{k}: {v}")

print(f"\nROC_AUC 95% CI: {results['ROC_CI']}")
print(f"PR_AUC  95% CI: {results['PR_CI']}")

# =======================================
# 5️⃣ Save Plots (non-blocking)
# =======================================
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

prec, rec, _ = precision_recall_curve(y_true, iso_scores)
plt.figure(figsize=(5, 4))
plt.plot(rec, prec, label="Stacked PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Synthetic Ensemble)")
plt.legend()
plt.tight_layout()

plot_path = os.path.join(TEST_DIR, "synthetic_eval_prcurve_v2.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")

print(f"\nEvaluation complete. Plot saved -> {plot_path}")
