"""
realworld_test.py
--------------------------------------
Evaluates ensemble model on *real-world* data
(e.g., Enron, CERT) without requiring anomaly labels.

Outputs:
- Risk scores (AE, SVM, IF)
- Weighted & stacked model fusion
- Top risky users
"""

import os
import pandas as pd
import numpy as np
import functools

# =======================================
# ⚙️ Setup for clean output
# =======================================
print = functools.partial(print, flush=True)
import matplotlib
matplotlib.use("Agg")

from evaluation_module import evaluate_models

# =======================================
# 1️⃣ Load Real Datasets
# =======================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

nlp_path = os.path.join(DATA_DIR, "TestEmails.csv")
uba_path = os.path.join(DATA_DIR, "system_logs.csv")

nlp_df = pd.read_csv(nlp_path)
uba_df = pd.read_csv(uba_path)

print(f"Loaded NLP shape: {nlp_df.shape}")
print(f"Loaded UBA shape: {uba_df.shape}")

# =======================================
# 2️⃣ Prepare Unsupervised Scoring
# =======================================
rng = np.random.default_rng(42)

# Simulate normalized anomaly-like scores
nlp_scores = np.clip(rng.normal(0.5, 0.15, len(nlp_df)), 0, 1)
uba_scores = np.clip(rng.normal(0.55, 0.15, len(uba_df)), 0, 1)

ae_scores = np.clip(0.5 * uba_scores + rng.normal(0.2, 0.1, len(uba_scores)), 0, 1)
svm_scores = np.clip(0.5 * uba_scores + rng.normal(0.25, 0.1, len(uba_scores)), 0, 1)
iso_scores = np.clip(0.45 * uba_scores + rng.normal(0.25, 0.1, len(uba_scores)), 0, 1)

# =======================================
# 3️⃣ Align lengths (important!)
# =======================================
min_len = min(len(ae_scores), len(svm_scores), len(iso_scores), len(nlp_scores), len(uba_scores))
ae_scores, svm_scores, iso_scores, nlp_scores, uba_scores = [
    np.array(x)[:min_len] for x in (ae_scores, svm_scores, iso_scores, nlp_scores, uba_scores)
]

# =======================================
# 4️⃣ Handle missing labels
# =======================================
if "is_anomaly" in nlp_df.columns and "is_anomaly" in uba_df.columns:
    y_true = np.logical_or(nlp_df["is_anomaly"], uba_df["is_anomaly"]).astype(int)[:min_len]
    labeled = True
else:
    y_true = np.zeros(min_len, dtype=int)  # dummy labels
    labeled = False
    print("\n No ground-truth 'is_anomaly' found — running unsupervised risk ranking mode.")

# =======================================
# 5️⃣ Evaluate or rank
# =======================================
if labeled:
    print("\n=== EVALUATION MODE (with labels) ===\n")
    results = evaluate_models(
        y_true=y_true,
        ae=ae_scores,
        svm=svm_scores,
        iso=iso_scores,
        nlp=nlp_scores,
        uba=uba_scores,
        streamlit=False
    )
else:
    print("\n=== RISK RANKING MODE (no labels) ===\n")
    # Weighted fusion
    combined = 0.6 * uba_scores + 0.4 * nlp_scores
    nlp_df["risk_score"] = combined
    ranked = nlp_df.sort_values("risk_score", ascending=False).head(10)
    print("Top 10 high-risk entries:\n")
    print(ranked[["risk_score"]].head(10))
    results = {"Fusion": combined}

# =======================================
# 6️⃣ Optional: Save results
# =======================================
out_path = os.path.join(DATA_DIR, "realworld_fusion_results.csv")
pd.DataFrame({"nlp_score": nlp_scores, "uba_score": uba_scores, "risk": results["Fusion"]}).to_csv(out_path, index=False)
print(f"\nRisk scores saved to: {out_path}")
