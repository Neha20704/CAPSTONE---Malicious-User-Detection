# evaluation_module.py (updated)
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, precision_score,
    recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import matplotlib.pyplot as plt
import streamlit as st


# ---------- Helper metrics ----------
def compute_metrics(y_true, y_score, threshold=None):
    """Compute standard performance metrics (used only in synthetic mode)."""
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    if threshold is None:
        precision, recall, thresh = precision_recall_curve(y_true, y_score)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_idx = np.argmax(f1)
        threshold = thresh[best_idx] if best_idx < len(thresh) else 0.5
    preds = (y_score >= threshold).astype(int)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    f1_final = 2 * (prec * rec) / (prec + rec + 1e-9)
    return {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1_final,
        "Threshold": threshold
    }


# ---------- Bootstrap CI ----------
def bootstrap_ci(metric_func, y_true, y_score, n_boot=1000, alpha=0.95):
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = resample(range(n))
        vals.append(metric_func(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(vals, [(1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100])
    return lo, hi


# ---------- Grid search for ensemble weights ----------
def grid_search_weights(y_true, ae, svm, iso, step=0.1):
    results = []
    for a, b in product(np.arange(0, 1.01, step), repeat=2):
        if a + b > 1:
            continue
        c = 1 - (a + b)
        combo = a * ae + b * svm + c * iso
        pr_auc = average_precision_score(y_true, combo)
        results.append({"AE": a, "SVM": b, "IF": c, "PR_AUC": pr_auc})
    df = pd.DataFrame(results).sort_values("PR_AUC", ascending=False)
    return df.reset_index(drop=True)


# ---------- Stacking meta-learner ----------
def stacking_meta(y_true, ae, svm, iso):
    X = np.vstack([ae, svm, iso]).T
    model = LogisticRegression(max_iter=200)
    model.fit(X, y_true)
    proba = model.predict_proba(X)[:, 1]
    return proba, model


# ---------- Evaluation master ----------
def evaluate_models(
    ae, svm, iso, nlp=None, uba=None,
    y_true=None, streamlit=False, real_world=False
):
    """
    Evaluate ensemble model either in:
      - Synthetic mode: has labels, computes metrics
      - Real-world mode: no labels, shows risk ranking & fusion plots
    """

    # === 🔧 Automatic pseudo-label generation (no leakage) ===
    # If user asked for synthetic quantitative and did not supply y_true,
    # create pseudo-labels from an independent score (UBA preferred) and
    # reduce to a test split to avoid leakage.
    if not real_world and y_true is None:
        # Use an independent score source for pseudo-labels
        if uba is not None and isinstance(uba, pd.DataFrame) and "final_risk" in uba.columns:
            pseudo = (uba["final_risk"] >= np.percentile(uba["final_risk"], 95)).astype(int)
            label_source = "UBA"
            y_true = pseudo
        elif nlp is not None and isinstance(nlp, pd.DataFrame) and "anomaly_score" in nlp.columns:
            pseudo = (nlp["anomaly_score"] >= np.percentile(nlp["anomaly_score"], 95)).astype(int)
            label_source = "NLP"
            y_true = pseudo
        else:
            raise ValueError("No valid column ('final_risk' or 'anomaly_score') found to generate pseudo-labels.")

        # Optional: split train/test to avoid overfitting (we only evaluate on test indices)
        from sklearn.model_selection import train_test_split
        idx_train, idx_test = train_test_split(
            np.arange(len(y_true)), test_size=0.3, random_state=42, shuffle=True
        )

        if streamlit:
            st.info(f"📊 Synthetic mode: Pseudo-labels generated from {label_source}, using {len(idx_test)} test samples.")
        else:
            print(f"[INFO] Synthetic mode: Pseudo-labels generated from {label_source}, using {len(idx_test)} test samples.")

        # Reduce to test subset (no leakage)
        # keep types (Series or ndarray)
        if isinstance(y_true, pd.Series):
            y_true = y_true.iloc[idx_test].reset_index(drop=True)
        else:
            y_true = np.asarray(y_true)[idx_test]

        if isinstance(nlp, pd.DataFrame):
            try:
                nlp = nlp.reset_index(drop=True).iloc[idx_test].reset_index(drop=True)
            except Exception:
                nlp = nlp.iloc[idx_test]
        if isinstance(uba, pd.DataFrame):
            try:
                uba = uba.reset_index(drop=True).iloc[idx_test].reset_index(drop=True)
            except Exception:
                uba = uba.iloc[idx_test]
        if isinstance(ae, np.ndarray):
            ae, svm, iso = ae[idx_test], svm[idx_test], iso[idx_test]

    # --- Sanity check for validation mode ---
    if not real_world and y_true is None:
        # Quantitative mode requires labels for metric computation
        raise ValueError(
            "Synthetic (Quantitative) mode selected, but no ground-truth labels (`y_true`) were provided. "
            "Provide labels to compute accuracy, precision, recall, ROC-AUC, etc."
        )

    if real_world and y_true is not None:
        # Real-world mode should not use labels — warn but proceed safely
        msg = (
            "⚠️ Real-world (Qualitative) mode is intended for unlabeled data. "
            "Since `y_true` was provided, metric computations may not be meaningful."
        )
        if streamlit:
            st.warning(msg)
        else:
            print(msg)


    def norm(x):
        return (x - np.min(x)) / (np.ptp(x) + 1e-9)

    ae, svm, iso = map(norm, [ae, svm, iso])

    # === If Real-World Mode ===
    if real_world:
        if streamlit:
            st.info("🧠 Running in Real-World (Qualitative) Validation Mode — No Ground Truth Labels Found.")

        if nlp is not None and uba is not None:
            # 🧩 Handle DataFrame inputs for real-world qualitative fusion
            if isinstance(nlp, pd.DataFrame) and isinstance(uba, pd.DataFrame):
                # Expect columns: ['from', 'anomaly_score'] and ['user', 'final_risk']
                nlp_df = nlp[["from", "anomaly_score"]].rename(columns={"from": "user"}).copy()
                uba_df = uba[["user", "final_risk"]].copy()

                # Inner merge to align common users
                merged_df = pd.merge(nlp_df, uba_df, on="user", how="inner")

                if merged_df.empty:
                    msg = "⚠️ No overlapping users between NLP and UBA datasets — cannot perform fusion."
                    if streamlit: st.warning(msg)
                    else: print(msg)
                    return {}

                if streamlit:
                    st.info(f"✅ Matched {len(merged_df)} common users between NLP ({len(nlp_df)}) and UBA ({len(uba_df)}).")

                nlp_vals = norm(merged_df["anomaly_score"].to_numpy())
                uba_vals = norm(merged_df["final_risk"].to_numpy())

            else:
                # If arrays passed instead of DataFrames, truncate to align
                min_len = min(len(nlp), len(uba))
                nlp_vals, uba_vals = map(norm, [nlp[:min_len], uba[:min_len]])
                if streamlit:
                    st.warning(f"⚠️ Length mismatch (NLP={len(nlp)}, UBA={len(uba)}). Using first {min_len} aligned elements.")

            # Combine aligned scores (weighted fusion)
            combined = 0.6 * uba_vals + 0.4 * nlp_vals

            # Display results in Streamlit
            if streamlit:
                st.subheader("🔍 Risk Ranking (NLP + UBA Fusion)")
                fusion_df = pd.DataFrame({
                    "User": merged_df["user"],
                    "NLP_Score": nlp_vals,
                    "UBA_Score": uba_vals,
                    "Risk_Score": combined
                }).sort_values("Risk_Score", ascending=False)

                st.dataframe(fusion_df.head(10))
                st.line_chart(pd.DataFrame({"Risk_Score": np.sort(combined)[::-1]}))

            return {"Fusion_Only": True}

        else:
            msg = "⚠️ No NLP/UBA scores provided for real-world ranking."
            st.warning(msg) if streamlit else print(msg)
            return {}

    # === Synthetic / Quantitative Mode ===
    if y_true is None:
        raise ValueError("y_true must be provided for synthetic (quantitative) evaluation mode.")

    # --- Step 1: grid search for weights ---
    gs = grid_search_weights(y_true, ae, svm, iso)
    best_combo = gs.iloc[0]
    if streamlit:
        st.subheader("Grid Search (AE/SVM/IF weights)")
        st.dataframe(gs.head(10))

    weighted_score = (
        best_combo.AE * ae +
        best_combo.SVM * svm +
        best_combo.IF * iso
    )
    weighted_metrics = compute_metrics(y_true, weighted_score)

    # --- Step 2: stacking ---
    stacked_score, model = stacking_meta(y_true, ae, svm, iso)
    stacked_metrics = compute_metrics(y_true, stacked_score)

    # --- Step 3: NLP + UBA Fusion (optional, synthetic mode) ---
    fusion_metrics = None
    if nlp is not None and uba is not None:
        # 🆕 Handle DataFrame alignment automatically for synthetic mode as well
        if isinstance(nlp, pd.DataFrame) and isinstance(uba, pd.DataFrame):
            # build user-level frames from inputs
            nlp_df = nlp[["from", "anomaly_score"]].rename(columns={"from": "user"}).copy()
            uba_df = uba[["user", "final_risk"]].copy()
            merged_df = pd.merge(nlp_df, uba_df, on="user", how="inner")

            if merged_df.empty:
                msg = "⚠️ No overlapping users found for fusion in synthetic mode."
                if streamlit:
                    st.warning(msg)
                else:
                    print(msg)
                fusion_metrics = {}
            else:
                if streamlit:
                    st.info(f"🆕 Aligned {len(merged_df)} common users for synthetic NLP-UBA fusion.")

                # === Fusion diagnostics ===
                overlap_count = len(merged_df)
                raw_nlp_vals = merged_df["anomaly_score"].to_numpy()
                raw_uba_vals = merged_df["final_risk"].to_numpy()
                correlation = np.corrcoef(raw_nlp_vals, raw_uba_vals)[0, 1]

                if streamlit:
                    st.markdown("### 🧩 Fusion Diagnostics")
                    st.write(f"**Overlapping users:** {overlap_count}")
                    st.write(f"**Correlation (NLP vs UBA):** `{correlation:.3f}`")

                    # Scatter with matplotlib to avoid streamlit API mismatch
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.scatter(raw_nlp_vals, raw_uba_vals, alpha=0.6)
                    ax.set_xlabel("NLP anomaly_score (raw)")
                    ax.set_ylabel("UBA final_risk (raw)")
                    ax.set_title("NLP vs UBA (raw) — overlap scatter")
                    st.pyplot(fig)
                else:
                    print("🧩 Fusion Diagnostics:")
                    print(f" - Overlapping users: {overlap_count}")
                    print(f" - Correlation (NLP vs UBA): {correlation:.3f}")

                # === Normalize before fusion ===
                nlp_vals = norm(raw_nlp_vals)
                uba_vals = norm(raw_uba_vals)

                # === Align labels to user-level if possible ===
                # If the original nlp DataFrame has message-level 'is_anomaly' labels,
                # aggregate to user-level (max) to produce a ground-truth per user.
                y_true_aligned = None
                if isinstance(nlp, pd.DataFrame) and "is_anomaly" in nlp.columns:
                    # aggregate message-level labels to user-level (presence of any anomaly => user anomalous)
                    user_labels = nlp.groupby("from")["is_anomaly"].max().reset_index().rename(columns={"from": "user"})
                    # merge with merged_df to get aligned labels in same order
                    aligned = pd.merge(merged_df[["user"]], user_labels, on="user", how="left")
                    aligned["is_anomaly"] = aligned["is_anomaly"].fillna(0).astype(int)
                    y_true_aligned = aligned["is_anomaly"].to_numpy()
                    if streamlit:
                        st.write(f"Aligned ground-truth to {len(y_true_aligned)} users (from message-level labels).")
                elif isinstance(y_true, (np.ndarray, pd.Series)) and len(y_true) >= len(merged_df):
                    # fallback: slice the provided y_true to merged size (best-effort)
                    if isinstance(y_true, pd.Series):
                        y_true_aligned = y_true.reset_index(drop=True)[:len(merged_df)].to_numpy()
                    else:
                        y_true_aligned = np.asarray(y_true)[:len(merged_df)]
                    if streamlit:
                        st.warning("Using sliced y_true to align with merged users (fallback).")
                else:
                    # cannot align labels -> skip fusion metrics but still provide diagnostics
                    if streamlit:
                        st.warning("Could not align y_true to user-level for fusion metrics — skipping numeric fusion evaluation.")
                    fusion_metrics = {}
                    y_true_aligned = None

                # If we have aligned labels, perform the weighted fusion search
                if y_true_aligned is not None:
                    best_pr = -np.inf
                    for w in np.linspace(0, 1, 11):
                        fused = w * nlp_vals + (1 - w) * uba_vals
                        pr = average_precision_score(y_true_aligned, fused)
                        if pr > best_pr:
                            best_pr = pr
                            best_weight = w
                            best_fused = fused

                    fusion_metrics = compute_metrics(y_true_aligned, best_fused)
                    fusion_metrics["NLP_WEIGHT"] = best_weight

        else:
            # 🆕 Default numeric-array alignment fallback
            nlp_arr, uba_arr = map(norm, [nlp, uba])
            min_len = min(len(nlp_arr), len(uba_arr), len(y_true))
            nlp_arr, uba_arr, y_true_trim = nlp_arr[:min_len], uba_arr[:min_len], y_true[:min_len]
            best_pr = -np.inf
            for w in np.linspace(0, 1, 11):
                fused = w * nlp_arr + (1 - w) * uba_arr
                pr = average_precision_score(y_true_trim, fused)
                if pr > best_pr:
                    best_pr = pr
                    best_weight = w
                    best_fused = fused
            fusion_metrics = compute_metrics(y_true_trim, best_fused)
            fusion_metrics["NLP_WEIGHT"] = best_weight

    # --- Step 4: bootstrap CI ---
    roc_ci = bootstrap_ci(roc_auc_score, y_true, stacked_score)
    pr_ci = bootstrap_ci(average_precision_score, y_true, stacked_score)
    print(f"Fusion overlap user count: {len(merged_df)}")
    print(f"Labels in overlapped users: {np.unique(y_true, return_counts=True)}")

    results = {
        "Best_Weights": dict(best_combo),
        "Weighted_Model": weighted_metrics,
        "Stacked_Model": stacked_metrics,
        "Fusion": fusion_metrics,
        "ROC_CI": roc_ci,
        "PR_CI": pr_ci
    }

    # --- Inside evaluate_models(), near the end ---
    if streamlit:
        st.subheader("Performance Metrics")
        st.write(results)

        # --- PRECISION–RECALL + F1 OVERLAY ---
        prec, rec, thr = precision_recall_curve(y_true, stacked_score)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
        best_idx = np.argmax(f1)
        best_f1 = f1[best_idx]
        best_threshold = thr[best_idx] if best_idx < len(thr) else 0.5

        st.subheader("📈 Precision–Recall–F1 Curve")
        st.caption("Shows trade-off between precision and recall, with F1-score overlay.")
        st.line_chart(pd.DataFrame({
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1
        }))

        # Display optimal F1 threshold and score
        st.markdown(
            f"**Optimal Threshold (by F1 peak):** `{best_threshold:.3f}` "
            f" | **Best F1:** `{best_f1:.3f}`"
        )

        # --- ROC CURVE + RANDOM BASELINE ---
        fpr, tpr, _ = roc_curve(y_true, stacked_score)
        st.subheader("📊 ROC Curve (True vs False Positive Rate)")
        st.line_chart(pd.DataFrame({
            "TPR (Recall)": tpr,
            "FPR": fpr
        }))

    return results
