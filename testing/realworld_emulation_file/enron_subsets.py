import pandas as pd
import numpy as np
nlp_df = pd.read_csv("../../data/enron_recleaned.csv")
uba_df = pd.read_csv("../../data/system_logs.csv")

# 1. Randomly pick 20 users that appear in both
common_users = list(set(nlp_df["from"]).intersection(set(uba_df["user"])))
subset_users = np.random.choice(common_users, 20, replace=False)

# 2. Filter both datasets to only those users
nlp_subset = nlp_df[nlp_df["from"].isin(subset_users)].copy()
uba_subset = uba_df[uba_df["user"].isin(subset_users)].copy()

# 3. Add mock anomaly scores (if missing)
nlp_subset["anomaly_score"] = np.clip(np.random.normal(0.4, 0.2, len(nlp_subset)), 0, 1)
uba_subset["final_risk"] = np.clip(np.random.normal(0.5, 0.25, len(uba_subset)), 0, 1)

# 4. Save and use
nlp_subset.to_csv("real_subset_nlp.csv", index=False)
uba_subset.to_csv("real_subset_uba.csv", index=False)
