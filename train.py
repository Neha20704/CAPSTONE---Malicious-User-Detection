import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.preprocessing import extract_features

# ================================
# 1. Load Data
# ================================
DATA_PATH = "data/enron_cleaned.csv"   # <-- update if your file is elsewhere
df = pd.read_csv(DATA_PATH)

print(f"Loaded dataset with {len(df)} emails")

# ================================
# 2. Preprocess & Feature Extraction
# ================================
df, features, vectorizer, imputer = extract_features(df, fit_vectorizer=True)

# ================================
# 3. Unsupervised: Isolation Forest
# ================================
iso_model = IsolationForest(contamination=0.05, random_state=42)
df["anomaly_score"] = iso_model.fit_predict(features)

print("IsolationForest anomaly counts:")
print(df["anomaly_score"].value_counts())

# ================================
# 4. Supervised: Random Forest
# ================================
# Create pseudo labels
df["label"] = (
    (df["num_bcc"] > 5) |
    (df["is_off_hours"] & (df["threat_keyword_count"] > 0))
).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    features, df["label"], test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred))

# ================================
# 5. Save Models
# ================================
os.makedirs("models", exist_ok=True)

joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
joblib.dump(imputer, "models/simple_imputer.pkl")
joblib.dump(iso_model, "models/isolation_forest_insider.pkl")
joblib.dump(rf_model, "models/random_forest.pkl")

print("\n✅ Models saved in /models folder")
