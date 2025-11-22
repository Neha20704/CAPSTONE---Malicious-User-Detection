# train_local.py
import os
import pickle
import pandas as pd
from src.preprocessing import extract_features
from sklearn.ensemble import RandomForestClassifier

# --- Load dataset ---
df = pd.read_csv("data/enron_recleaned.csv")

# --- Extract features ---
df, features, vectorizer, imputer = extract_features(df, fit_vectorizer=True)

# --- Train model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(features, df["label"])  # assuming "label" column exists

# --- Save everything in models_2/ ---
os.makedirs("models_2", exist_ok=True)

with open("models_2/random_forest_insider.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models_2/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models_2/simple_imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

# Save feature column order for inference
with open("models_2/feature_columns.pkl", "wb") as f:
    pickle.dump(features.columns.tolist(), f)

print("✅ Models and preprocessing objects saved in models_2/")
