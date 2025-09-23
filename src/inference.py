from src.preprocessing import clean_text, extract_features

# Example new email
data = {
    "from": ["employee@enron.com"],
    "to": ["external@unknown.com"],
    "cc": [""],
    "bcc": [""],
    "date": ["2001-06-15 23:50:00"],
    "subject": ["Confidential report"],
    "cleaned_message": [clean_text("Please find attached confidential report, do not share.")]
}

new_df = pd.DataFrame(data)

# Extract features (using existing vectorizer + imputer)
_, features, _, _ = extract_features(
    new_df,
    fit_vectorizer=False,
    vectorizer=vectorizer,
    imputer=imputer
)

# 🔸 Isolation Forest
iso_pred = iso.predict(features)   # -1 = anomaly, 1 = normal

# 🔸 One-Class SVM
features_scaled = scaler.transform(features)
svm_pred = ocsvm.predict(features_scaled)  # -1 = anomaly, 1 = normal

# 🔸 Autoencoder
reconstructions = autoencoder.predict(features_scaled)
mse = ((features_scaled - reconstructions) ** 2).mean(axis=1)
threshold = mse.mean() + 2*mse.std()  # heuristic threshold
auto_pred = (mse > threshold).astype(int)  # 1 = anomaly, 0 = normal

print("Isolation Forest:", iso_pred)
print("One-Class SVM:", svm_pred)
print("Autoencoder:", auto_pred)
