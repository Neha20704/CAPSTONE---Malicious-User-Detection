import pickle
import pandas as pd
from tensorflow import keras

# --- Load preprocessing objects ---
with open("models/models_2/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/models_2/simple_imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

with open("models/models_2/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# --- Load models ---
with open("models/models_2/isolation_forest.pkl", "rb") as f:
    iso = pickle.load(f)

with open("models/models_2/ocsvm.pkl", "rb") as f:
    ocsvm = pickle.load(f)

with open("models/models_2/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

autoencoder = keras.models.load_model("models/models_2/autoencoder_model.keras")
