import joblib
import pandas as pd

class ThreatDetector:
    def __init__(self, model_paths):
        self.rf = joblib.load(model_paths["rf"])
        self.iso = joblib.load(model_paths["iso"])
        self.vectorizer = joblib.load(model_paths["vectorizer"])
        self.imputer = joblib.load(model_paths["imputer"])

    def predict(self, features: pd.DataFrame):
        X = self.imputer.transform(features)
        preds_rf = self.rf.predict(X)
        preds_iso = self.iso.predict(X)
        return preds_rf, preds_iso
