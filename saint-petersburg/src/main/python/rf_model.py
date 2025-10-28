# rf_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

class RFModel:
    def __init__(self, n_estimators=200, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False

    def train(self, csv_path, target_col="is_winner"):
        df = pd.read_csv(csv_path)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.model.fit(X, y)
        self.is_trained = True
        return "Model trained on {} samples".format(len(df))

    def predict(self, features):
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        X = np.array(features).reshape(1, -1)
        proba = self.model.predict_proba(X)[0, 1]
        return float(proba)


    def save(self, path="rf_model.joblib"):
        joblib.dump(self.model, path)
        return f"Model saved to {path}"

    def load(self, path="rf_model.joblib"):
        self.model = joblib.load(path)
        self.is_trained = True
        return f"Model loaded from {path}"
