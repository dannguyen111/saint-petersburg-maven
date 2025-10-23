import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

class GameStateRegressor:
    def __init__(self, df, feature_cols=None, target_col="rounds_left"):
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame is empty or None.")

            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

            if feature_cols is None:
                feature_cols = [c for c in df.columns if c != target_col]

            self.feature_cols = feature_cols
            self.X = df[feature_cols]
            self.y = df[target_col]

            if self.X.isnull().any().any() or self.y.isnull().any():
                raise ValueError("Missing values detected in feature or target columns.")

            # Initialize and fit CatBoost model
            self.model = CatBoostRegressor(
                depth=5,
                learning_rate=0.1,
                iterations=500,
                loss_function='RMSE',
                verbose=False
            )
            self.model.fit(self.X, self.y)
            print("[INFO] Model successfully trained.")
        
        except Exception as e:
            print(f"[ERROR][GameStateRegressor.__init__]: {e}")
            self.model = None
            self.feature_cols = []
            self.X = None
            self.y = None

    def query(self, query_data, rounds=None):
        try:
            if self.model is None:
                raise RuntimeError("Model not initialized. Cannot perform query.")

            # Convert from Java ArrayList or Python list to proper 2D array
            query_data = np.array(query_data, dtype=float)
            if query_data.ndim == 1:
                query_data = query_data.reshape(1, -1)

            # Ensure feature count matches model
            if query_data.shape[1] != len(self.feature_cols):
                raise ValueError(
                    f"Query data has {query_data.shape[1]} features, expected {len(self.feature_cols)}"
                )

            prediction = self.model.predict(query_data)
            
            # if rounds:
            #     total_round_pred = round(float(prediction[0]) + rounds)
            #     return total_round_pred - rounds
            # else:
            return float(prediction[0])

        except Exception as e:
            print(f"[ERROR][GameStateRegressor.query]: {e}")
            return None
        
if __name__ == "__main__":
    df = pd.read_csv("AIDanSPRoundsLeftTrainingData.csv")
    feature_cols = df.columns.tolist()
    model = GameStateRegressor(df, feature_cols)
    test_query = df[feature_cols].iloc[2857].to_numpy()
    print("Predicted rounds left:", model.query(test_query, k=5))

