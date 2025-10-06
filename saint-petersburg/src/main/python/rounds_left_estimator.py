import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class GameStateKNN:
    def __init__(self, df, feature_cols=None, target_col="rounds_left", metric="euclidean"):
        """
        df: DataFrame containing game data
        feature_cols: list of columns to use as features
        target_col: column with the target value (rounds_left)
        metric: distance metric (default: euclidean)
        """
        try:
            if df.empty:
                raise ValueError("DataFrame is empty.")
            # if not all(col in df.columns for col in feature_cols):
            #     raise ValueError(f"Some feature columns are missing: {feature_cols}")
            
            # df = add_rounds_left(df)

            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
            
            if feature_cols:
                self.features = df[feature_cols].to_numpy()
            else:
                feature_cols = df.columns.tolist()
                feature_cols.remove(target_col)
                self.features = df[feature_cols].to_numpy()
    
            self.targets = df[target_col].to_numpy()

            if len(self.features) == 0 or len(self.targets) == 0:
                raise ValueError("Features or target data are empty.")

            # Fit NearestNeighbors model
            self.nn = NearestNeighbors(metric=metric)
            self.nn.fit(self.features)

        except Exception as e:
            print(f"[GameStateKNN.__init__] Error during initialization: {e}")
            self.nn = None
            self.features = None
            self.targets = None

    def query(self, query_data, k=5):
        """
        query_data: np.array or list of shape (n_features,) or (n_queries, n_features)
        k: number of neighbors
        Returns average rounds_left of neighbors
        """
        try:
            if self.nn is None or self.features is None or self.targets is None:
                raise RuntimeError("Model not properly initialized.")

            query_data = np.atleast_2d(query_data)  # ensure 2D
            if query_data.shape[1] != self.features.shape[1]:
                raise ValueError(
                    f"Query data has {query_data.shape[1]} features, expected {self.features.shape[1]}"
                )

            distances, indices = self.nn.kneighbors(query_data, n_neighbors=k)

            # Look up targets for neighbors
            neighbor_targets = self.targets[indices]
            return float(neighbor_targets.mean(axis=1)[0])

        except Exception as e:
            print(f"[GameStateKNN.query] Error during query: {e}")
            return None
        
    def print_features(self):
        if self.features is not None:
            return str(self.features)
        else:
            return "Features not available."

def add_rounds_left(df):
    df = df.copy()

    new_game = df['round'] < df['round'].shift(fill_value=0)
    df['game_id'] = new_game.cumsum()
    max_rounds = df.groupby('game_id')['round'].transform('max')
    df['rounds_left'] = max_rounds - df['round']

    return df


if __name__ == "__main__":
    df = pd.read_csv("AIDanSPRoundsLeftTrainingData.csv")
    feature_cols = df.columns.tolist()
    model = GameStateKNN(df, feature_cols)
    test_query = df[feature_cols].iloc[2857].to_numpy()
    print("Predicted rounds left:", model.query(test_query, k=5))


# cat boost