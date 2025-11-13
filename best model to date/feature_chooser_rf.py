from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def find_best_features(file_path, target_col="is_winner", max_features=10):
    """
    Find the optimal number and combination of top features that maximize prediction accuracy.
    
    Args:
        file_path (str): Path to the CSV file.
        target_col (str): Target column name (default: "is_winner").
        max_features (int): Number of top features to consider.
    
    Returns:
        tuple: (best_num_features, best_features_list, best_score)
    """
    # Load data
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Base model
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Rank features by importance
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    ranked_features = importances.sort_values(ascending=False)

    best_score = 0
    best_features = []
    best_num = 0

    print(ranked_features)

    # Test from 1 to max_features
    for i in range(5, max_features + 1):
        selected = ranked_features.index[:i]
        score = cross_val_score(rf, X[selected], y, cv=3, scoring='accuracy', n_jobs=-1).mean()
        print(f"{i} features: accuracy = {score:.4f}")
        if score > best_score:
            best_score = score
            best_features = selected.tolist()
            best_num = i

    print("\n✅ Best feature set:")
    print(f"Number of features: {best_num}")
    print(f"Features: {best_features}")
    print(f"Accuracy: {best_score:.4f}")

    return best_num, best_features, best_score


# Example usage
find_best_features("AIDanSPTrainingDataFlatMCvsFlatMC.csv", target_col="is_winner", max_features=27)