import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def compare_ml_models(file_path, feature_list, target_col="is_winner"):
    """
    Compare multiple ML models to predict the target column.
    
    Args:
        file_path (str): Path to CSV dataset.
        feature_list (list[str]): Features to use for prediction.
        target_col (str): Target column name.

    Returns:
        dict: Best model summary with algorithm name, metrics, and fitted model.
    """
    print("📂 Loading dataset...")
    df = pd.read_csv(file_path)
    
    X = df[feature_list]
    y = df[target_col]
    
    print(f"✅ Loaded {len(df)} rows, {len(feature_list)} features.")
    print(f"🎯 Target column: {target_col}\n")

    # Define models to compare
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    }

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    print("🚀 Starting model comparisons...\n")

    for name, model in models.items():
        print(f"🔹 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
        
        print(f"   Accuracy: {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall: {rec:.4f}")
        print(f"   F1-score: {f1:.4f}")
        print(f"   Cross-val (5-fold): {cv_score:.4f}\n")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "CrossVal": cv_score,
            "FittedModel": model
        })

    # Pick the best based on F1-score (you can switch to accuracy if you prefer)
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["F1"].idxmax()]

    print("🏁 Comparison complete!\n")
    print("📊 Summary:")
    print(results_df[["Model", "Accuracy", "Precision", "Recall", "F1", "CrossVal"]].sort_values(by="F1", ascending=False))
    print("\n🏆 Best Model:", best["Model"])
    print(f"   Accuracy: {best['Accuracy']:.4f}")
    print(f"   F1-score: {best['F1']:.4f}")

    return {
        "best_model_name": best["Model"],
        "best_model": best["FittedModel"],
        "metrics": best.to_dict(),
        "all_results": results_df
    }

# Example usage:
best_result = compare_ml_models("best model to date/SPTrainingData.csv", ['rubles_round_gain_diff_x_min_deck_size', 'rubles_round_gain_diff', 'points_diff', 'unique_aristocrats_points_diff', 'points_round_gain_diff', 'rubles_diff', 'points_round_gain_diff_x_min_deck_size', 'rubles_diff_x_min_deck_size', 'unique_aristocrats_points_diff_x_min_deck_size', 'points_diff_x_min_deck_size', 'rubles_round_gain', 'rubles_round_gain_x_min_deck_size', 'points', 'points_round_gain'], "is_winner")
